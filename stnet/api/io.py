# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Protocol

import torch
from torch import nn

try:
    from torch.serialization import add_safe_globals as _add_safe_globals
except ImportError:
    _add_safe_globals = None

if _add_safe_globals is not None:
    with contextlib.suppress(Exception):
        import torch.torch_version as _torch_version

        _add_safe_globals([_torch_version.TorchVersion])

from torch.distributed.checkpoint import FileSystemReader, load as dcp_load
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from ..model import Root
from ..api.config import ModelConfig, coerce_model_config
from ..functional.fx import Fusion, Gradient


class MissingDependencyError(ImportError):
    pass


def _require(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise MissingDependencyError(f"{module} is required for this operation{hint}") from err


def _to_cpu_if_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    return value


def _run_cmd(cmd: Sequence[str], desc: str) -> None:
    try:
        subprocess.run(list(cmd), check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"{desc} failed: {e}") from e


class _ExportCompat(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_flat, _ = self.net(x, labels_flat=None, net_loss=None)
        return y_flat


def _infer_tensor_shape(model: nn.Module, sample_input: Optional[torch.Tensor]) -> Tuple[int, Tuple[int, ...]]:
    in_dim: Optional[int] = None
    out_shape: Optional[Tuple[int, ...]] = None
    if hasattr(model, "in_dim"):
        try:
            in_dim = int(getattr(model, "in_dim"))
        except (TypeError, ValueError):
            in_dim = None
    if hasattr(model, "out_shape"):
        try:
            _shape = getattr(model, "out_shape")
            out_shape = tuple(int(x) for x in _shape)
        except (TypeError, ValueError):
            out_shape = None
    if (in_dim is None or out_shape is None) and sample_input is not None:
        dev = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
        sample = sample_input.to(dev)
        model.eval()
        with Gradient.inference(model):
            if sample.ndim == 1:
                sample = sample.unsqueeze(0)
            y_flat, _ = model(sample, labels_flat=None, net_loss=None)
        if in_dim is None:
            in_dim = int(sample.numel() // int(sample.shape[0]))
        if out_shape is None:
            out_shape = tuple(int(x) for x in y_flat.shape[1:])
            if len(out_shape) == 1:
                out_shape = (out_shape[0],)
    if in_dim is None or out_shape is None:
        raise RuntimeError("failed to infer I/O shapes")
    return int(in_dim), tuple(out_shape)


def _pad_sample(model: nn.Module, sample_input: Optional[torch.Tensor]) -> torch.Tensor:
    if sample_input is not None:
        return sample_input
    in_dim, _ = _infer_tensor_shape(model, sample_input)
    try:
        p = next(model.parameters())
        dtype, device = (p.dtype, p.device)
    except StopIteration:
        dtype, device = (torch.float32, torch.device("cpu"))
    return torch.zeros(1, in_dim, dtype=dtype, device=device)


def _model_config_dict(model: nn.Module) -> Dict[str, Any]:
    cfg_obj = getattr(model, "_Root__config", None)
    if cfg_obj is None:
        cfg_obj = getattr(model, "__stnet_root_config__", None)
    if cfg_obj is None:
        for submodule in model.modules():
            cfg_obj = getattr(submodule, "_Root__config", None)
            if cfg_obj is not None:
                break
    if isinstance(cfg_obj, ModelConfig):
        return asdict(coerce_model_config(cfg_obj))
    if isinstance(cfg_obj, dict):
        return asdict(coerce_model_config(cfg_obj))
    return asdict(ModelConfig())


class TorchIO:
    NATIVE_EXTS = {".pt", ".pth", ".safetensors"}

    @staticmethod
    def is_native_target(path: str | Path) -> bool:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix:
            return suffix in TorchIO.NATIVE_EXTS
        return True

    @staticmethod
    def save(
        model: nn.Module,
        path: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None,
        *args: Any,
        **opts: Any,
    ) -> Path:
        p = Path(path)
        suffix = p.suffix.lower()

        if not suffix:
            if p.exists():
                if p.is_file():
                    suffix = ".pt"
                elif p.is_dir():
                    from torch.distributed.checkpoint import save as dcp_save, FileSystemWriter

                    opts_sd = StateDictOptions(full_state_dict=True)
                    m_sd = get_model_state_dict(model, options=opts_sd)
                    dcp_save(
                        state_dict={"model": m_sd},
                        storage_writer=FileSystemWriter(str(p)),
                    )
                    return p

        p.parent.mkdir(parents=True, exist_ok=True)

        if suffix == ".safetensors":
            _require("safetensors", "pip install safetensors")
            from safetensors.torch import save_file as save_tensors

            sd = model.state_dict()
            cpu_sd = {k: _to_cpu_if_tensor(v) for k, v in sd.items()}
            save_tensors(cpu_sd, str(p), metadata={"format": "safetensors-v1"})
            meta = {
                "version": 1,
                "in_dim": int(getattr(model, "in_dim", 0)),
                "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
                "config": _model_config_dict(model),
                "pytorch_version": torch.__version__,
                "extra": extra or {},
            }
            p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return p

        payload: Dict[str, Any] = {
            "version": 1,
            "in_dim": int(getattr(model, "in_dim", 0)),
            "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
            "config": _model_config_dict(model),
            "state_dict": model.state_dict(),
            "pytorch_version": torch.__version__,
        }
        if optimizer is not None and hasattr(optimizer, "state_dict"):
            try:
                payload["optimizer_state_dict"] = optimizer.state_dict()
            except Exception:
                pass
        if extra:
            payload["extra"] = extra
        torch.save(payload, str(p))
        return p

    @staticmethod
    def load_state(path: str | Path, map_location: Optional[str] = None) -> Dict[str, Any] | torch.Tensor:
        p = Path(path)
        if p.is_dir():
            raise RuntimeError("Use DCP-specific load with a pre-allocated model and state_dict")
        if p.suffix.lower() == ".safetensors":
            _require("safetensors", "pip install safetensors")
            from safetensors.torch import load_file as load_tensors
            return load_tensors(str(p), device=map_location or "cpu")
        load_kwargs = {"map_location": map_location or "cpu"}
        try:
            return torch.load(str(p), weights_only=True, **load_kwargs)
        except TypeError:
            return torch.load(str(p), **load_kwargs)


class ConverterBase(Protocol):
    name: str
    def convert(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]: ...


class Converter:
    _by_name: Dict[str, ConverterBase] = {}
    _ext_map: Dict[str, str] = {}

    class _OnnxLayer:
        @staticmethod
        def export(
            model: nn.Module,
            onnx_path: Path,
            *args: Any,
            sample_input: Optional[torch.Tensor] = None,
            opset_version: int = 18,
            dynamic_batch: bool = True,
            **kwargs: Any,
        ) -> Path:
            _require("onnx", "pip install onnx")
            wrapper = _ExportCompat(model).eval()
            sample = _pad_sample(model, sample_input)
            input_names = ["features"]
            output_names = ["preds_flat"]
            dynamic_axes = (
                {"features": {0: "batch"}, "preds_flat": {0: "batch"}}
                if dynamic_batch
                else None
            )
            export_error = getattr(torch.onnx, "OnnxExporterError", RuntimeError)
            fallback_errors = (
                (RuntimeError,)
                if export_error is RuntimeError
                else (RuntimeError, export_error)
            )
            common_kwargs = {
                "export_params": True,
                "opset_version": opset_version,
                "do_constant_folding": True,
                "input_names": input_names,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
            }
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.onnx.export(
                    wrapper,
                    sample,
                    str(onnx_path),
                    dynamo=True,
                    **common_kwargs,
                )
            except fallback_errors:
                torch.onnx.export(
                    wrapper,
                    sample,
                    str(onnx_path),
                    dynamo=False,
                    **common_kwargs,
                )
            return onnx_path

        @staticmethod
        def ensure(model: nn.Module, onnx_path: Path, *args: Any, **opts: Any) -> Path:
            if not onnx_path.exists():
                return Converter._OnnxLayer.export(model, onnx_path, **opts)
            return onnx_path

    class _OrtLayer:
        @staticmethod
        def save_ort(
            onnx_path: Path,
            ort_path: Path,
            *args: Any,
            optimization_level: str = "all",
            optimization_style: str = "fixed",
            target_platform: Optional[str] = None,
            save_optimized_onnx_model: bool = False,
            **kwargs: Any,
        ) -> Tuple[Path, Optional[Path]]:
            _require("onnxruntime", "pip install onnxruntime")
            import onnxruntime as ort

            opt_map = {
                "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            level = opt_map.get(
                optimization_level.lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            platform = (target_platform or "").lower()
            disabled_optimizers = (
                ["NchwcTransformer"]
                if level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                and platform not in {"", "amd64"}
                else None
            )
            optimized_onnx_path: Optional[Path] = None
            if save_optimized_onnx_model:
                optimized_onnx_path = ort_path.with_suffix(".optimized.onnx")
                so_onnx = ort.SessionOptions()
                so_onnx.optimized_model_filepath = str(optimized_onnx_path)
                so_onnx.graph_optimization_level = level
                if optimization_style.lower() == "runtime":
                    so_onnx.add_session_config_entry(
                        "optimization.minimal_build_optimizations", "apply"
                    )
                ort.InferenceSession(
                    str(onnx_path),
                    sess_options=so_onnx,
                    providers=["CPUExecutionProvider"],
                    disabled_optimizers=disabled_optimizers,
                )
            so = ort.SessionOptions()
            so.optimized_model_filepath = str(ort_path)
            so.graph_optimization_level = level
            so.add_session_config_entry("session.save_model_format", "ORT")
            if optimization_style.lower() == "runtime":
                so.add_session_config_entry("optimization.minimal_build_optimizations", "save")
            ort.InferenceSession(
                str(onnx_path), sess_options=so,
                providers=["CPUExecutionProvider"], disabled_optimizers=disabled_optimizers
            )
            return ort_path, optimized_onnx_path

    @classmethod
    def register(cls, name: str, exts: Tuple[str, ...], impl: ConverterBase) -> None:
        cls._by_name[name] = impl
        for e in exts:
            cls._ext_map[e.lower()] = name

    @classmethod
    def for_ext(cls, ext: str) -> Optional[ConverterBase]:
        name = cls._ext_map.get(ext.lower())
        return cls._by_name.get(name) if name else None


class OnnxConverter:
    name = "onnx"
    def convert(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        out = Converter._OnnxLayer.export(
            model, dst,
            sample_input=opts.get("sample_input"),
            opset_version=int(opts.get("opset_version", 18)),
            dynamic_batch=bool(opts.get("dynamic_batch", True)),
        )
        return (out,)


class OrtConverter:
    name = "ort"
    def convert(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        onnx_path = Path(opts.get("onnx_path") or dst.with_suffix(".onnx"))
        onnx_path = Converter._OnnxLayer.ensure(
            model, onnx_path,
            sample_input=opts.get("sample_input"),
            opset_version=int(opts.get("opset_version", 18)),
            dynamic_batch=bool(opts.get("dynamic_batch", True)),
        )
        ort_path, optimized = Converter._OrtLayer.save_ort(
            onnx_path, dst,
            optimization_level=str(opts.get("optimization_level", "all")),
            optimization_style=str(opts.get("optimization_style", "fixed")),
            target_platform=opts.get("target_platform"),
            save_optimized_onnx_model=bool(opts.get("save_optimized_onnx_model", False)),
        )
        return (ort_path, optimized) if optimized is not None else (ort_path,)


class TensorRTConverter:
    name = "tensorrt"
    def convert(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        onnx_path = Path(opts.get("onnx_path") or dst.with_suffix(".onnx"))
        onnx_path = Converter._OnnxLayer.ensure(
            model, onnx_path,
            sample_input=opts.get("sample_input"),
            opset_version=int(opts.get("opset_version", 18)),
            dynamic_batch=bool(opts.get("dynamic_batch", True)),
        )
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise MissingDependencyError("tensorrt required") from exc

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as config:
            workspace_size_bytes = int(opts.get("workspace_size_bytes", 1 << 30))
            if hasattr(config, "set_memory_pool_limit"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_bytes)
            else:
                config.max_workspace_size = workspace_size_bytes
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(parser.get_error(i))
                    raise RuntimeError("TensorRT failed to parse ONNX")
            if bool(opts.get("fp16", True)) and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if bool(opts.get("int8", False)):
                if not builder.platform_has_fast_int8:
                    warnings.warn("INT8 not supported; ignoring")
                else:
                    calibrator = opts.get("calibrator")
                    if calibrator is not None:
                        config.set_int8_calibrator(calibrator)
                    config.set_flag(trt.BuilderFlag.INT8)
            input_tensor = network.get_input(0)
            sample = _pad_sample(model, opts.get("sample_input"))
            s = tuple(int(x) for x in sample.shape)
            min_shape = (1, *s[1:])
            opt_shape = (int(opts.get("opt_batch", s[0])), *s[1:])
            max_shape = (int(opts.get("max_batch", 8)), *s[1:])
            profile = builder.create_optimization_profile()
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("engine build failed")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                f.write(engine_bytes)
        return (dst,)


class NnefConverter:
    name = "nnef"
    def convert(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        onnx_path = Path(opts.get("onnx_path") or dst.with_suffix(".onnx"))
        onnx_path = Converter._OnnxLayer.ensure(
            model, onnx_path,
            sample_input=opts.get("sample_input"),
            opset_version=int(opts.get("opset_version", 18)),
            dynamic_batch=bool(opts.get("dynamic_batch", True)),
        )
        try:
            import importlib
            importlib.import_module("nnef_tools.convert")
        except ImportError as exc:
            raise MissingDependencyError("nnef-tools[onnx] required") from exc
        input_shapes = opts.get("input_shapes")
        if input_shapes is None:
            sample = _pad_sample(model, opts.get("sample_input"))
            input_shapes = {"features": tuple(int(x) for x in sample.shape)}
        cmd = [
            os.sys.executable, "-m", "nnef_tools.convert",
            "--input-format", "onnx", "--output-format", "nnef",
            "--input-model", str(onnx_path), "--output-model", str(dst),
            "--input-shapes", json.dumps(input_shapes),
        ]
        toggles = (
            ("keep_io_names", "--keep-io-names", True),
            ("io_transpose", "--io-transpose", False),
            ("fold_constants", "--fold-constants", True),
            ("optimize", "--optimize", True),
            ("compress", "--compress", True),
        )
        for key, flag, default in toggles:
            if bool(opts.get(key, default)):
                cmd.append(flag)
        _run_cmd(cmd, "nnef convert")
        return (dst,)


class CoreMLConverter:
    name = "coreml"
    def convert(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        _require("coremltools", "pip install coremltools")
        import coremltools as ct
        sample = _pad_sample(model, opts.get("sample_input"))
        wrapper = _ExportCompat(model).eval()
        with Gradient.inference(wrapper):
            scripted = torch.jit.trace(wrapper, sample)
        cu_map = {
            "ALL": getattr(ct.ComputeUnit, "ALL", None),
            "CPU_ONLY": getattr(ct.ComputeUnit, "CPU_ONLY", None),
            "CPU_AND_GPU": getattr(ct.ComputeUnit, "CPU_AND_GPU", None),
            "CPU_AND_NE": getattr(ct.ComputeUnit, "CPU_AND_NE", None),
        }
        convert_to = str(opts.get("convert_to", "mlprogram"))
        compute_units = cu_map.get(str(opts.get("compute_units", "ALL")).upper(), cu_map["ALL"])
        kwargs_dict: Dict[str, Any] = {
            "inputs": [ct.TensorType(shape=tuple(int(x) for x in sample.shape))],
            "convert_to": convert_to,
            "compute_units": compute_units,
        }
        mdt = opts.get("minimum_deployment_target")
        if mdt:
            target = getattr(ct.target, mdt, None)
            if target is not None:
                kwargs_dict["minimum_deployment_target"] = target
        mlmodel = ct.convert(scripted, **kwargs_dict)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(dst))
        return (dst,)


class LiteRTConverter:
    name = "litert"
    def convert(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        sample_input = opts.get("sample_input")
        opset_version = int(opts.get("opset_version", 18))
        dynamic_batch = bool(opts.get("dynamic_batch", True))
        onnx_path = Path(opts.get("onnx_path") or dst.with_suffix(".onnx"))
        onnx_path = Converter._OnnxLayer.ensure(
            model,
            onnx_path,
            sample_input=sample_input,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
        )
        if bool(opts.get("prefer_onnx2tf", True)):
            try:
                out_dir = dst.with_suffix("")
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    os.sys.executable, "-m", "onnx2tf",
                    "-i", str(onnx_path), "-o", str(out_dir),
                    "--copy_onnx_input_output_names_to_tflite",
                ]
                _run_cmd(cmd, "onnx2tf")
                tflites = list(out_dir.glob("*.tflite"))
                if not tflites:
                    raise RuntimeError("onnx2tf did not produce .tflite")
                shutil.copyfile(tflites[0], dst)
                return (dst,)
            except Exception as exc:
                warnings.warn(f"onnx2tf failed; fallback to onnx-tf path: {exc}")
        _require("onnx", "pip install onnx")
        _require("onnx-tf", "pip install onnx-tf")
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        model_onnx = onnx.load(str(onnx_path))
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpd:
            saved_model_dir = Path(tmpd) / "saved_model"
            prepare(model_onnx).export_graph(str(saved_model_dir))
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            if bool(opts.get("allow_fp16", False)):
                converter.target_spec.supported_types = [tf.float16]
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if bool(opts.get("int8_quantize", False)):
                rep_ds = opts.get("representative_dataset")
                if rep_ds is None:
                    raise ValueError("representative_dataset required for INT8 quantization")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = rep_ds
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            tflite_model = converter.convert()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                f.write(tflite_model)
        return (dst,)


Converter.register("onnx", (".onnx",), OnnxConverter())
Converter.register("ort", (".ort",), OrtConverter())
Converter.register("tensorrt", (".engine",), TensorRTConverter())
Converter.register("nnef", (".nnef",), NnefConverter())
Converter.register("coreml", (".mlmodel",), CoreMLConverter())
Converter.register("litert", (".tflite",), LiteRTConverter())


def new_model(
    in_dim: int,
    out_shape: Sequence[int],
    config: ModelConfig | Dict[str, Any] | None,
) -> nn.Module:
    cfg = coerce_model_config(config)
    core = Root(in_dim, tuple(int(x) for x in out_shape), config=cfg)
    model_td = Fusion.td_from_module(core, in_key="features", out_key="pred", add_loss=True)
    return model_td


def load_model(
    checkpoint_path: str,
    in_dim: Optional[int] = None,
    out_shape: Optional[Sequence[int]] = None,
    config: ModelConfig | Dict[str, Any] | None = None,
    map_location: Optional[str] = None,
) -> nn.Module:

    if os.path.isdir(checkpoint_path):
        if in_dim is None or out_shape is None:
            raise ValueError("Loading from a checkpoint directory requires in_dim and out_shape.")
        model = new_model(int(in_dim), tuple(out_shape), config)
        opts = StateDictOptions(full_state_dict=True)
        m_sd = get_model_state_dict(model, options=opts)
        dcp_load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(checkpoint_path))
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        return model

    p = Path(checkpoint_path)
    if p.suffix.lower() == ".safetensors":
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            raise RuntimeError("missing sidecar JSON for safetensors model")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        use_in_dim = int(in_dim if in_dim is not None else meta.get("in_dim"))
        use_out_shape = tuple(int(x) for x in (out_shape if out_shape is not None else meta.get("out_shape") or ()))
        use_config = coerce_model_config(
            config if config is not None else meta.get("config")
        )
        model = new_model(use_in_dim, use_out_shape, use_config)
        _require("safetensors", "pip install safetensors")
        from safetensors.torch import load_file as load_tensors
        sd = load_tensors(str(p), device=map_location or "cpu")
        model.load_state_dict(sd)
        return model

    obj = torch.load(str(p), map_location=map_location or "cpu", weights_only=False)
    meta_in_dim = obj.get("in_dim") if isinstance(obj, dict) else None
    meta_out_shape = obj.get("out_shape") if isinstance(obj, dict) else None
    meta_cfg = obj.get("config") if isinstance(obj, dict) else None
    use_in_dim = int(in_dim if in_dim is not None else meta_in_dim)
    use_out_shape = tuple(out_shape if out_shape is not None else meta_out_shape or ())
    use_config = coerce_model_config(config if config is not None else meta_cfg)
    model = new_model(use_in_dim, use_out_shape, use_config)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(sd)
    return model


def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> str:

    p = Path(path)

    if TorchIO.is_native_target(p):
        out = TorchIO.save(model, p, optimizer=optimizer, extra=extra, **kwargs)
        return str(out)

    conv = Converter.for_ext(p.suffix)
    if conv is None:
        raise ValueError(f"Unknown format for path: {path}")
    conv.convert(model, p, **kwargs)
    return str(p)
