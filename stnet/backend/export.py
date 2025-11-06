# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from ..api.io import _Format
from ..functional.fx import Gradient


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is required at runtime."""


def _in_console(cmd: Sequence[str], desc: str) -> None:
    try:
        subprocess.run(list(cmd), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{desc} failed: {exc}") from exc


def _get_tensor_shape(model: nn.Module, sample_input: Optional[torch.Tensor]) -> Tuple[int, Tuple[int, ...]]:
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
    in_dim, _ = _get_tensor_shape(model, sample_input)
    try:
        param = next(model.parameters())
        dtype, device = (param.dtype, param.device)
    except StopIteration:
        dtype, device = (torch.float32, torch.device("cpu"))
    return torch.zeros(1, in_dim, dtype=dtype, device=device)


class _CompatLayer(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_flat, _ = self.net(x, labels_flat=None, net_loss=None)
        return y_flat


def _onnx_options(opts: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sample_input": opts.get("sample_input"),
        "opset_version": int(opts.get("opset_version", 18)),
        "dynamic_batch": bool(opts.get("dynamic_batch", True)),
    }


def _resolve_onnx_path(dst: Path, opts: Mapping[str, Any]) -> Path:
    override = opts.get("onnx_path")
    if override:
        return Path(override)
    return dst.with_suffix(".onnx")


def is_required(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise MissingDependencyError(f"{module} is required for this operation{hint}") from err


# Every backend compiler implements the shared stnet.api.io._Format protocol.
class Onnx(_Format):
    name = "onnx"

    def save(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        out = Format._OnnxLayer.export(model, dst, **_onnx_options(opts))
        return (out,)


class Ort(_Format):
    name = "ort"

    def save(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        onnx_path = Format._OnnxLayer.coerce(model, _resolve_onnx_path(dst, opts), **_onnx_options(opts))
        ort_path, optimized = Format._OrtLayer.to_ort(
            onnx_path,
            dst,
            optimization_level=str(opts.get("optimization_level", "all")),
            optimization_style=str(opts.get("optimization_style", "fixed")),
            target_platform=opts.get("target_platform"),
            save_optimized_onnx_model=bool(opts.get("save_optimized_onnx_model", False)),
        )
        return (ort_path, optimized) if optimized is not None else (ort_path,)


class TensorRT(_Format):
    name = "tensorrt"

    def save(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        onnx_path = Format._OnnxLayer.coerce(model, _resolve_onnx_path(dst, opts), **_onnx_options(opts))
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise MissingDependencyError("tensorrt required") from exc

        trt_logger = trt.Logger(trt.Logger.WARNING)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(trt_logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, trt_logger) as parser, builder.create_builder_config() as config:
            workspace_size_bytes = int(opts.get("workspace_size_bytes", 1 << 30))
            if hasattr(config, "set_memory_pool_limit"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_bytes)
            else:
                config.max_workspace_size = workspace_size_bytes
            with open(onnx_path, "rb") as handle:
                if not parser.parse(handle.read()):
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
            shape = tuple(int(x) for x in sample.shape)
            min_shape = (1, *shape[1:])
            opt_shape = (int(opts.get("opt_batch", shape[0])), *shape[1:])
            max_shape = (int(opts.get("max_batch", 8)), *shape[1:])
            profile = builder.create_optimization_profile()
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("engine build failed")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as handle:
                handle.write(engine_bytes)
        return (dst,)


class Nnef(_Format):
    name = "nnef"

    def save(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        onnx_path = Format._OnnxLayer.coerce(model, _resolve_onnx_path(dst, opts), **_onnx_options(opts))
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
            sys.executable,
            "-m",
            "nnef_tools.convert",
            "--input-format",
            "onnx",
            "--output-format",
            "nnef",
            "--input-model",
            str(onnx_path),
            "--output-model",
            str(dst),
            "--input-shapes",
            json.dumps(input_shapes),
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
        _in_console(cmd, "nnef convert")
        return (dst,)


class CoreML(_Format):
    name = "coreml"

    def save(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        is_required("coremltools", "pip install coremltools")
        import coremltools as ct

        sample = _pad_sample(model, opts.get("sample_input"))
        wrapper = _CompatLayer(model).eval()
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
        deployment_target = opts.get("minimum_deployment_target")
        if deployment_target:
            target = getattr(ct.target, deployment_target, None)
            if target is not None:
                kwargs_dict["minimum_deployment_target"] = target
        mlmodel = ct.convert(scripted, **kwargs_dict)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(dst))
        return (dst,)


class LiteRT(_Format):
    name = "litert"

    def save(self, model: nn.Module, dst: Path, *args: Any, **opts: Any) -> Tuple[Path, ...]:
        onnx_path = Format._OnnxLayer.coerce(
            model,
            _resolve_onnx_path(dst, opts),
            **_onnx_options(opts),
        )
        if bool(opts.get("prefer_onnx2tf", True)):
            try:
                out_dir = dst.with_suffix("")
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-m",
                    "onnx2tf",
                    "-i",
                    str(onnx_path),
                    "-o",
                    str(out_dir),
                    "--copy_onnx_input_output_names_to_tflite",
                ]
                _in_console(cmd, "onnx2tf")
                tflites = list(out_dir.glob("*.tflite"))
                if not tflites:
                    raise RuntimeError("onnx2tf did not produce .tflite")
                shutil.copyfile(tflites[0], dst)
                return (dst,)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"onnx2tf failed; fallback to onnx-tf path: {exc}")
        is_required("onnx", "pip install onnx")
        is_required("onnx-tf", "pip install onnx-tf")
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        model_onnx = onnx.load(str(onnx_path))

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
            with open(dst, "wb") as handle:
                handle.write(tflite_model)
        return (dst,)


class Format:
    _by_name: Dict[str, _Format] = {}
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
            is_required("onnx", "pip install onnx")
            wrapper = _CompatLayer(model).eval()
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
                (RuntimeError,) if export_error is RuntimeError else (RuntimeError, export_error)
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
        def coerce(model: nn.Module, onnx_path: Path, *args: Any, **opts: Any) -> Path:
            if not onnx_path.exists():
                return Format._OnnxLayer.export(model, onnx_path, **opts)
            return onnx_path

    class _OrtLayer:
        @staticmethod
        def to_ort(
            onnx_path: Path,
            ort_path: Path,
            *args: Any,
            optimization_level: str = "all",
            optimization_style: str = "fixed",
            target_platform: Optional[str] = None,
            save_optimized_onnx_model: bool = False,
            **kwargs: Any,
        ) -> Tuple[Path, Optional[Path]]:
            is_required("onnxruntime", "pip install onnxruntime")
            import onnxruntime as ort

            opt_map = {
                "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            level = opt_map.get(optimization_level.lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            platform = (target_platform or "").lower()
            disabled_optimizers = (
                ["NchwcTransformer"]
                if level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL and platform not in {"", "amd64"}
                else None
            )
            optimized_onnx_path: Optional[Path] = None
            if save_optimized_onnx_model:
                optimized_onnx_path = ort_path.with_suffix(".optimized.onnx")
                so_onnx = ort.SessionOptions()
                so_onnx.optimized_model_filepath = str(optimized_onnx_path)
                so_onnx.graph_optimization_level = level
                if optimization_style.lower() == "runtime":
                    so_onnx.add_session_config_entry("optimization.minimal_build_optimizations", "apply")
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
                str(onnx_path),
                sess_options=so,
                providers=["CPUExecutionProvider"],
                disabled_optimizers=disabled_optimizers,
            )
            return ort_path, optimized_onnx_path

    @classmethod
    def register(cls, name: str, exts: Tuple[str, ...], impl: _Format) -> None:
        cls._by_name[name] = impl
        for ext in exts:
            cls._ext_map[ext.lower()] = name

    @classmethod
    def for_export(cls, ext: str) -> Optional[_Format]:
        name = cls._ext_map.get(ext.lower())
        return cls._by_name.get(name) if name else None


Format.register("onnx", (".onnx",), Onnx())
Format.register("ort", (".ort",), Ort())
Format.register("tensorrt", (".engine",), TensorRT())
Format.register("nnef", (".nnef",), Nnef())
Format.register("coreml", (".mlmodel",), CoreML())
Format.register("litert", (".tflite",), LiteRT())

# Backwards compatibility: expose the legacy Export symbol.
Export = Format
