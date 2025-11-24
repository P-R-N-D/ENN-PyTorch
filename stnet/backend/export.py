# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
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

from ..api.io import Format
from ..model.layers import History


def _in_console(cmd: Sequence[str], desc: str) -> None:
    try:
        subprocess.run(list(cmd), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{desc} failed with error: {exc}") from exc


def _prepare_serving_model(model: nn.Module) -> nn.Module:

    model.eval()
    candidate_attrs = (
        "optimizer",
        "optimizer_state",
        "optim",
        "training_history",
        "history",
        "metrics",
        "_training_history",
    )
    for name in candidate_attrs:
        if hasattr(model, name):
            with contextlib.suppress(Exception):
                delattr(model, name)
    with contextlib.suppress(Exception):
        for module in model.modules():
            if hasattr(module, "history") and isinstance(getattr(module, "history"), History):
                delattr(module, "history")
    return model


def _get_tensor_shape(
    model: nn.Module, sample_input: Optional[torch.Tensor]
) -> Tuple[int, Tuple[int, ...]]:

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
        from ..functional.fx import Gradient

        dev = next(
            (p.device for p in model.parameters() if p is not None), torch.device("cpu")
        )
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
        raise RuntimeError("Failed to infer input and output shapes.")
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


def _onnx_options(kwargs: Mapping[str, Any]) -> Dict[str, Any]:

    return {
        "sample_input": kwargs.get("sample_input"),
        "opset_version": int(kwargs.get("opset_version", 18)),
        "dynamic_batch": bool(kwargs.get("dynamic_batch", True)),
    }


def _resolve_onnx_path(dst: Path, kwargs: Mapping[str, Any]) -> Path:

    override = kwargs.get("onnx_path")
    if override:
        return Path(override)
    return dst.with_suffix(".onnx")


def is_required(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise ImportError(f"{module} is required for this operation{hint}") from err


class _CompatLayer(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_flat, _ = self.net(x, labels_flat=None, net_loss=None)
        return y_flat



class Onnx(Format):
    name = "onnx"

    def save(
        self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any
    ) -> Tuple[Path, ...]:
        serving_model = _prepare_serving_model(model)
        out = Model._OnnxLayer.export(serving_model, dst, **_onnx_options(kwargs))
        return (out,)


class Ort(Format):
    name = "ort"

    def save(
        self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any
    ) -> Tuple[Path, ...]:
        serving_model = _prepare_serving_model(model)
        onnx_path = Model._OnnxLayer.coerce(
            serving_model, _resolve_onnx_path(dst, kwargs), **_onnx_options(kwargs)
        )
        ort_path, optimized = Model._OrtLayer.to_ort(
            onnx_path,
            dst,
            optimization_level=str(kwargs.get("optimization_level", "all")),
            optimization_style=str(kwargs.get("optimization_style", "fixed")),
            target_platform=kwargs.get("target_platform"),
            save_optimized_onnx_model=bool(
                kwargs.get("save_optimized_onnx_model", False)
            ),
        )
        return (ort_path, optimized) if optimized is not None else (ort_path,)


class TensorRT(Format):
    name = "tensorrt"

    def save(
        self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any
    ) -> Tuple[Path, ...]:
        serving_model = _prepare_serving_model(model)
        onnx_path = Model._OnnxLayer.coerce(
            serving_model, _resolve_onnx_path(dst, kwargs), **_onnx_options(kwargs)
        )
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("TensorRT is required for this export.") from exc

        trt_logger = trt.Logger(trt.Logger.WARNING)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with (
            trt.Builder(trt_logger) as builder,
            builder.create_network(explicit_batch) as network,
            trt.OnnxParser(network, trt_logger) as parser,
            builder.create_builder_config() as config,
        ):
            workspace_size_bytes = int(kwargs.get("workspace_size_bytes", 1 << 30))
            if hasattr(config, "set_memory_pool_limit"):
                config.set_memory_pool_limit(
                    trt.MemoryPoolType.WORKSPACE, workspace_size_bytes
                )
            else:
                config.max_workspace_size = workspace_size_bytes
            with open(onnx_path, "rb") as handle:
                if not parser.parse(handle.read()):
                    for i in range(parser.num_errors):
                        print(parser.get_error(i))
                    raise RuntimeError("TensorRT could not parse the ONNX model.")
            use_fp16 = bool(kwargs.get("fp16", True))
            use_int8 = bool(kwargs.get("int8", False))

            if use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if use_int8:
                if not builder.platform_has_fast_int8:
                    warnings.warn(
                        "INT8 precision is not supported on this platform; ignoring request."
                    )
                else:
                    calibrator = kwargs.get("calibrator")
                    if calibrator is not None:
                        config.set_int8_calibrator(calibrator)
                    config.set_flag(trt.BuilderFlag.INT8)
            input_tensor = network.get_input(0)
            sample = _pad_sample(model, kwargs.get("sample_input"))
            shape = tuple(int(x) for x in sample.shape)
            min_shape = (1, *shape[1:])
            opt_shape = (int(kwargs.get("opt_batch", shape[0])), *shape[1:])
            max_shape = (int(kwargs.get("max_batch", 8)), *shape[1:])
            profile = builder.create_optimization_profile()
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes is None:
                raise RuntimeError("Failed to build the TensorRT engine.")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as handle:
                handle.write(engine_bytes)
        return (dst,)


class Nnef(Format):
    name = "nnef"

    def save(
        self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any
    ) -> Tuple[Path, ...]:
        serving_model = _prepare_serving_model(model)
        onnx_path = Model._OnnxLayer.coerce(
            serving_model, _resolve_onnx_path(dst, kwargs), **_onnx_options(kwargs)
        )
        try:
            import importlib

            importlib.import_module("nnef_tools.convert")
        except ImportError as exc:
            raise ImportError("nnef-tools[onnx] is required for this export.") from exc
        input_shapes = kwargs.get("input_shapes")
        if input_shapes is None:
            sample = _pad_sample(model, kwargs.get("sample_input"))
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
            if bool(kwargs.get(key, default)):
                cmd.append(flag)
        _in_console(cmd, "nnef convert")
        return (dst,)


class CoreML(Format):
    name = "coreml"

    def save(
        self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any
    ) -> Tuple[Path, ...]:
        is_required("coremltools", "pip install coremltools")
        serving_model = _prepare_serving_model(model)
        import coremltools as ct
        from ..functional.fx import Gradient

        sample = _pad_sample(model, kwargs.get("sample_input"))
        wrapper = _CompatLayer(serving_model).eval()
        with Gradient.inference(wrapper):
            scripted = torch.jit.trace(wrapper, sample)
        cu_map = {
            "ALL": getattr(ct.ComputeUnit, "ALL", None),
            "CPU_ONLY": getattr(ct.ComputeUnit, "CPU_ONLY", None),
            "CPU_AND_GPU": getattr(ct.ComputeUnit, "CPU_AND_GPU", None),
            "CPU_AND_NE": getattr(ct.ComputeUnit, "CPU_AND_NE", None),
        }
        convert_to = str(kwargs.get("convert_to", "mlprogram"))
        compute_units = cu_map.get(
            str(kwargs.get("compute_units", "ALL")).upper(), cu_map["ALL"]
        )
        kwargs_dict: Dict[str, Any] = {
            "inputs": [ct.TensorType(shape=tuple(int(x) for x in sample.shape))],
            "convert_to": convert_to,
            "compute_units": compute_units,
        }
        deployment_target = kwargs.get("minimum_deployment_target")
        if deployment_target:
            target = getattr(ct.target, deployment_target, None)
            if target is not None:
                kwargs_dict["minimum_deployment_target"] = target
        mlmodel = ct.convert(scripted, **kwargs_dict)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(dst))
        return (dst,)


class LiteRT(Format):
    name = "litert"

    def save(
        self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any
    ) -> Tuple[Path, ...]:
        serving_model = _prepare_serving_model(model)
        onnx_path = Model._OnnxLayer.coerce(
            serving_model,
            _resolve_onnx_path(dst, kwargs),
            **_onnx_options(kwargs),
        )
        if bool(kwargs.get("prefer_onnx2tf", True)):
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
                    raise RuntimeError("onnx2tf did not produce a TFLite model.")
                shutil.copyfile(tflites[0], dst)
                return (dst,)
            except Exception as exc:
                warnings.warn(
                    f"Falling back to the onnx-tf export path because onnx2tf failed: {exc}."
                )
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
            if bool(kwargs.get("allow_fp16", False)):
                converter.target_spec.supported_types = [tf.float16]
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if bool(kwargs.get("int8_quantize", False)):
                rep_ds = kwargs.get("representative_dataset")
                if rep_ds is None:
                    raise ValueError(
                        "A representative_dataset is required for INT8 quantization."
                    )
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = rep_ds
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            tflite_model = converter.convert()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as handle:
                handle.write(tflite_model)
        return (dst,)



class TorchScript(Format):
    name = "torchscript"

    def save(
        self,
        model: nn.Module,
        dst: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Path, ...]:
        method = str(kwargs.get("method", "script")).lower()
        sample = kwargs.get("sample_input")
        serving_model = _prepare_serving_model(model)
        wrapper = _CompatLayer(serving_model).eval()
        from ..functional.fx import Gradient

        if method == "trace":
            if sample is None:
                sample = _pad_sample(model, None)
            with Gradient.inference(wrapper):
                scripted = torch.jit.trace(wrapper, sample)
        else:
            with Gradient.inference(wrapper):
                scripted = torch.jit.script(wrapper)

        if bool(kwargs.get("optimize_for_mobile", False)):
            try:
                from torch.utils.mobile_optimizer import optimize_for_mobile
            except ImportError as exc:
                raise ImportError(
                    "torch.utils.mobile_optimizer is required for optimize_for_mobile=True"
                ) from exc
            backend = str(kwargs.get("mobile_backend", "cpu")).lower()
            try:
                scripted = optimize_for_mobile(scripted, backend=backend)
            except TypeError:
                scripted = optimize_for_mobile(scripted)

        dst.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(dst))
        return (dst,)



class ExecuTorch(Format):
    name = "executorch"

    def save(
        self,
        model: nn.Module,
        dst: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Path, ...]:
        is_required("executorch", "pip install executorch")
        try:
            from torch.export import export as torch_export
        except ImportError as exc:
            raise ImportError(
                "torch.export is required for ExecuTorch export (PyTorch 2.0+)."
            ) from exc

        import executorch.exir as exir

        serving_model = _prepare_serving_model(model)
        sample = kwargs.get("sample_input")
        sample = _pad_sample(serving_model, sample)
        wrapper = _CompatLayer(serving_model).eval()

        from ..functional.fx import Gradient

        with Gradient.inference(wrapper):
            exported = torch_export(wrapper, (sample,))

        edge = exir.to_edge(exported)
        exec_prog = exir.to_executorch(edge)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as fh:
            exec_prog.write_to_file(fh)
        return (dst,)



class TensorFlow(Format):
    name = "tensorflow"

    def save(
        self,
        model: nn.Module,
        dst: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Path, ...]:
        serving_model = _prepare_serving_model(model)
        onnx_path = Model._OnnxLayer.coerce(
            serving_model,
            _resolve_onnx_path(dst, kwargs),
            **_onnx_options(kwargs),
        )
        is_required("onnx", "pip install onnx")
        is_required("onnx-tf", "pip install onnx-tf")
        import onnx
        from onnx_tf.backend import prepare

        model_onnx = onnx.load(str(onnx_path))
        if dst.suffix:
            saved_model_dir = dst.with_suffix("")
        else:
            saved_model_dir = dst
        saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
        prepare(model_onnx).export_graph(str(saved_model_dir))
        return (saved_model_dir,)


class Model:
    _by_name: Dict[str, Format] = {}
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
        def coerce(
            model: nn.Module, onnx_path: Path, *args: Any, **kwargs: Any
        ) -> Path:
            if not onnx_path.exists():
                return Model._OnnxLayer.export(model, onnx_path, **kwargs)
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
                so.add_session_config_entry(
                    "optimization.minimal_build_optimizations", "save"
                )
            ort.InferenceSession(
                str(onnx_path),
                sess_options=so,
                providers=["CPUExecutionProvider"],
                disabled_optimizers=disabled_optimizers,
            )
            return ort_path, optimized_onnx_path

    @classmethod
    def register(cls: object, name: str, exts: Tuple[str, ...], impl: Format) -> None:
        cls._by_name[name] = impl
        for ext in exts:
            cls._ext_map[ext.lower()] = name

    @classmethod
    def for_export(cls: object, ext: str) -> Optional[Format]:
        name = cls._ext_map.get(ext.lower())
        return cls._by_name.get(name) if name else None


Model.register("onnx", (".onnx",), Onnx())
Model.register("ort", (".ort",), Ort())
Model.register("tensorrt", (".engine",), TensorRT())
Model.register("nnef", (".nnef",), Nnef())
Model.register("coreml", (".mlmodel",), CoreML())
Model.register("litert", (".tflite",), LiteRT())
Model.register("torchscript", (".ts", ".torchscript"), TorchScript())
Model.register("executorch", (".pte",), ExecuTorch())
Model.register(
    "tensorflow",
    (".savedmodel", ".pb", ".tf"),
    TensorFlow(),
)
