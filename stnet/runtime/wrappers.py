# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
import threading
import weakref
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence, TypeAlias

import torch
from tensordict import TensorDict
from torch import nn

from ..core.concurrency import Mutex
from ..core.tensor import extract_tensor, from_buffer
from ..core.datatypes import PathLike, write_json
from ..nn.layers import Recorder
from .io import Format, _load_model_config, _temp_environ, is_required


_FORWARD_PARAM_CACHE: dict[object, object] = {}
_FORWARD_PARAM_CACHE_LOCK = Mutex()

_EXPORT_SIG_CACHE: object | None = None
_EXPORT_SIG_LOCK = Mutex()


@contextlib.contextmanager
def _onnx_model(model: object) -> Iterator[object]:
    was_training = getattr(model, "training", False)
    removed_top, removed_sub = {}, []
    model.eval()
    for name in (
        "optimizer",
        "optimizer_state",
        "optim",
        "training_history",
        "history",
        "logger",
        "metrics",
        "_training_history",
    ):
        if hasattr(model, name):
            with contextlib.suppress(Exception):
                removed_top[name] = getattr(model, name)
                delattr(model, name)
    with contextlib.suppress(Exception):
        for module in model.modules():
            for attr in ("logger", "history"):
                if hasattr(module, attr) and isinstance(v := getattr(module, attr), Recorder):
                    with contextlib.suppress(Exception):
                        removed_sub.append((module, attr, v))
                        delattr(module, attr)
    try:
        with _temp_environ({"STNET_MSR_FORCE_TORCH": "1"}, only_if_unset=True):
            eager_ctx = getattr(model, "eager_for_export", None)
            with eager_ctx() if callable(eager_ctx) else contextlib.nullcontext():
                yield model
    finally:
        for module, attr, v in removed_sub:
            with contextlib.suppress(Exception):
                setattr(module, attr, v)
        for name, v in removed_top.items():
            with contextlib.suppress(Exception):
                setattr(model, name, v)
        if was_training:
            with contextlib.suppress(Exception):
                model.train(True)


def _get_forward_parameters(model_cls: object) -> object:
    with _FORWARD_PARAM_CACHE_LOCK:
        cached = _FORWARD_PARAM_CACHE.get(model_cls)
        if cached is not None:
            return cached
    try:
        sig = inspect.signature(model_cls.forward)
    except Exception:
        return None
    info = (
        set(sig.parameters.keys()),
        any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()),
    )
    with _FORWARD_PARAM_CACHE_LOCK:
        if len(_FORWARD_PARAM_CACHE) > 512:
            _FORWARD_PARAM_CACHE.clear()
        return _FORWARD_PARAM_CACHE.setdefault(model_cls, info)


def _forward(model: object, x: object) -> object:
    fwd_export = getattr(model, "forward_export", None)
    if callable(fwd_export) and isinstance(x, torch.Tensor):
        return fwd_export(x)
    info = _get_forward_parameters(type(model))
    if info is None:
        return model(x)
    names, accepts_kwargs = info
    kwargs = {}
    for key in ("labels_flat", "net_loss"):
        if accepts_kwargs or key in names:
            kwargs[key] = None
    return model(x, **kwargs) if kwargs else model(x)


def _get_tensor_shape(model: object, sample_input: object) -> object:
    in_dim = int(getattr(model, "in_dim")) if hasattr(model, "in_dim") else None
    out_shape = (
        tuple(int(x) for x in getattr(model, "out_shape")) if hasattr(model, "out_shape") else None
    )
    if (in_dim is None or out_shape is None) and sample_input is not None:
        dev = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
        sample = sample_input.to(dev)
        model.eval()
        with torch.no_grad():
            y_flat = extract_tensor(
                _forward(model, sample.unsqueeze(0) if sample.ndim == 1 else sample)
            )
        in_dim = in_dim or int(sample.numel() // sample.shape[0])
        out_shape = out_shape or tuple(y_flat.shape[1:])
    if in_dim is None or out_shape is None:
        raise RuntimeError("Failed to infer shapes.")
    return (int(in_dim), tuple(out_shape))


def _pad_sample(model: object, sample_input: object, *, batch: int = 1) -> object:
    if sample_input is not None:
        return sample_input
    in_dim, _ = _get_tensor_shape(model, sample_input)
    param = next(model.parameters(), None)
    dtype, device = (
        (param.dtype, param.device) if param is not None else (torch.float32, torch.device("cpu"))
    )
    b = max(1, int(batch))
    return torch.zeros(b, in_dim, dtype=dtype, device=device)


def _onnx_options(kwargs: object, *args: Any, target: str = "onnx") -> object:
    target_l = str(target or "onnx").strip().lower()
    defaults = {
        "tensorrt": (18, True, True, True, [18, 17, 16, 15, 13]),
        "tensorflow": (18, False, True, True, [18, 17, 16, 15, 13]),
        "default": (18, False, True, False, [18, 17, 16, 15, 13]),
    }
    key = target_l.replace("-", "").replace("_", "")
    if key == "trt":
        key = "tensorrt"
    d_opset, d_dyn, d_pref, d_simp, d_fb = defaults.get(
        target_l, defaults.get(key, defaults["default"])
    )
    opset = int(kwargs.get("opset_version", d_opset))
    fb = kwargs.get("opset_fallback", d_fb)
    fallback = (
        [int(x) for x in re.split(r"[\s,]+", fb) if x]
        if isinstance(fb, str)
        else [int(x) for x in fb]
        if isinstance(fb, (list, tuple))
        else [int(fb)]
    )
    return {
        "sample_input": kwargs.get("sample_input"),
        "opset_version": opset,
        "opset_fallback": [opset] + [v for v in fallback if v != opset],
        "dynamic_batch": kwargs.get("dynamic_batch", d_dyn),
        "prefer_dynamo": kwargs.get("prefer_dynamo", kwargs.get("dynamo", d_pref)),
        "simplify": kwargs.get("simplify_onnx", kwargs.get("onnx_simplify", d_simp)),
    }


def _coerce_onnx_path(dst: PathLike, kwargs: object) -> object:
    return Path(kwargs.get("onnx_path") or dst.with_suffix(".onnx"))


def _in_console(cmd: object, desc: object) -> None:
    try:
        subprocess.run(list(cmd), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{desc} failed with error: {exc}") from exc


def _export_sig() -> object:
    global _EXPORT_SIG_CACHE
    if _EXPORT_SIG_CACHE is not None:
        return _EXPORT_SIG_CACHE
    with _EXPORT_SIG_LOCK:
        if _EXPORT_SIG_CACHE is not None:
            return _EXPORT_SIG_CACHE
        try:
            sig = inspect.signature(torch.onnx.export)
        except Exception:
            sig = None
        _EXPORT_SIG_CACHE = sig
        return sig


def _export_sig_keys() -> set[str]:
    sig = _export_sig()
    if sig is None:
        return set()
    try:
        params = getattr(sig, "parameters", None)
        if isinstance(params, dict):
            return set(params.keys())
    except Exception:
        pass
    try:
        return set(sig.parameters.keys())  # type: ignore[attr-defined]
    except Exception:
        return set()


class _TensorDictPack(nn.Module):
    def __init__(self, averaged_module: nn.Module, key: str) -> None:
        super().__init__()
        self._averaged_module = averaged_module
        self._key = str(key)

    def forward(self, x: torch.Tensor) -> Any:
        bs = int(x.shape[0]) if (hasattr(x, "ndim") and x.ndim >= 1) else 1
        td = TensorDict({self._key: x}, batch_size=[bs], device=x.device)
        return self._averaged_module(td)


class _TensorOutputModule(nn.Module):
    def __init__(self, net: object) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: object) -> object:
        return extract_tensor(_forward(self.net, x))


class _ONNXExporter:
    @staticmethod
    def export(
        model: nn.Module,
        onnx_path: PathLike,
        *args: Any,
        sample_input: object | None = None,
        opset_version: int = 18,
        opset_fallback: Sequence[int] | None = None,
        dynamic_batch: bool = False,
        prefer_dynamo: bool = True,
        simplify: bool = False,
    ) -> object:
        is_required("onnx", "pip install onnx")
        wrapper = _TensorOutputModule(model).eval()
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        input_names = ["features"]
        dyn_axes = None
        dyn_shapes = None
        if dynamic_batch and hasattr(torch, "export") and hasattr(torch.export, "Dim"):
            try:
                batch_dim = torch.export.Dim("batch", min=2)
            except TypeError:
                dynamic_batch = False
            else:
                dyn_axes = {"features": {0: "batch"}, "preds_flat": {0: "batch"}}
                dyn_shapes = ({0: batch_dim},)
        min_export_batch = 2 if dynamic_batch else 1
        sample = _pad_sample(model, sample_input, batch=min_export_batch)
        if isinstance(sample, torch.Tensor) and sample.ndim == 1:
            sample = sample.unsqueeze(0)
        if (
            dynamic_batch
            and isinstance(sample, torch.Tensor)
            and sample.ndim >= 2
            and int(sample.shape[0]) < min_export_batch
        ):
            pad = torch.zeros(
                (min_export_batch - int(sample.shape[0]),) + tuple(sample.shape[1:]),
                device=sample.device,
                dtype=sample.dtype,
            )
            sample = torch.cat([sample, pad], dim=0)
        training = None
        with contextlib.suppress(Exception):
            training = torch.onnx.TrainingMode.EVAL
        sig_keys = _export_sig_keys()
        has_dynamo = "dynamo" in sig_keys
        exporters = [True, False] if prefer_dynamo and has_dynamo else [False]
        errors: list[str] = []
        seen: set[int] = set()
        for opset in (opset_version, *(opset_fallback or ())):
            if opset in seen or opset < 1:
                continue
            seen.add(opset)
            base_kw: dict[str, Any] = {
                "export_params": True,
                "f": str(onnx_path),
                "opset_version": int(opset),
                "do_constant_folding": True,
                "keep_initializers_as_inputs": False,
                "input_names": input_names,
                "output_names": ["preds_flat"],
            }
            if training is not None:
                base_kw["training"] = training
            if dyn_axes:
                base_kw["dynamic_axes"] = dyn_axes
            valid_kw = {k: v for k, v in base_kw.items() if k in sig_keys}
            for use_dyn in exporters:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"torchvision|Setting ONNX exporter to use operator set version",
                        )
                        call_kw = dict(valid_kw)
                        if use_dyn and dyn_shapes:
                            call_kw["dynamic_shapes"] = dyn_shapes
                        if has_dynamo:
                            call_kw["dynamo"] = use_dyn
                        call_kw.pop("model", None)
                        call_kw.pop("args", None)
                        if isinstance(sample, tuple):
                            args = sample
                        elif isinstance(sample, list):
                            args = tuple(sample)
                        else:
                            args = (sample,)
                        torch.onnx.export(model=wrapper, args=args, **call_kw)
                    if simplify:
                        with contextlib.suppress(Exception):
                            import onnx
                            import onnxsim

                            model_onnx = onnx.load(str(onnx_path))
                            model_simp, ok = onnxsim.simplify(model_onnx)
                            if ok:
                                onnx.save(model_simp, str(onnx_path))
                    return onnx_path
                except Exception as exc:
                    errors.append(f"opset={opset} dynamo={use_dyn} -> {exc}")
        raise RuntimeError("ONNX export failed. " + "; ".join(errors[-6:]))

    @staticmethod
    def coerce(model: nn.Module, onnx_path: PathLike, *args: Any, **kwargs: Any) -> object:
        if not onnx_path.exists():
            return _ONNXExporter.export(model, onnx_path, **kwargs)
        return onnx_path


class _ORTBuilder:
    @staticmethod
    def to_ort(
        onnx_path: PathLike,
        ort_path: PathLike,
        *args: Any,
        optimization_level: str = "all",
        optimization_style: str = "fixed",
        target_platform: object | None = None,
        save_optimized_onnx_model: bool = False,
        **kwargs: Any,
    ) -> object:
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
        optimized_onnx_path = None
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
            str(onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
            disabled_optimizers=disabled_optimizers,
        )
        return (ort_path, optimized_onnx_path)

@dataclass(frozen=True, slots=True)
class BorrowedModule:
    module: nn.Module
    name: str | None = None


@dataclass(frozen=True, slots=True)
class OwnedModule:
    module: nn.Module
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ModulePath:
    path: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class CallArguments:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class TorchInductor(Format):
    name = "aoti"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        try:
            import torch.export

            torch_export = torch.export.export
        except Exception as exc:
            raise ImportError(
                "torch.export is required for TorchInductor export (PyTorch 2.0+)."
            ) from exc
        try:
            from torch._inductor import aoti_compile_and_package
        except Exception as exc:
            raise ImportError(
                "torch._inductor (TorchInductor) is required for AOT compilation. "
                "Install a PyTorch build with torch.compile/TorchInductor support."
            ) from exc

        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                sample = sample.unsqueeze(0)
            wrapper = _TensorOutputModule(serving_model).eval()

            dynamic_batch = bool(kwargs.get("dynamic_batch", True))
            dynamic_seq = bool(kwargs.get("dynamic_seq", False))
            strict = bool(kwargs.get("strict", True))

            dynamic_shapes = None
            if (dynamic_batch or dynamic_seq) and hasattr(torch, "export") and hasattr(torch.export, "Dim"):
                spec: dict[int, object] = {}
                if dynamic_batch:
                    spec[0] = torch.export.Dim("batch")
                if dynamic_seq and sample.ndim >= 2:
                    spec[1] = torch.export.Dim("seq")
                if spec:
                    dynamic_shapes = {"x": spec}

            export_kw: dict[str, Any] = {}
            try:
                sig = inspect.signature(torch_export)
                params = getattr(sig, "parameters", None)
                if isinstance(params, dict):
                    if "dynamic_shapes" in params and dynamic_shapes is not None:
                        export_kw["dynamic_shapes"] = dynamic_shapes
                    if "strict" in params:
                        export_kw["strict"] = strict
            except Exception:
                if dynamic_shapes is not None:
                    export_kw["dynamic_shapes"] = dynamic_shapes
                export_kw["strict"] = strict

            try:
                with torch.no_grad():
                    exported = torch_export(wrapper, (sample,), **export_kw)
            except Exception as exc:
                strict_supported = "strict" in export_kw
                if strict_supported and export_kw.get("strict", True):
                    warnings.warn(
                        "torch.export strict=True failed; retrying strict=False for TorchInductor export",
                        RuntimeWarning,
                    )
                    export_kw["strict"] = False
                    with torch.no_grad():
                        exported = torch_export(wrapper, (sample,), **export_kw)
                else:
                    raise

            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)

            inductor_configs = kwargs.get("inductor_configs")
            aoti_kw: dict[str, Any] = {}
            try:
                sig = inspect.signature(aoti_compile_and_package)
                params = getattr(sig, "parameters", None)
                if isinstance(params, dict):
                    if "package_path" in params:
                        aoti_kw["package_path"] = str(dst)
                    if inductor_configs is not None and "inductor_configs" in params:
                        aoti_kw["inductor_configs"] = inductor_configs
            except Exception:
                aoti_kw["package_path"] = str(dst)
                if inductor_configs is not None:
                    aoti_kw["inductor_configs"] = inductor_configs

            out_path = aoti_compile_and_package(exported, **aoti_kw)

            out = Path(str(out_path))
            if out != dst:
                with contextlib.suppress(Exception):
                    shutil.copyfile(out, dst)
                out = dst

            with contextlib.suppress(Exception):
                write_json(dst.with_suffix(".json"), _load_model_config(model), indent=2)

        return (out,)


class TorchExport(Format):
    name = "pt2"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        try:
            import torch.export

            torch_save = torch.export.save
        except Exception as exc:
            raise ImportError("torch.export is required for PT2 export (PyTorch 2.0+).") from exc

        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                sample = sample.unsqueeze(0)
            wrapper = _TensorOutputModule(serving_model).eval()

            exported = self._export_program(wrapper, sample, **kwargs)

            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with from_buffer():
                torch_save(exported, str(dst))

            with contextlib.suppress(Exception):
                write_json(dst.with_suffix(".json"), _load_model_config(model), indent=2)

        return (dst,)

    def _export_program(self, wrapper: nn.Module, sample: torch.Tensor, **kwargs: Any) -> object:
        try:
            import torch.export

            torch_export = torch.export.export
        except Exception as exc:
            raise ImportError("torch.export is required for PT2 export (PyTorch 2.0+).") from exc

        dynamic_batch = bool(kwargs.get("dynamic_batch", True))
        dynamic_seq = bool(kwargs.get("dynamic_seq", False))
        strict = bool(kwargs.get("strict", True))

        dynamic_shapes = None
        if (dynamic_batch or dynamic_seq) and hasattr(torch, "export") and hasattr(torch.export, "Dim"):
            spec: dict[int, object] = {}
            if dynamic_batch:
                spec[0] = torch.export.Dim("batch")
            if dynamic_seq and sample.ndim >= 2:
                spec[1] = torch.export.Dim("seq")
            if spec:
                dynamic_shapes = {"x": spec}

        call_kw: dict[str, Any] = {}
        try:
            sig = inspect.signature(torch_export)
            params = getattr(sig, "parameters", None)
            if isinstance(params, dict):
                if "dynamic_shapes" in params and dynamic_shapes is not None:
                    call_kw["dynamic_shapes"] = dynamic_shapes
                if "strict" in params:
                    call_kw["strict"] = strict
        except Exception:
            if dynamic_shapes is not None:
                call_kw["dynamic_shapes"] = dynamic_shapes
            call_kw["strict"] = strict

        try:
            with torch.no_grad():
                return torch_export(wrapper, (sample,), **call_kw)
        except Exception as exc:
            strict_supported = "strict" in call_kw
            if strict_supported and call_kw.get("strict", True):
                warnings.warn(
                    "torch.export strict=True failed; retrying strict=False for PT2 export",
                    RuntimeWarning,
                )
                call_kw["strict"] = False
                with torch.no_grad():
                    return torch_export(wrapper, (sample,), **call_kw)
            raise


class ExecuTorch(Format):
    name = "executorch"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        is_required("executorch", "pip install executorch")
        try:
            from torch.export import export as torch_export
        except ImportError as exc:
            raise ImportError(
                "torch.export is required for ExecuTorch export (PyTorch 2.0+)."
            ) from exc
        import executorch.exir as exir

        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            wrapper = _TensorOutputModule(serving_model).eval()
            with torch.no_grad():
                try:
                    exported = torch_export(wrapper, (sample,))
                except Exception as exc:
                    strict_supported = False
                    with contextlib.suppress(Exception):
                        sig = inspect.signature(torch_export)
                        strict_supported = "strict" in sig.parameters
                    if not strict_supported:
                        raise
                    warnings.warn(
                        "torch.export strict=True failed; retrying strict=False for ExecuTorch export",
                        RuntimeWarning,
                    )
                    exported = torch_export(wrapper, (sample,), strict=False)
            edge = exir.to_edge(exported)
            exec_prog = exir.to_executorch(edge)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as fh:
                exec_prog.write_to_file(fh)
        return (dst,)


class ONNX(Format):
    name = "onnx"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        with _onnx_model(model) as serving:
            out = _ONNXExporter.export(serving, dst, **_onnx_options(kwargs, target="onnx"))
        return (out,)


class ORT(Format):
    name = "ort"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        with _onnx_model(model) as serving:
            onnx_path = _ONNXExporter.coerce(
                serving,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="onnx"),
            )
            ort_path, optimized = _ORTBuilder.to_ort(
                onnx_path,
                dst,
                optimization_level=str(kwargs.get("optimization_level", "all")),
                optimization_style=str(kwargs.get("optimization_style", "fixed")),
                target_platform=kwargs.get("target_platform"),
                save_optimized_onnx_model=bool(kwargs.get("save_optimized_onnx_model", False)),
            )
        return (ort_path, optimized) if optimized is not None else (ort_path,)


class TensorRT(Format):
    name = "tensorrt"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        del args
        with _onnx_model(model) as serving_model:
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="tensorrt"),
            )
            try:
                import tensorrt as trt
            except ImportError as exc:
                raise ImportError("TensorRT is required for this export.") from exc
            trt_logger = trt.Logger(trt.Logger.WARNING)
            explicit_batch_flag = 0
            with contextlib.suppress(Exception):
                explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with contextlib.ExitStack() as stack:
                builder = stack.enter_context(trt.Builder(trt_logger))
                create_network = getattr(builder, "create_network", None)
                if not callable(create_network):
                    raise RuntimeError("TensorRT builder.create_network is not available.")
                try:
                    network = stack.enter_context(create_network(explicit_batch_flag))
                except TypeError:
                    network = stack.enter_context(create_network())

                parser = stack.enter_context(trt.OnnxParser(network, trt_logger))
                config = stack.enter_context(builder.create_builder_config())

                workspace_size_bytes = int(kwargs.get("workspace_size_bytes", 1 << 30))
                if hasattr(config, "set_memory_pool_limit"):
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_bytes)
                else:
                    config.max_workspace_size = workspace_size_bytes
                with open(onnx_path, "rb") as handle:
                    if not parser.parse(handle.read()):
                        for i in range(parser.num_errors):
                            print(parser.get_error(i))
                        raise RuntimeError("TensorRT could not parse the ONNX model.")
                use_fp16 = bool(kwargs.get("fp16", True))
                use_int8 = bool(kwargs.get("int8", False))
                if use_fp16 and bool(getattr(builder, "platform_has_fast_fp16", False)):
                    with contextlib.suppress(Exception):
                        config.set_flag(trt.BuilderFlag.FP16)
                if use_int8:
                    if not bool(getattr(builder, "platform_has_fast_int8", False)):
                        warnings.warn(
                            "INT8 precision is not supported on this platform; ignoring request."
                        )
                    else:
                        calibrator = kwargs.get("calibrator")
                        if calibrator is not None:
                            with contextlib.suppress(Exception):
                                config.set_int8_calibrator(calibrator)
                        with contextlib.suppress(Exception):
                            config.set_flag(trt.BuilderFlag.INT8)
                input_tensor = network.get_input(0)
                sample = _pad_sample(serving_model, kwargs.get("sample_input"))
                shape = tuple((int(x) for x in sample.shape))
                min_batch = max(1, int(kwargs.get("min_batch", 1)))
                opt_batch = max(min_batch, int(kwargs.get("opt_batch", shape[0])))
                max_batch = max(opt_batch, int(kwargs.get("max_batch", 8)))
                min_shape = (min_batch, *shape[1:])
                opt_shape = (opt_batch, *shape[1:])
                max_shape = (max_batch, *shape[1:])
                profile = builder.create_optimization_profile()
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)
                engine_blob = None
                build_serialized = getattr(builder, "build_serialized_network", None)
                if callable(build_serialized):
                    engine_blob = build_serialized(network, config)
                else:
                    build_engine = getattr(builder, "build_engine", None) or getattr(
                        builder, "build_cuda_engine", None
                    )
                    if callable(build_engine):
                        engine = build_engine(network, config)
                        if engine is not None:
                            engine_blob = getattr(engine, "serialize", lambda: None)()

                if engine_blob is None:
                    raise RuntimeError("Failed to build the TensorRT engine.")
                try:
                    engine_bytes = bytes(engine_blob)
                except Exception:
                    engine_bytes = engine_blob
                if not isinstance(engine_bytes, (bytes, bytearray)):
                    buf = getattr(engine_blob, "buffer", None)
                    if buf is not None:
                        engine_bytes = buf
                if not isinstance(engine_bytes, (bytes, bytearray)):
                    raise RuntimeError(
                        f"TensorRT engine serialization returned unexpected type: {type(engine_blob)}"
                    )
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as handle:
                    handle.write(engine_bytes)
        return (dst,)


class CoreML(Format):
    name = "coreml"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        is_required("coremltools", "pip install coremltools")
        import coremltools as ct

        with _onnx_model(model) as serving_model:
            sample = _pad_sample(serving_model, kwargs.get("sample_input"))
            wrapper = _TensorOutputModule(serving_model).eval()
            with torch.no_grad():
                scripted = torch.jit.trace(
                    wrapper,
                    sample,
                    check_trace=False,
                    strict=False,
                )
            with contextlib.suppress(Exception):
                scripted = torch.jit.freeze(scripted)
            with contextlib.suppress(Exception):
                scripted = torch.jit.optimize_for_inference(scripted)
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
            kwargs_dict = {
                "inputs": [ct.TensorType(shape=tuple((int(x) for x in sample.shape)))],
                "convert_to": convert_to,
                "compute_units": compute_units,
            }
            deployment_target = kwargs.get("minimum_deployment_target")
            if deployment_target:
                target = getattr(ct.target, deployment_target, None)
                if target is not None:
                    kwargs_dict["minimum_deployment_target"] = target
            convert_to_try = [str(convert_to)]
            ct_to_l = str(convert_to).strip().lower()
            if ct_to_l != "neuralnetwork":
                convert_to_try.append("neuralnetwork")
            if ct_to_l != "mlprogram":
                convert_to_try.append("mlprogram")
            mlmodel = None
            last_exc: Exception | None = None
            for ct_to in convert_to_try:
                kwargs_dict["convert_to"] = ct_to
                try:
                    mlmodel = ct.convert(scripted, **kwargs_dict)
                    break
                except Exception as exc:
                    last_exc = exc
                    mlmodel = None

            if mlmodel is None:
                raise RuntimeError(
                    f"CoreML conversion failed. Tried convert_to={convert_to_try}."
                ) from last_exc
            dst.parent.mkdir(parents=True, exist_ok=True)
            mlmodel.save(str(dst))
        return (dst,)


class LiteRT(Format):
    name = "litert"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        with _onnx_model(model) as serving_model:
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="litert"),
            )
            if bool(kwargs.get("prefer_onnx2tf", True)) and (
                importlib.util.find_spec("onnx2tf") is not None
            ):
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
            import tensorflow as tf
            from onnx_tf.backend import prepare

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
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                tflite_model = converter.convert()
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as handle:
                    handle.write(tflite_model)
        return (dst,)


class TensorFlow(Format):
    name = "tensorflow"

    def save(
        self,
        model: object,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        with _onnx_model(model) as serving_model:
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                dst,
                **_onnx_options(kwargs, target="tensorflow"),
            )
            saved_model_dir = dst.with_suffix("") if dst.suffix else dst
            saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
            prefer_onnx2tf = bool(kwargs.get("prefer_onnx2tf", True))
            if prefer_onnx2tf and shutil.which("onnx2tf") is not None:
                with contextlib.suppress(Exception):
                    _in_console(
                        [
                            "onnx2tf",
                            "-i",
                            str(onnx_path),
                            "-o",
                            str(saved_model_dir),
                        ],
                        "onnx2tf",
                    )
                    if (saved_model_dir / "saved_model.pb").exists():
                        return (saved_model_dir,)
                    found = list(saved_model_dir.rglob("saved_model.pb"))
                    if len(found) > 0:
                        return (found[0].parent,)
            is_required("onnx-tf", "pip install onnx-tf")
            from onnx_tf.backend import prepare
            import onnx

            model_onnx = onnx.load(str(onnx_path))
            prepare(model_onnx).export_graph(str(saved_model_dir))
        return (saved_model_dir,)


class ReduceMean(nn.Module):
    def __init__(self, dim: int = 1, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.keepdim = bool(keepdim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class GraphSequential(nn.Module):
    _CONTROL_ATTR = "__stnet_subgraph_control_op__"
    def __init__(
        self,
        steps: Sequence[object],
        *args: Any,
        out_shape: object | None = None,
        name: str | None = None,
        root: nn.Module | None = None,
    ) -> None:
        super().__init__()
        del args
        self._name = str(name or "subgraph")
        self._owned = nn.ModuleList()

        self._root_ref: weakref.ReferenceType[nn.Module] | None = (
            weakref.ref(root) if root is not None else None
        )
        self._path_cache: dict[str, weakref.ReferenceType[nn.Module]] = {}
        self._path_cache_lock = threading.Lock()

        self._out_shape_kind, self._out_shape_spec = self._normalize_out_shape(out_shape)

        compiled_steps: list[tuple[object, ...]] = []
        for raw in steps:
            step, extra_args, extra_kwargs = self._parse_step(raw)
            meta: dict[str, Any] | None = None

            if isinstance(step, BorrowedModule):
                if step.name:
                    meta = {"name": str(step.name)}
                compiled_steps.append(("ref", weakref.ref(step.module), extra_args, extra_kwargs, meta))
                continue
            if isinstance(step, ModulePath):
                meta = {"path": str(step.path), "name": (str(step.name) if step.name else None)}
                compiled_steps.append(("path", str(step.path), extra_args, extra_kwargs, meta))
                continue
            if isinstance(step, OwnedModule):
                if step.name:
                    meta = {"name": str(step.name)}
                self._owned.append(step.module)
                compiled_steps.append(("owned", len(self._owned) - 1, extra_args, extra_kwargs, meta))
                continue
            if isinstance(step, nn.Module):
                compiled_steps.append(("ref", weakref.ref(step), extra_args, extra_kwargs, meta))
                continue
            if callable(step):
                tag = getattr(step, self._CONTROL_ATTR, None)
                if tag is not None:
                    meta = {"control": str(tag)}
                compiled_steps.append(("fn", step, extra_args, extra_kwargs, meta))
                continue

            raise TypeError(f"Unsupported GraphSequential step: {type(step)!r}")

        if not compiled_steps:
            raise ValueError("GraphSequential requires at least one step.")
        self._steps: list[tuple[object, ...]] = compiled_steps

    @staticmethod
    def ref(module: nn.Module, *args: Any, name: str | None = None) -> BorrowedModule:
        del args
        return BorrowedModule(module=module, name=name)

    @staticmethod
    def own(module: nn.Module, *args: Any, name: str | None = None) -> OwnedModule:
        del args
        return OwnedModule(module=module, name=name)

    @staticmethod
    def path(path: str, *, name: str | None = None) -> ModulePath:
        return ModulePath(path=str(path), name=name)

    @staticmethod
    def mean(dim: int = 1, *args: Any, keepdim: bool = False) -> OwnedModule:
        del args
        return OwnedModule(module=ReduceMean(dim=int(dim), keepdim=bool(keepdim)), name="mean")

    @staticmethod
    def io(*args: Any, **kwargs: Any) -> CallArguments:
        return CallArguments(args=tuple(args), kwargs=dict(kwargs))

    @staticmethod
    def _tag_control(fn: Callable[..., Any], tag: str) -> Callable[..., Any]:
        try:
            setattr(fn, GraphSequential._CONTROL_ATTR, str(tag))
        except Exception:
            pass
        return fn

    @staticmethod
    def break_graph() -> Callable[..., Any]:
        from ..core.graph import graph_break

        def _op(*a: Any, **kw: Any) -> Any:
            graph_break()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        return GraphSequential._tag_control(_op, "graph_break")

    @staticmethod
    def cudagraph_begin(*args: Any, disable_compile: bool = True) -> Callable[..., Any]:
        from ..core.graph import cudagraph_mark_step_begin, torch_compiler_disable

        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_begin()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        _op = GraphSequential._tag_control(_op, "cudagraph_begin")
        return (
            torch_compiler_disable(_op, reason="subgraph:cudagraph_begin", recursive=False)
            if disable_compile
            else _op
        )

    @staticmethod
    def cudagraph_end(*args: Any, disable_compile: bool = True) -> Callable[..., Any]:
        from ..core.graph import cudagraph_mark_step_end, torch_compiler_disable

        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_end()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        _op = GraphSequential._tag_control(_op, "cudagraph_end")
        return (
            torch_compiler_disable(_op, reason="subgraph:cudagraph_end", recursive=False)
            if disable_compile
            else _op
        )

    @staticmethod
    def no_compile(
        step: nn.Module | Callable[..., Any],
        *args: Any,
        reason: str | None = None,
        recursive: bool = False,
    ) -> Callable[..., Any]:
        from ..core.graph import torch_compiler_disable

        if isinstance(step, nn.Module):
            ref = weakref.ref(step)

            def _call(*a: Any, **kw: Any) -> Any:
                mod = ref()
                if mod is None:
                    raise RuntimeError(
                        "A shared submodule reference was cleared before GraphSequential.forward()."
                    )
                return mod(*a, **kw)

        else:

            def _call(*a: Any, **kw: Any) -> Any:
                return step(*a, **kw)

        wrapped = torch_compiler_disable(
            _call,
            reason=str(reason or "subgraph:no_compile"),
            recursive=bool(recursive),
        )
        return GraphSequential._tag_control(wrapped, "no_compile")

    @staticmethod
    def checkpoint(
        step: nn.Module | Callable[..., Any],
        *args: Any,
        use_reentrant: bool | None = None,
        preserve_rng_state: bool | None = None,
        determinism_check: str | None = None,
    ) -> Callable[..., Any]:
        from ..core.graph import coerce_checkpoint

        if isinstance(step, nn.Module):
            ref = weakref.ref(step)

            def _call(*a: Any, **kw: Any) -> Any:
                mod = ref()
                if mod is None:
                    raise RuntimeError(
                        "A shared submodule reference was cleared before GraphSequential.forward()."
                    )

                def _inner(*aa: Any) -> Any:
                    return mod(*aa, **kw)

                return coerce_checkpoint(
                    _inner,
                    *a,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                    determinism_check=determinism_check,
                )

        else:

            def _call(*a: Any, **kw: Any) -> Any:
                def _inner(*aa: Any) -> Any:
                    return step(*aa, **kw)

                return coerce_checkpoint(
                    _inner,
                    *a,
                    use_reentrant=use_reentrant,
                    preserve_rng_state=preserve_rng_state,
                    determinism_check=determinism_check,
                )

        return GraphSequential._tag_control(_call, "checkpoint")

    def set_root(self, root: nn.Module | None) -> "GraphSequential":
        self._root_ref = weakref.ref(root) if root is not None else None
        with self._path_cache_lock:
            self._path_cache.clear()
        return self

    def bind(self, root: nn.Module | None = None, *args: Any, strict: bool = True) -> "GraphSequential":
        if root is not None:
            self.set_root(root)

        rebound: list[tuple[object, ...]] = []
        for item in list(self._steps):
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)

            if kind == "path":
                path = str(payload)
                mod = self._resolve_path(path)
                m = dict(meta) if isinstance(meta, dict) else {}
                m["path"] = path
                rebound.append(("ref", weakref.ref(mod), extra_args, extra_kwargs, m))
                continue

            if kind == "ref" and payload is None:
                path = meta.get("path") if isinstance(meta, dict) else None
                if isinstance(path, str):
                    mod = self._resolve_path(path)
                    rebound.append(("ref", weakref.ref(mod), extra_args, extra_kwargs, meta))
                    continue
                if strict:
                    raise RuntimeError(
                        "GraphSequential.bind() encountered an unresolved ref without a path hint."
                    )

            rebound.append((kind, payload, extra_args, extra_kwargs, meta))

        self._steps = rebound
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            cur: Any = CallArguments(args=tuple(args), kwargs=dict(kwargs))
        else:
            cur = args[0] if len(args) == 1 else tuple(args)

        for item in self._steps:
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)
            cur = self._apply_step(kind, payload, cur, extra_args, extra_kwargs, meta=meta)

        return self._apply_out_shape(cur)

    def extra_repr(self) -> str:
        return f"name={self._name!r}, out_shape={self._out_shape_spec!r}, steps={len(self._steps)}"

    def __getstate__(self) -> dict[str, object]:
        state = super().__getstate__()

        steps = state.get("_steps", [])
        sanitized: list[tuple[object, ...]] = []
        if isinstance(steps, list):
            for item in steps:
                kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)
                if kind == "ref":
                    sanitized.append((kind, None, extra_args, extra_kwargs, meta))
                else:
                    sanitized.append((kind, payload, extra_args, extra_kwargs, meta))
            state["_steps"] = sanitized

        state["_root_ref"] = None
        state["_path_cache"] = {}
        state["_path_cache_lock"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        if getattr(self, "_path_cache", None) is None:
            self._path_cache = {}
        if getattr(self, "_path_cache_lock", None) is None:
            self._path_cache_lock = threading.Lock()
        if getattr(self, "_root_ref", None) is None:
            self._root_ref = None

    @staticmethod
    def _parse_step(raw: object) -> tuple[object, tuple[Any, ...], dict[str, Any]]:
        if isinstance(raw, (tuple, list)):
            if len(raw) == 2 and isinstance(raw[1], dict):
                return raw[0], (), dict(raw[1])
            if len(raw) == 3 and isinstance(raw[1], (tuple, list)) and isinstance(raw[2], dict):
                return raw[0], tuple(raw[1]), dict(raw[2])
        return raw, (), {}

    @staticmethod
    def _split_step(
        item: object,
    ) -> tuple[str, object, tuple[Any, ...], dict[str, Any], object | None]:
        if not isinstance(item, tuple) or len(item) < 4:
            raise TypeError("Invalid GraphSequential internal step format.")
        kind = str(item[0])
        payload = item[1]

        raw_args = item[2]
        if raw_args is None:
            extra_args: tuple[Any, ...] = ()
        elif isinstance(raw_args, tuple):
            extra_args = raw_args
        else:
            try:
                extra_args = tuple(raw_args)
            except TypeError:
                extra_args = (raw_args,)

        raw_kwargs = item[3]
        if raw_kwargs is None:
            extra_kwargs: dict[str, Any] = {}
        elif isinstance(raw_kwargs, dict):
            extra_kwargs = dict(raw_kwargs)
        else:
            extra_kwargs = dict(raw_kwargs) if hasattr(raw_kwargs, "items") else {}

        meta = item[4] if len(item) >= 5 else None
        return kind, payload, extra_args, extra_kwargs, meta

    @staticmethod
    def _normalize_out_shape(out_shape: object | None) -> tuple[str | None, object | None]:
        if out_shape is None:
            return None, None
        if isinstance(out_shape, dict):
            spec: dict[str, object] = {}
            for k, v in out_shape.items():
                if v is None:
                    spec[str(k)] = None
                else:
                    spec[str(k)] = tuple(int(x) for x in v)
            return "dict", spec
        if (
            isinstance(out_shape, (list, tuple))
            and out_shape
            and isinstance(out_shape[0], (list, tuple, type(None)))
        ):
            shapes: list[object] = []
            for s in out_shape:
                if s is None:
                    shapes.append(None)
                else:
                    shapes.append(tuple(int(x) for x in s))
            return "seq", tuple(shapes)
        return "single", tuple(int(x) for x in out_shape)

    @staticmethod
    def _unpack(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if isinstance(value, CallArguments):
            return tuple(value.args), dict(value.kwargs)
        if isinstance(value, tuple):
            return value, {}
        if isinstance(value, list):
            return tuple(value), {}
        if isinstance(value, dict):
            return (), value
        return (value,), {}

    def _resolve_path(self, path: str) -> nn.Module:
        with self._path_cache_lock:
            ref = self._path_cache.get(path)
        if ref is not None:
            mod = ref()
            if mod is not None:
                return mod

        root = self._root_ref() if self._root_ref is not None else None
        if root is None:
            raise RuntimeError(
                "GraphSequential requires `root=` (or set_root()) when using ModulePath steps."
            )

        mod: nn.Module | None = None
        if hasattr(root, "get_submodule"):
            try:
                mod = root.get_submodule(path)
            except Exception:
                mod = None
        if mod is None:
            cur: nn.Module = root
            for part in str(path).split("."):
                child = getattr(cur, "_modules", None)
                if isinstance(child, dict) and part in child:
                    nxt = child.get(part)
                else:
                    nxt = getattr(cur, part, None)
                if not isinstance(nxt, nn.Module):
                    raise AttributeError(f"Failed to resolve submodule path {path!r} at {part!r}.")
                cur = nxt
            mod = cur

        if not isinstance(mod, nn.Module):
            raise TypeError(f"get_submodule({path!r}) did not return an nn.Module")

        with self._path_cache_lock:
            self._path_cache[path] = weakref.ref(mod)
        return mod

    def _apply_step(
        self,
        kind: str,
        payload: object,
        cur: Any,
        extra_args: tuple[Any, ...],
        extra_kwargs: dict[str, Any],
        *args: Any,
        meta: object | None = None,
    ) -> Any:
        args, kwargs = self._unpack(cur)
        if extra_args:
            args = tuple(args) + tuple(extra_args)
        if extra_kwargs:
            merged = dict(kwargs)
            merged.update(extra_kwargs)
            kwargs = merged

        if kind == "ref":
            mod = payload() if callable(payload) else None
            if mod is None:
                path = meta.get("path") if isinstance(meta, dict) else None
                if isinstance(path, str):
                    mod = self._resolve_path(path)
                else:
                    raise RuntimeError(
                        "A shared submodule reference was cleared (or not bound) before GraphSequential.forward()."
                    )
            if not isinstance(mod, nn.Module):
                raise TypeError(f"GraphSequential ref step did not resolve to nn.Module: {type(mod)!r}")
            return mod(*args, **kwargs)
        if kind == "owned":
            return self._owned[int(payload)](*args, **kwargs)
        if kind == "path":
            return self._resolve_path(str(payload))(*args, **kwargs)
        return payload(*args, **kwargs)

    def _apply_out_shape(self, out: Any) -> Any:
        kind = self._out_shape_kind
        spec = self._out_shape_spec
        if kind is None or spec is None:
            return out

        def _reshape_one(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
            if t.ndim == 0:
                raise RuntimeError("Cannot reshape a scalar output in GraphSequential.")
            return t.reshape(t.shape[0], *shape)

        if kind == "single":
            if not isinstance(out, torch.Tensor):
                raise RuntimeError("out_shape is set but the pipeline output is not a Tensor.")
            return _reshape_one(out, spec)

        if kind == "seq":
            if not isinstance(out, (tuple, list)):
                raise RuntimeError("out_shape expects tuple/list output but got a different type.")
            shapes = list(spec)
            if len(out) != len(shapes):
                raise RuntimeError("out_shape length does not match tuple/list output length.")
            out_list = list(out)
            for i, sh in enumerate(shapes):
                if sh is None:
                    continue
                if not isinstance(out_list[i], torch.Tensor):
                    raise RuntimeError("out_shape expects Tensor outputs in tuple/list.")
                out_list[i] = _reshape_one(out_list[i], sh)
            return tuple(out_list) if isinstance(out, tuple) else out_list
        if not isinstance(out, dict):
            raise RuntimeError("out_shape expects dict output but got a different type.")
        out_dict = dict(out)
        for k, sh in spec.items():
            if sh is None:
                continue
            if k not in out_dict:
                raise RuntimeError(f"out_shape missing key in output dict: {k!r}")
            if not isinstance(out_dict[k], torch.Tensor):
                raise RuntimeError("out_shape expects Tensor values in dict output.")
            out_dict[k] = _reshape_one(out_dict[k], sh)
        return out_dict

    def extract_for_serving(
        self,
        *args: Any,
        root: nn.Module | None = None,
        clone_modules: bool = True,
        strip_control_ops: bool = True,
        name: str | None = None,
    ) -> "GraphSequential":
        import copy

        if root is not None:
            self.set_root(root)

        steps_out: list[object] = []
        for item in list(self._steps):
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(item)

            if kind == "fn":
                fn = payload
                if strip_control_ops and bool(getattr(fn, self._CONTROL_ATTR, "")):
                    continue
                step_obj: object = fn
            else:
                mod: nn.Module | None = None
                if kind == "owned":
                    mod = self._owned[int(payload)]
                elif kind == "path":
                    mod = self._resolve_path(str(payload))
                elif kind == "ref":
                    mod = payload() if callable(payload) else None
                    if mod is None and isinstance(meta, dict) and isinstance(meta.get("path"), str):
                        mod = self._resolve_path(str(meta.get("path")))
                else:
                    raise TypeError(f"Unknown GraphSequential step kind: {kind!r}")

                if not isinstance(mod, nn.Module):
                    raise RuntimeError(
                        f"extract_for_serving could not resolve module for step kind={kind!r}"
                    )

                if clone_modules:
                    try:
                        mod = copy.deepcopy(mod)
                    except Exception as e:
                        warnings.warn(
                            f"GraphSequential.extract_for_serving: deepcopy failed for {mod.__class__.__name__}: {e}. "
                            "Falling back to sharing the original module object.",
                            RuntimeWarning,
                        )
                step_obj = OwnedModule(module=mod)

            if extra_args and extra_kwargs:
                steps_out.append((step_obj, extra_args, extra_kwargs))
            elif extra_args:
                steps_out.append((step_obj, extra_args, {}))
            elif extra_kwargs:
                steps_out.append((step_obj, extra_kwargs))
            else:
                steps_out.append(step_obj)

        out = GraphSequential(
            steps=steps_out,
            out_shape=(self._out_shape_spec if self._out_shape_kind is not None else None),
            name=str(name or f"{self._name}_serving"),
            root=None,
        )
        out.eval()
        with contextlib.suppress(Exception):
            out.requires_grad_(False)
        with contextlib.suppress(Exception):
            setattr(out, "__compiled_for_serving__", True)
        return out
