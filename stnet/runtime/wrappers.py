# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import glob
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

_EXPORT_WARN_FILTERS_INSTALLED = False

_ONNX2TF_HELP_CACHE: str | None = None
_ONNX2TF_HELP_LOCK = Mutex(reentrant=True)


@contextlib.contextmanager
def _no_empty_tensor(root: nn.Module) -> Iterator[None]:
    patched: list[tuple[nn.Module, str, torch.Tensor]] = []
    try:
        for module in root.modules():
            for name in ("pw_x", "pw_y"):
                tensor = getattr(module, name, None)
                if isinstance(tensor, torch.Tensor) and tensor.numel() == 0:
                    patched.append((module, name, tensor))
                    placeholder = tensor.detach().new_zeros((1, 1))
                    setattr(module, name, placeholder)
        yield
    finally:
        for module, name, old in patched:
            with contextlib.suppress(Exception):
                setattr(module, name, old)


def _install_export_warning_filters() -> None:
    global _EXPORT_WARN_FILTERS_INSTALLED
    if _EXPORT_WARN_FILTERS_INSTALLED:
        return
    _EXPORT_WARN_FILTERS_INSTALLED = True
    warnings.filterwarnings(
        "ignore",
        message=r".*Converting a tensor to a Python boolean.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*LeafSpec.*deprecated.*",
    )
    with contextlib.suppress(Exception):
        tw = getattr(torch.jit, "TracerWarning", None)
        if tw is not None:
            warnings.filterwarnings(
                "ignore",
                category=tw,
                message=r".*Converting a tensor to a Python boolean.*",
            )


@contextlib.contextmanager
def _onnx_model(model: object) -> Iterator[object]:
    _install_export_warning_filters()
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
                if hasattr(module, attr) and isinstance(
                    v := getattr(module, attr), Recorder
                ):
                    with contextlib.suppress(Exception):
                        removed_sub.append((module, attr, v))
                        delattr(module, attr)
    try:
        with _temp_environ(
            {
                "STNET_MSR_FORCE_TORCH": "1",
                "STNET_DISABLE_PIECEWISE_CALIB": "1",
            },
            only_if_unset=True,
        ):
            eager_ctx = getattr(model, "eager_for_export", None)
            with (
                eager_ctx()
                if callable(eager_ctx)
                else contextlib.nullcontext()
            ):
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
        any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ),
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
    in_dim = (
        int(getattr(model, "in_dim")) if hasattr(model, "in_dim") else None
    )
    out_shape = (
        tuple(int(x) for x in getattr(model, "out_shape"))
        if hasattr(model, "out_shape")
        else None
    )
    if in_dim is not None and out_shape is not None:
        return (int(in_dim), tuple(int(x) for x in out_shape))
    if sample_input is not None:
        if not isinstance(sample_input, torch.Tensor):
            raise TypeError("sample_input must be a torch.Tensor")
        dev = next(
            (p.device for p in model.parameters() if p is not None),
            torch.device("cpu"),
        )
        sample = sample_input.to(dev)
        model.eval()
        with torch.no_grad():
            sample_batched = (
                sample.unsqueeze(0) if sample.ndim == 1 else sample
            )
            y_flat = extract_tensor(_forward(model, sample_batched))
        if in_dim is None:
            in_dim = (
                int(sample.numel())
                if sample.ndim == 1
                else int(sample.numel() // sample.shape[0])
            )
        if out_shape is None:
            out_shape = tuple(int(x) for x in tuple(y_flat.shape[1:]))
    if in_dim is None or out_shape is None:
        raise RuntimeError("Failed to infer shapes.")
    return (int(in_dim), tuple(int(x) for x in out_shape))


def _pad_sample(
    model: object, sample_input: object, *args: Any, batch: int = 1
) -> object:
    if sample_input is not None:
        return sample_input
    in_dim, _ = _get_tensor_shape(model, sample_input)
    param = next(model.parameters(), None)
    dtype, device = (
        (param.dtype, param.device)
        if param is not None
        else (torch.float32, torch.device("cpu"))
    )
    b = max(1, int(batch))
    return torch.zeros(b, in_dim, dtype=dtype, device=device)


def _onnx_options(kwargs: object, *args: Any, target: str = "onnx") -> object:
    target_l = str(target or "onnx").strip().lower()
    defaults = {
        "tensorrt": (18, True, True, True, []),
        "tensorflow": (18, False, True, True, []),
        "default": (18, False, True, False, []),
    }
    key = target_l.replace("-", "").replace("_", "")
    if key == "trt":
        key = "tensorrt"
    d_opset, d_dyn, d_pref, d_simp, d_fb = defaults.get(
        target_l, defaults.get(key, defaults["default"])
    )
    opset = int(kwargs.get("opset_version", d_opset))
    fb = kwargs.get("opset_fallback", d_fb)
    if not fb:
        fallback: list[int] = []
    elif isinstance(fb, str):
        fallback = [int(x) for x in re.split(r"[\s,]+", fb) if x]
    elif isinstance(fb, (list, tuple)):
        fallback = [int(x) for x in fb]
    else:
        fallback = [int(fb)]
    clean_fb: list[int] = []
    for v in fallback:
        iv = int(v)
        if iv == int(opset):
            continue
        if iv not in clean_fb:
            clean_fb.append(iv)
    return {
        "sample_input": kwargs.get("sample_input"),
        "opset_version": opset,
        "opset_fallback": clean_fb,
        "dynamic_batch": kwargs.get("dynamic_batch", d_dyn),
        "prefer_dynamo": kwargs.get(
            "prefer_dynamo", kwargs.get("dynamo", d_pref)
        ),
        "simplify": kwargs.get(
            "simplify_onnx", kwargs.get("onnx_simplify", d_simp)
        ),
        "optimize_onnx": kwargs.get(
            "optimize_onnx", kwargs.get("onnx_optimize", True)
        ),
        "onnxoptimizer_passes": kwargs.get(
            "onnxoptimizer_passes", kwargs.get("onnx_optimizer_passes")
        ),
    }


def _coerce_onnx_path(dst: PathLike, kwargs: object) -> object:
    dst_p = Path(dst)

    onnx_override = None
    with contextlib.suppress(Exception):
        if isinstance(kwargs, dict):
            onnx_override = kwargs.get("onnx_path")
        else:
            onnx_override = getattr(kwargs, "onnx_path", None)

    if onnx_override:
        return Path(onnx_override)

    if dst_p.suffix.lower() == ".onnx":
        return dst_p

    return dst_p.with_name(dst_p.name + ".onnx")


def _sidecar_json_path(dst: PathLike) -> Path:
    p = Path(dst)
    return p.with_name(p.name + ".json")


def _write_export_meta(
    model: nn.Module,
    dst: PathLike,
    *args: Any,
    format_name: str,
    extra: dict[str, Any] | None = None,
) -> None:
    p = Path(dst)
    out_shape = getattr(model, "out_shape", None) or ()
    if not isinstance(out_shape, (list, tuple)):
        out_shape = (out_shape,)
    payload: dict[str, Any] = {
        "version": 1,
        "format": str(format_name),
        "in_dim": int(getattr(model, "in_dim", 0) or 0),
        "out_shape": tuple(int(x) for x in out_shape),
        "config": _load_model_config(model),
        "pytorch_version": torch.__version__,
    }
    if extra:
        payload["extra"] = extra
    write_json(_sidecar_json_path(p), payload, indent=2)


def _pad_to_batch(sample: torch.Tensor, min_batch: int) -> torch.Tensor:
    if not isinstance(sample, torch.Tensor):
        return sample
    if sample.ndim == 0:
        return sample
    b = int(sample.shape[0]) if sample.shape else 1
    if b >= int(min_batch):
        return sample
    pad_shape = (int(min_batch) - b,) + tuple(sample.shape[1:])
    pad = torch.zeros(pad_shape, dtype=sample.dtype, device=sample.device)
    return torch.cat([sample, pad], dim=0)


def _onnx2tf_help_text() -> str:
    global _ONNX2TF_HELP_CACHE
    cached = _ONNX2TF_HELP_CACHE
    if cached is not None:
        return cached
    with _ONNX2TF_HELP_LOCK:
        cached = _ONNX2TF_HELP_CACHE
        if cached is not None:
            return cached
        try:
            out = subprocess.check_output(
                [sys.executable, "-m", "onnx2tf", "-h"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            out = ""
        _ONNX2TF_HELP_CACHE = out
        return out


def _onnx2tf_supports(flag: str) -> bool:
    try:
        return flag in _onnx2tf_help_text()
    except Exception:
        return False


def _find_latest_onnx2tf_auto_json(out_dir: Path) -> Path | None:
    try:
        candidates = list(Path(out_dir).rglob("*_auto.json"))
    except Exception:
        return None
    if not candidates:
        return None
    try:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        pass
    return candidates[0]


def _run_onnx2tf(
    onnx_path: Path,
    out_dir: Path,
    *extra_args: str,
    dynamic_batch: bool = True,
    retry_auto_prf: bool = True,
    retry_batch_1: bool = True,
) -> None:
    out_dir = Path(out_dir)
    with contextlib.suppress(Exception):
        out_dir.mkdir(parents=True, exist_ok=True)
    base_args: list[str] = []
    if _onnx2tf_supports("-nuo"):
        base_args.append("-nuo")
    if dynamic_batch and _onnx2tf_supports("-osd"):
        base_args.append("-osd")

    def _cmd(*more: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(onnx_path),
            "-o",
            str(out_dir),
        ]
        cmd.extend([x for x in base_args if x])
        cmd.extend([x for x in extra_args if x])
        cmd.extend([x for x in more if x])
        return cmd

    last_exc: Exception | None = None
    auto_json: Path | None = None
    try:
        _in_console(_cmd(), "onnx2tf")
        return
    except Exception as exc:
        last_exc = exc

    if retry_auto_prf and _onnx2tf_supports("-prf"):
        auto_json = _find_latest_onnx2tf_auto_json(out_dir)
        if auto_json is not None:
            try:
                _in_console(_cmd("-prf", str(auto_json)), "onnx2tf")
                return
            except Exception as exc:
                last_exc = exc

    if dynamic_batch and retry_batch_1 and _onnx2tf_supports("-b"):
        try:
            _in_console(_cmd("-b", "1"), "onnx2tf")
            return
        except Exception as exc:
            last_exc = exc

        if auto_json is not None and _onnx2tf_supports("-prf"):
            try:
                _in_console(_cmd("-b", "1", "-prf", str(auto_json)), "onnx2tf")
                return
            except Exception as exc:
                last_exc = exc
    if last_exc is not None:
        raise last_exc


def _torch_export_program(
    torch_export: Callable[..., Any],
    wrapper: nn.Module,
    sample: torch.Tensor,
    *args: Any,
    dynamic_batch: bool = True,
    dynamic_seq: bool = False,
    strict: bool = True,
    tag: str = "torch.export",
) -> object:
    if isinstance(sample, torch.Tensor) and sample.ndim == 1:
        sample = sample.unsqueeze(0)
    if dynamic_batch and isinstance(sample, torch.Tensor) and sample.ndim >= 1:
        sample = _pad_to_batch(sample, 2)
    dynamic_shapes = None
    if (
        (dynamic_batch or dynamic_seq)
        and hasattr(torch, "export")
        and hasattr(torch.export, "Dim")
    ):
        spec: dict[int, object] = {}
        if dynamic_batch:
            spec[0] = torch.export.Dim("batch")
        if (
            dynamic_seq
            and isinstance(sample, torch.Tensor)
            and sample.ndim >= 2
        ):
            spec[1] = torch.export.Dim("seq")
        if spec:
            dynamic_shapes = (spec,)
    call_kw: dict[str, Any] = {}
    try:
        sig = inspect.signature(torch_export)
        params = sig.parameters
        if dynamic_shapes is not None and "dynamic_shapes" in params:
            call_kw["dynamic_shapes"] = dynamic_shapes
        if "strict" in params:
            call_kw["strict"] = bool(strict)
    except Exception:
        if dynamic_shapes is not None:
            call_kw["dynamic_shapes"] = dynamic_shapes
        call_kw["strict"] = bool(strict)

    def _call(**kw: Any) -> Any:
        with torch.no_grad():
            return torch_export(wrapper, (sample,), **kw)

    try:
        return _call(**call_kw)
    except TypeError as exc:
        msg = str(exc)
        stripped = dict(call_kw)
        retry = False
        for k in ("dynamic_shapes", "strict"):
            if k in stripped and f"'{k}'" in msg:
                stripped.pop(k, None)
                retry = True
        if retry:
            return _call(**stripped)
        raise
    except Exception:
        if call_kw.get("strict", False):
            call_kw["strict"] = False
            return _call(**call_kw)
        raise


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
        return set(sig.parameters.keys())
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
        prefer_dynamo: bool = False,
        simplify: bool = False,
        optimize_onnx: bool = True,
        onnxoptimizer_passes: Sequence[str] | None = None,
    ) -> object:
        is_required("onnx", "pip install onnx")
        wrapper = _TensorOutputModule(model).eval()
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        input_names = ["features"]
        dyn_axes = None
        dyn_shapes = None
        if dynamic_batch:
            dyn_axes = {"features": {0: "batch"}, "preds_flat": {0: "batch"}}
            if hasattr(torch, "export") and hasattr(torch.export, "Dim"):
                mode = (
                    os.environ.get("STNET_EXPORT_BATCH_DIM", "auto")
                    .strip()
                    .lower()
                )
                batch_dim = None
                if mode == "explicit":
                    try:
                        batch_dim = torch.export.Dim("batch", min=1)
                    except Exception:
                        try:
                            batch_dim = torch.export.Dim("batch")
                        except Exception:
                            batch_dim = None
                else:
                    try:
                        batch_dim = getattr(torch.export.Dim, "AUTO")
                    except Exception:
                        batch_dim = None
                if batch_dim is not None:
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
                (min_export_batch - int(sample.shape[0]),)
                + tuple(sample.shape[1:]),
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
                "do_constant_folding": False,
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
                        if (
                            use_dyn
                            and dyn_shapes
                            and "dynamic_shapes" in sig_keys
                        ):
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
                    if optimize_onnx:
                        if (
                            importlib.util.find_spec("onnx") is not None
                            and importlib.util.find_spec("onnxoptimizer")
                            is not None
                        ):
                            try:
                                import onnx
                                import onnxoptimizer

                                model_onnx = onnx.load(str(onnx_path))
                                passes = (
                                    list(onnxoptimizer_passes)
                                    if onnxoptimizer_passes is not None
                                    else None
                                )
                                model_opt = onnxoptimizer.optimize(
                                    model_onnx, passes
                                )
                                onnx.save(model_opt, str(onnx_path))
                            except Exception as exc:
                                warnings.warn(
                                    f"ONNX optimization failed; keeping the exported model. ({exc})"
                                )
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
    def coerce(
        model: nn.Module, onnx_path: PathLike, *args: Any, **kwargs: Any
    ) -> object:
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
        level = opt_map.get(
            optimization_level.lower(),
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        )
        disabled_optimizers: list[str] | None = None
        if level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL:
            disabled_optimizers = ["NchwcTransformer"]
        optimized_onnx_path = None

        if save_optimized_onnx_model:
            optimized_onnx_path = ort_path.with_suffix(".optimized.onnx")
            try:
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
            except Exception as exc:
                warnings.warn(
                    f"ORT optimized ONNX save failed; continuing without it. ({exc})",
                    RuntimeWarning,
                )
                optimized_onnx_path = None

        levels_to_try = [level]
        if level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL:
            levels_to_try.extend(
                [
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                ]
            )
        elif level == ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED:
            levels_to_try.extend(
                [
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                ]
            )
        elif level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC:
            levels_to_try.append(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

        last_exc: Exception | None = None
        for lvl in levels_to_try:
            try:
                so = ort.SessionOptions()
                so.optimized_model_filepath = str(ort_path)
                so.graph_optimization_level = lvl
                so.add_session_config_entry("session.save_model_format", "ORT")
                if optimization_style.lower() == "runtime":
                    so.add_session_config_entry(
                        "optimization.minimal_build_optimizations", "save"
                    )
                dis = (
                    disabled_optimizers
                    if lvl == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    else None
                )
                ort.InferenceSession(
                    str(onnx_path),
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                    disabled_optimizers=dis,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise RuntimeError("ORT export failed.") from last_exc

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

            exported = _torch_export_program(
                torch_export,
                wrapper,
                sample,
                dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
                dynamic_seq=bool(kwargs.get("dynamic_seq", False)),
                strict=bool(kwargs.get("strict", True)),
                tag="TorchInductor export",
            )

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
                    if (
                        inductor_configs is not None
                        and "inductor_configs" in params
                    ):
                        aoti_kw["inductor_configs"] = inductor_configs
            except Exception:
                aoti_kw["package_path"] = str(dst)
                if inductor_configs is not None:
                    aoti_kw["inductor_configs"] = inductor_configs

            try:
                out_path = aoti_compile_and_package(exported, **aoti_kw)
            except TypeError as exc:
                msg = str(exc)
                stripped = dict(aoti_kw)
                for k in ("package_path", "inductor_configs"):
                    if k in stripped and f"'{k}'" in msg:
                        stripped.pop(k, None)
                out_path = aoti_compile_and_package(exported, **stripped)

            out = Path(str(out_path))
            if out != dst:
                with contextlib.suppress(Exception):
                    shutil.copyfile(out, dst)
                out = dst

            with contextlib.suppress(Exception):
                _write_export_meta(model, out, format_name=self.name or "aoti")

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
            raise ImportError(
                "torch.export is required for PT2 export (PyTorch 2.0+)."
            ) from exc

        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                sample = sample.unsqueeze(0)
            wrapper = _TensorOutputModule(serving_model).eval()

            with _no_empty_tensor(serving_model):
                exported = self._export_program(wrapper, sample, **kwargs)

            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with from_buffer():
                torch_save(exported, str(dst))

            with contextlib.suppress(Exception):
                _write_export_meta(model, dst, format_name=self.name or "pt2")

        return (dst,)

    def _export_program(
        self, wrapper: nn.Module, sample: torch.Tensor, **kwargs: Any
    ) -> object:
        try:
            import torch.export

            torch_export = torch.export.export
        except Exception as exc:
            raise ImportError(
                "torch.export is required for PT2 export (PyTorch 2.0+)."
            ) from exc

        return _torch_export_program(
            torch_export,
            wrapper,
            sample,
            dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
            dynamic_seq=bool(kwargs.get("dynamic_seq", False)),
            strict=bool(kwargs.get("strict", True)),
            tag="PT2 export",
        )


class ExecuTorch(Format):
    name = "executorch"

    @staticmethod
    def _truthy_env(name: str) -> bool:
        v = os.environ.get(name, "")
        return v.strip().lower() in ("1", "true", "yes", "y", "on")

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

        dst = Path(dst)
        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            sample = _pad_sample(serving_model, sample)
            if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                sample = sample.unsqueeze(0)
            wrapper = _TensorOutputModule(serving_model).eval()
            dst.parent.mkdir(parents=True, exist_ok=True)

            dyn_batch = bool(kwargs.get("dynamic_batch", True))
            dyn_seq = bool(kwargs.get("dynamic_seq", False))
            strict = bool(kwargs.get("strict", True))
            silent_fallback = bool(
                kwargs.get("silent_fallback", False)
            ) or self._truthy_env("STNET_EXPORT_SILENT_FALLBACK")

            def _do_export(db: bool, ds: bool) -> object:
                return _torch_export_program(
                    torch_export,
                    wrapper,
                    sample,
                    dynamic_batch=db,
                    dynamic_seq=ds,
                    strict=strict,
                    tag="ExecuTorch export"
                    + (" (static)" if (not db and not ds) else ""),
                )

            try:
                exported = _do_export(dyn_batch, dyn_seq)
            except Exception:
                if dyn_batch or dyn_seq:
                    if os.environ.get(
                        "STNET_EXPORT_WARNINGS", ""
                    ).strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                    ):
                        warnings.warn(
                            "ExecuTorch export failed with dynamic shapes; retrying with dynamic_batch=False, dynamic_seq=False",
                            RuntimeWarning,
                        )
                    exported = _do_export(False, False)
                else:
                    raise

            def _to_executorch_program(exported_program: object) -> object:
                edge = exir.to_edge(exported_program)
                if hasattr(exir, "to_executorch"):
                    return exir.to_executorch(edge)
                if hasattr(edge, "to_executorch"):
                    return edge.to_executorch()
                if hasattr(edge, "to_executorch_program"):
                    return edge.to_executorch_program()
                raise AttributeError(
                    "ExecuTorch export: could not find `exir.to_executorch(edge)` nor `edge.to_executorch()`."
                )

            used_dyn_batch, used_dyn_seq = dyn_batch, dyn_seq
            fallback_error: str | None = None
            with _temp_environ(
                {"ET_EXIR_SAVE_FLATC_INPUTS_ON_FAILURE": "1"},
                only_if_unset=True,
            ):
                try:
                    exec_prog = _to_executorch_program(exported)
                    with open(dst, "wb") as fh:
                        exec_prog.write_to_file(fh)
                except Exception as exc:
                    if dyn_batch or dyn_seq:
                        used_dyn_batch, used_dyn_seq = False, False
                        fallback_error = repr(exc)
                        if not silent_fallback:
                            warnings.warn(
                                "ExecuTorch export with dynamic shapes failed; "
                                "falling back to a static program (dynamic_batch=False, dynamic_seq=False). "
                                "This static artifact may not accept variable-length inputs. "
                                "To silence this warning, pass silent_fallback=True or set "
                                "STNET_EXPORT_SILENT_FALLBACK=1.",
                                RuntimeWarning,
                            )
                        exported_static = _torch_export_program(
                            torch_export,
                            wrapper,
                            sample,
                            dynamic_batch=False,
                            dynamic_seq=False,
                            strict=strict,
                            tag="ExecuTorch export (static fallback)",
                        )
                        exec_prog = _to_executorch_program(exported_static)
                        with open(dst, "wb") as fh:
                            exec_prog.write_to_file(fh)
                    else:
                        raise
            with contextlib.suppress(Exception):
                _write_export_meta(
                    model,
                    dst,
                    format_name=self.name or "executorch",
                    extra={
                        "requested_dynamic_batch": dyn_batch,
                        "requested_dynamic_seq": dyn_seq,
                        "exported_dynamic_batch": used_dyn_batch,
                        "exported_dynamic_seq": used_dyn_seq,
                        "static_fallback": bool(
                            (dyn_batch or dyn_seq)
                            and (not used_dyn_batch and not used_dyn_seq)
                        ),
                        "fallback_error": fallback_error,
                    },
                )
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
        dst = Path(dst)
        with _onnx_model(model) as serving:
            out = _ONNXExporter.export(
                serving, dst, **_onnx_options(kwargs, target="onnx")
            )
        with contextlib.suppress(Exception):
            _write_export_meta(model, out, format_name=self.name or "onnx")
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
        dst = Path(dst)
        with _onnx_model(model) as serving:
            onnx_path = _ONNXExporter.coerce(
                serving,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="onnx"),
            )
            ort_path, optimized = _ORTBuilder.to_ort(
                onnx_path,
                dst,
                optimization_level=str(
                    kwargs.get("optimization_level", "all")
                ),
                optimization_style=str(
                    kwargs.get("optimization_style", "fixed")
                ),
                target_platform=kwargs.get("target_platform"),
                save_optimized_onnx_model=bool(
                    kwargs.get("save_optimized_onnx_model", False)
                ),
            )
        with contextlib.suppress(Exception):
            _write_export_meta(
                model,
                ort_path,
                format_name=self.name or "ort",
                extra={
                    "onnx_path": str(onnx_path),
                    "optimized_onnx": (
                        str(optimized) if optimized is not None else None
                    ),
                },
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
        dst = Path(dst)
        with _onnx_model(model) as serving_model:
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="tensorrt"),
            )
            if bool(kwargs.get("graphsurgeon", True)):
                if (
                    importlib.util.find_spec("onnx") is not None
                    and importlib.util.find_spec("onnx_graphsurgeon")
                    is not None
                ):
                    try:
                        import onnx
                        import onnx_graphsurgeon as gs

                        gs_model = onnx.load(str(onnx_path))
                        graph = gs.import_onnx(gs_model)
                        graph.cleanup().toposort()
                        onnx.save(gs.export_onnx(graph), str(onnx_path))
                    except Exception as exc:
                        warnings.warn(
                            f"TensorRT graphsurgeon optimization failed; using unoptimized ONNX. ({exc})"
                        )
            try:
                import tensorrt as trt
            except ImportError as exc:
                raise ImportError(
                    "TensorRT is required for this export."
                ) from exc
            trt_logger = trt.Logger(trt.Logger.WARNING)
            explicit_batch_flag = 0
            with contextlib.suppress(Exception):
                explicit_batch_flag = 1 << int(
                    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
                )
            with contextlib.ExitStack() as stack:
                builder = stack.enter_context(trt.Builder(trt_logger))
                create_network = getattr(builder, "create_network", None)
                if not callable(create_network):
                    raise RuntimeError(
                        "TensorRT builder.create_network is not available."
                    )
                try:
                    network = stack.enter_context(
                        create_network(explicit_batch_flag)
                    )
                except TypeError:
                    network = stack.enter_context(create_network())

                parser = stack.enter_context(
                    trt.OnnxParser(network, trt_logger)
                )
                config = stack.enter_context(builder.create_builder_config())

                workspace_size_bytes = int(
                    kwargs.get("workspace_size_bytes", 1 << 30)
                )
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
                        raise RuntimeError(
                            "TensorRT could not parse the ONNX model."
                        )
                use_fp16 = bool(kwargs.get("fp16", True))
                use_int8 = bool(kwargs.get("int8", False))
                if use_fp16 and bool(
                    getattr(builder, "platform_has_fast_fp16", False)
                ):
                    with contextlib.suppress(Exception):
                        config.set_flag(trt.BuilderFlag.FP16)
                if use_int8:
                    if not bool(
                        getattr(builder, "platform_has_fast_int8", False)
                    ):
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
                opt_batch = max(
                    min_batch, int(kwargs.get("opt_batch", shape[0]))
                )
                max_batch = max(opt_batch, int(kwargs.get("max_batch", 8)))
                min_shape = (min_batch, *shape[1:])
                opt_shape = (opt_batch, *shape[1:])
                max_shape = (max_batch, *shape[1:])
                profile = builder.create_optimization_profile()
                profile.set_shape(
                    input_tensor.name, min_shape, opt_shape, max_shape
                )
                config.add_optimization_profile(profile)
                engine_blob = None
                build_serialized = getattr(
                    builder, "build_serialized_network", None
                )
                if callable(build_serialized):
                    engine_blob = build_serialized(network, config)
                else:
                    build_engine = getattr(
                        builder, "build_engine", None
                    ) or getattr(builder, "build_cuda_engine", None)
                    if callable(build_engine):
                        engine = build_engine(network, config)
                        if engine is not None:
                            engine_blob = getattr(
                                engine, "serialize", lambda: None
                            )()

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

                with contextlib.suppress(Exception):
                    _write_export_meta(
                        model,
                        dst,
                        format_name=self.name or "tensorrt",
                        extra={"onnx_path": str(onnx_path)},
                    )
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
        if sys.platform != "darwin":
            raise ImportError(
                "coremltools is required for this operation (try: pip install coremltools on macOS)"
            )
        spec = importlib.util.find_spec("coremltools")
        if spec is None or not getattr(
            spec, "submodule_search_locations", None
        ):
            raise ImportError(
                "coremltools is required for this operation (try: pip install coremltools on macOS)"
            )
        pkg_dirs = list(spec.submodule_search_locations or [])
        has_native = any(
            glob.glob(os.path.join(d, "libcoremlpython*")) for d in pkg_dirs
        )
        if not has_native:
            raise ImportError(
                "coremltools is required for this operation (try: pip install coremltools on macOS)"
            )
        import coremltools as ct

        dst = Path(dst)

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
                "inputs": [
                    ct.TensorType(shape=tuple((int(x) for x in sample.shape)))
                ],
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
            with contextlib.suppress(Exception):
                _write_export_meta(
                    model, dst, format_name=self.name or "coreml"
                )
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
        del args
        dst = Path(dst)
        prefer_ai_edge = bool(
            kwargs.get(
                "prefer_ai_edge_torch", kwargs.get("prefer_ai_edge", True)
            )
        )
        have_ai_edge = importlib.util.find_spec("ai_edge_torch") is not None
        have_onnx2tf = importlib.util.find_spec("onnx2tf") is not None

        with _onnx_model(model) as serving_model:
            if prefer_ai_edge and have_ai_edge:
                try:
                    import ai_edge_torch

                    sample = kwargs.get("sample_input")
                    sample = _pad_sample(serving_model, sample, batch=1)
                    if isinstance(sample, torch.Tensor) and sample.ndim == 1:
                        sample = sample.unsqueeze(0)

                    wrapper = _TensorOutputModule(serving_model).eval()
                    edge_model = ai_edge_torch.convert(wrapper, (sample,))
                    exporter = getattr(edge_model, "export", None)
                    if not callable(exporter):
                        raise AttributeError(
                            "ai-edge-torch conversion did not return an object with .export()"
                        )
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    exporter(str(dst))

                    with contextlib.suppress(Exception):
                        _write_export_meta(
                            model,
                            dst,
                            format_name=self.name or "litert",
                            extra={"backend": "ai-edge-torch"},
                        )
                    return (dst,)
                except Exception as exc:
                    if have_onnx2tf and bool(
                        kwargs.get("fallback_onnx2tf", True)
                    ):
                        warnings.warn(
                            f"ai-edge-torch conversion failed; retrying via onnx2tf: {exc}",
                            RuntimeWarning,
                        )
                    else:
                        raise

            if not have_onnx2tf:
                raise ImportError(
                    "LiteRT export requires ai-edge-torch (preferred) or onnx2tf (fallback). "
                    "Try: pip install ai-edge-torch (or: pip install onnx2tf)"
                )
            is_required("onnx2tf", "pip install onnx2tf")

            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="litert"),
            )

            work_dir = dst.with_name(dst.name + ".onnx2tf")
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            work_dir.mkdir(parents=True, exist_ok=True)

            _run_onnx2tf(
                Path(onnx_path),
                work_dir,
                "--copy_onnx_input_output_names_to_tflite",
                dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
            )
            tflites = list(work_dir.rglob("*.tflite"))
            if not tflites:
                raise RuntimeError("onnx2tf did not produce a .tflite model.")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tflites[0], dst)

            with contextlib.suppress(Exception):
                _write_export_meta(
                    model,
                    dst,
                    format_name=self.name or "litert",
                    extra={"backend": "onnx2tf", "onnx_path": str(onnx_path)},
                )
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
        del args
        is_required("onnx2tf", "pip install onnx2tf")
        dst = Path(dst)
        suffix = dst.suffix.lower()
        if suffix == ".savedmodel":
            saved_model_dir = dst
        elif suffix in {".pb", ".tf"}:
            saved_model_dir = dst.with_suffix("")
        else:
            saved_model_dir = dst

        with _onnx_model(model) as serving_model:
            onnx_path = _ONNXExporter.coerce(
                serving_model,
                _coerce_onnx_path(saved_model_dir, kwargs),
                **_onnx_options(kwargs, target="tensorflow"),
            )

            with TemporaryDirectory() as tmpd:
                tmp_out = Path(tmpd) / "onnx2tf_out"
                tmp_out.mkdir(parents=True, exist_ok=True)
                _run_onnx2tf(
                    Path(onnx_path),
                    tmp_out,
                    dynamic_batch=bool(kwargs.get("dynamic_batch", True)),
                )
                pb_files = list(tmp_out.rglob("saved_model.pb"))
                if not pb_files:
                    raise RuntimeError(
                        "onnx2tf did not produce a TensorFlow SavedModel (saved_model.pb)"
                    )
                src_dir = pb_files[0].parent
                if saved_model_dir.exists():
                    shutil.rmtree(saved_model_dir)
                saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_dir, saved_model_dir)

        with contextlib.suppress(Exception):
            _write_export_meta(
                model,
                saved_model_dir,
                format_name=self.name or "tensorflow",
                extra={"backend": "onnx2tf", "onnx_path": str(onnx_path)},
            )

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

        self._out_shape_kind, self._out_shape_spec = self._normalize_out_shape(
            out_shape
        )

        compiled_steps: list[tuple[object, ...]] = []
        for raw in steps:
            step, extra_args, extra_kwargs = self._parse_step(raw)
            meta: dict[str, Any] | None = None

            if isinstance(step, BorrowedModule):
                if step.name:
                    meta = {"name": str(step.name)}
                compiled_steps.append(
                    (
                        "ref",
                        weakref.ref(step.module),
                        extra_args,
                        extra_kwargs,
                        meta,
                    )
                )
                continue
            if isinstance(step, ModulePath):
                meta = {
                    "path": str(step.path),
                    "name": (str(step.name) if step.name else None),
                }
                compiled_steps.append(
                    ("path", str(step.path), extra_args, extra_kwargs, meta)
                )
                continue
            if isinstance(step, OwnedModule):
                if step.name:
                    meta = {"name": str(step.name)}
                self._owned.append(step.module)
                compiled_steps.append(
                    (
                        "owned",
                        len(self._owned) - 1,
                        extra_args,
                        extra_kwargs,
                        meta,
                    )
                )
                continue
            if isinstance(step, nn.Module):
                compiled_steps.append(
                    ("ref", weakref.ref(step), extra_args, extra_kwargs, meta)
                )
                continue
            if callable(step):
                tag = getattr(step, self._CONTROL_ATTR, None)
                if tag is not None:
                    meta = {"control": str(tag)}
                compiled_steps.append(
                    ("fn", step, extra_args, extra_kwargs, meta)
                )
                continue

            raise TypeError(
                f"Unsupported GraphSequential step: {type(step)!r}"
            )

        if not compiled_steps:
            raise ValueError("GraphSequential requires at least one step.")
        self._steps: list[tuple[object, ...]] = compiled_steps

    @staticmethod
    def ref(
        module: nn.Module, *args: Any, name: str | None = None
    ) -> BorrowedModule:
        del args
        return BorrowedModule(module=module, name=name)

    @staticmethod
    def own(
        module: nn.Module, *args: Any, name: str | None = None
    ) -> OwnedModule:
        del args
        return OwnedModule(module=module, name=name)

    @staticmethod
    def path(path: str, *args: Any, name: str | None = None) -> ModulePath:
        return ModulePath(path=str(path), name=name)

    @staticmethod
    def mean(dim: int = 1, *args: Any, keepdim: bool = False) -> OwnedModule:
        del args
        return OwnedModule(
            module=ReduceMean(dim=int(dim), keepdim=bool(keepdim)), name="mean"
        )

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
    def cudagraph_begin(
        *args: Any, disable_compile: bool = True
    ) -> Callable[..., Any]:
        from ..core.graph import (
            cudagraph_mark_step_begin,
            torch_compiler_disable,
        )

        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_begin()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        _op = GraphSequential._tag_control(_op, "cudagraph_begin")
        return (
            torch_compiler_disable(
                _op, reason="subgraph:cudagraph_begin", recursive=False
            )
            if disable_compile
            else _op
        )

    @staticmethod
    def cudagraph_end(
        *args: Any, disable_compile: bool = True
    ) -> Callable[..., Any]:
        from ..core.graph import (
            cudagraph_mark_step_end,
            torch_compiler_disable,
        )

        def _op(*a: Any, **kw: Any) -> Any:
            cudagraph_mark_step_end()
            if kw:
                return CallArguments(args=tuple(a), kwargs=dict(kw))
            if len(a) == 1:
                return a[0]
            return tuple(a)

        _op = GraphSequential._tag_control(_op, "cudagraph_end")
        return (
            torch_compiler_disable(
                _op, reason="subgraph:cudagraph_end", recursive=False
            )
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

    def bind(
        self, root: nn.Module | None = None, *args: Any, strict: bool = True
    ) -> "GraphSequential":
        if root is not None:
            self.set_root(root)

        rebound: list[tuple[object, ...]] = []
        for item in list(self._steps):
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(
                item
            )

            if kind == "path":
                path = str(payload)
                mod = self._resolve_path(path)
                m = dict(meta) if isinstance(meta, dict) else {}
                m["path"] = path
                rebound.append(
                    ("ref", weakref.ref(mod), extra_args, extra_kwargs, m)
                )
                continue

            if kind == "ref" and payload is None:
                path = meta.get("path") if isinstance(meta, dict) else None
                if isinstance(path, str):
                    mod = self._resolve_path(path)
                    rebound.append(
                        (
                            "ref",
                            weakref.ref(mod),
                            extra_args,
                            extra_kwargs,
                            meta,
                        )
                    )
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
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(
                item
            )
            cur = self._apply_step(
                kind, payload, cur, extra_args, extra_kwargs, meta=meta
            )

        return self._apply_out_shape(cur)

    def extra_repr(self) -> str:
        return f"name={self._name!r}, out_shape={self._out_shape_spec!r}, steps={len(self._steps)}"

    def __getstate__(self) -> dict[str, object]:
        state = super().__getstate__()

        steps = state.get("_steps", [])
        sanitized: list[tuple[object, ...]] = []
        if isinstance(steps, list):
            for item in steps:
                kind, payload, extra_args, extra_kwargs, meta = (
                    self._split_step(item)
                )
                if kind == "ref":
                    sanitized.append(
                        (kind, None, extra_args, extra_kwargs, meta)
                    )
                else:
                    sanitized.append(
                        (kind, payload, extra_args, extra_kwargs, meta)
                    )
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
    def _parse_step(
        raw: object,
    ) -> tuple[object, tuple[Any, ...], dict[str, Any]]:
        if isinstance(raw, (tuple, list)):
            if len(raw) == 2 and isinstance(raw[1], dict):
                return raw[0], (), dict(raw[1])
            if (
                len(raw) == 3
                and isinstance(raw[1], (tuple, list))
                and isinstance(raw[2], dict)
            ):
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
            extra_kwargs = (
                dict(raw_kwargs) if hasattr(raw_kwargs, "items") else {}
            )

        meta = item[4] if len(item) >= 5 else None
        return kind, payload, extra_args, extra_kwargs, meta

    @staticmethod
    def _normalize_out_shape(
        out_shape: object | None,
    ) -> tuple[str | None, object | None]:
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
                    raise AttributeError(
                        f"Failed to resolve submodule path {path!r} at {part!r}."
                    )
                cur = nxt
            mod = cur

        if not isinstance(mod, nn.Module):
            raise TypeError(
                f"get_submodule({path!r}) did not return an nn.Module"
            )

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
                raise TypeError(
                    f"GraphSequential ref step did not resolve to nn.Module: {type(mod)!r}"
                )
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

        def _reshape_one(
            t: torch.Tensor, shape: tuple[int, ...]
        ) -> torch.Tensor:
            if t.ndim == 0:
                raise RuntimeError(
                    "Cannot reshape a scalar output in GraphSequential."
                )
            return t.reshape(t.shape[0], *shape)

        if kind == "single":
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(
                    "out_shape is set but the pipeline output is not a Tensor."
                )
            return _reshape_one(out, spec)

        if kind == "seq":
            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    "out_shape expects tuple/list output but got a different type."
                )
            shapes = list(spec)
            if len(out) != len(shapes):
                raise RuntimeError(
                    "out_shape length does not match tuple/list output length."
                )
            out_list = list(out)
            for i, sh in enumerate(shapes):
                if sh is None:
                    continue
                if not isinstance(out_list[i], torch.Tensor):
                    raise RuntimeError(
                        "out_shape expects Tensor outputs in tuple/list."
                    )
                out_list[i] = _reshape_one(out_list[i], sh)
            return tuple(out_list) if isinstance(out, tuple) else out_list
        if not isinstance(out, dict):
            raise RuntimeError(
                "out_shape expects dict output but got a different type."
            )
        out_dict = dict(out)
        for k, sh in spec.items():
            if sh is None:
                continue
            if k not in out_dict:
                raise RuntimeError(
                    f"out_shape missing key in output dict: {k!r}"
                )
            if not isinstance(out_dict[k], torch.Tensor):
                raise RuntimeError(
                    "out_shape expects Tensor values in dict output."
                )
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
            kind, payload, extra_args, extra_kwargs, meta = self._split_step(
                item
            )

            if kind == "fn":
                fn = payload
                if strip_control_ops and bool(
                    getattr(fn, self._CONTROL_ATTR, "")
                ):
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
                    if (
                        mod is None
                        and isinstance(meta, dict)
                        and isinstance(meta.get("path"), str)
                    ):
                        mod = self._resolve_path(str(meta.get("path")))
                else:
                    raise TypeError(
                        f"Unknown GraphSequential step kind: {kind!r}"
                    )

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
            out_shape=(
                self._out_shape_spec
                if self._out_shape_kind is not None
                else None
            ),
            name=str(name or f"{self._name}_serving"),
            root=None,
        )
        out.eval()
        with contextlib.suppress(Exception):
            out.requires_grad_(False)
        with contextlib.suppress(Exception):
            setattr(out, "__compiled_for_serving__", True)
        return out
