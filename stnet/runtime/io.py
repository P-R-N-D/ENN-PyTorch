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
import tempfile
import threading
import warnings
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Protocol, Sequence

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from tensordict import TensorDictBase

from ..core.compat import is_meta_or_fake_tensor
from ..data.schemas import save_temp, write_json, coerce_json
from ..nn.layers import Recorder

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None


_IGNORED_WARNINGS = ("torch.distributed is disabled", "TypedStorage is deprecated")
_IGNORED_RE = r".*(?:" + "|".join(re.escape(s) for s in _IGNORED_WARNINGS) + r").*"

_FORWARD_PARAM_CACHE: dict[object, object] = {}
_FORWARD_PARAM_CACHE_LOCK = threading.Lock()

_WARNINGS_FILTER_LOCK = threading.Lock()

_SAVE_LOCK_GUARD = threading.Lock()
_SAVE_PATH_LOCKS = weakref.WeakValueDictionary()


def _register_safe_globals():
    with contextlib.suppress(Exception):
        if add_safe_globals:
            from torch.torch_version import TorchVersion

            add_safe_globals([TorchVersion])


@contextlib.contextmanager
def _filtered_warnings(sentences: Sequence[str] | None = None) -> Iterator[None]:
    msg_re = (
        _IGNORED_RE
        if sentences is None
        else (
            r".*(?:" + "|".join(re.escape(str(s)) for s in sentences) + r").*" if sentences else ""
        )
    )
    if not msg_re:
        yield
        return
    guard = _WARNINGS_FILTER_LOCK
    if getattr(sys.flags, "context_aware_warnings", False):
        guard = contextlib.nullcontext()
    with guard:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg_re)
            yield


@contextlib.contextmanager
def _temp_environ(
    updates: dict[str, str | None], *args: Any, only_if_unset: bool = True
) -> Iterator[None]:
    prev: dict[str, str | None] = {}
    for key, val in updates.items():
        if only_if_unset and key in os.environ:
            continue
        prev[key] = os.environ.get(key)
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(val)
    try:
        yield
    finally:
        for key, val in prev.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def _save_lock(path: PathLike | None = None) -> threading.RLock:
    try:
        key = str(Path(path).expanduser().resolve()) if path else "__global__"
    except Exception:
        key = str(path)
    with _SAVE_LOCK_GUARD:
        return _SAVE_PATH_LOCKS.setdefault(key, threading.RLock())


def _dist_op(op_name: str) -> object:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return getattr(dist, op_name)()
    except Exception:
        pass
    return None


def _is_rank0_global() -> bool:
    return _dist_op("get_rank") in (0, None)


def _dist_barrier() -> None:
    _dist_op("barrier")


@contextlib.contextmanager
def _save_sync(path: PathLike | None = None, *args: Any, barrier: bool = False) -> None:
    with _save_lock(path):
        if barrier:
            _dist_barrier()
        try:
            yield
        finally:
            if barrier:
                _dist_barrier()


def _load_model_config(model: object) -> object:
    try:
        from ..config import _extract_model_config_dict

        return _extract_model_config_dict(model)
    except Exception:
        return {}


def _to_tensor(
    value: object,
    *args: Any,
    materialize_meta: bool = True,
    make_contiguous: bool = True,
) -> object:
    if isinstance(value, torch.Tensor):
        t = value.to_local() if hasattr(value, "to_local") else value
        if materialize_meta and is_meta_or_fake_tensor(t):
            t = torch.zeros(t.shape, dtype=t.dtype, device="cpu")
        t = t.detach()
        if t.device.type != "cpu":
            t = t.to(device="cpu")
        if make_contiguous and not t.is_contiguous():
            t = t.contiguous()
        return t
    if isinstance(value, (list, tuple)):
        out = [
            _to_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous)
            for v in value
        ]
        return type(value)(*out) if hasattr(value, "_fields") else type(value)(out)
    if isinstance(value, dict):
        return type(value)(
            (
                k,
                _to_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous),
            )
            for k, v in value.items()
        )
    return value


def _torch_load_checkpoint(
    path: PathLike, *args: Any, map_location: object = None, weights_only: bool = True
) -> object:
    try:
        return torch.load(str(path), map_location=map_location or "cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(str(path), map_location=map_location or "cpu")
    except Exception as exc:
        if weights_only:
            raise RuntimeError("weights_only=True failed") from exc
        raise


def _in_console(cmd: object, desc: object) -> None:
    try:
        subprocess.run(list(cmd), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{desc} failed with error: {exc}") from exc


@contextlib.contextmanager
def _onnx_model(model: object) -> None:
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


def _extract_tensor(out: object) -> torch.Tensor:
    if isinstance(out, TensorDictBase):
        y = out.get("pred", None)
        if not isinstance(y, torch.Tensor):
            y = next((v for v in out.values() if isinstance(v, torch.Tensor)), None)
        if isinstance(y, torch.Tensor):
            return y
        raise RuntimeError("Failed to extract tensor from TensorDict output.")
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        if isinstance(out[0], torch.Tensor):
            return out[0]
        tensor = next((v for v in out if isinstance(v, torch.Tensor)), None)
        if tensor is not None:
            return tensor
    raise RuntimeError("Model forward did not return a tensor output.")


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
            y_flat = _extract_tensor(
                _forward(model, sample.unsqueeze(0) if sample.ndim == 1 else sample)
            )
        in_dim = in_dim or int(sample.numel() // sample.shape[0])
        out_shape = out_shape or tuple(y_flat.shape[1:])
    if in_dim is None or out_shape is None:
        raise RuntimeError("Failed to infer shapes.")
    return (int(in_dim), tuple(out_shape))


def _pad_sample(model: object, sample_input: object) -> object:
    if sample_input is not None:
        return sample_input
    in_dim, _ = _get_tensor_shape(model, sample_input)
    param = next(model.parameters(), None)
    dtype, device = (
        (param.dtype, param.device) if param is not None else (torch.float32, torch.device("cpu"))
    )
    return torch.zeros(1, in_dim, dtype=dtype, device=device)


def _onnx_options(kwargs: object, *args: Any, target: str = "onnx") -> object:
    target_l = str(target or "onnx").strip().lower()
    defaults = {
        "tensorrt": (17, True, False, True, [17, 16, 15, 14, 13]),
        "tensorflow": (15, False, False, True, [15, 13]),
        "nnef": (13, False, False, True, [13]),
        "default": (18, True, True, False, [18, 17, 16, 15, 13]),
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


def is_required(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise ImportError(f"{module} is required for this operation{hint}") from err


class _CompatLayer(nn.Module):
    def __init__(self, net: object) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: object) -> object:
        return _extract_tensor(_forward(self.net, x))


class _OnnxLayer:
    @staticmethod
    def export(
        model: nn.Module,
        onnx_path: PathLike,
        *args: Any,
        sample_input: object | None = None,
        opset_version: int = 18,
        opset_fallback: Sequence[int] | None = None,
        dynamic_batch: bool = True,
        prefer_dynamo: bool = True,
        simplify: bool = False,
    ) -> object:
        is_required("onnx", "pip install onnx")
        wrapper = _CompatLayer(model).eval()
        sample = _pad_sample(model, sample_input)
        if isinstance(sample, torch.Tensor) and sample.ndim == 1:
            sample = sample.unsqueeze(0)
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        input_names = ["features"]
        dyn_axes, dyn_shapes = (
            (
                {"features": {0: "batch"}, "preds_flat": {0: "batch"}},
                {"features": {0: torch.export.Dim("batch")}},
            )
            if dynamic_batch
            else (None, None)
        )
        if not (hasattr(torch, "export") and hasattr(torch.export, "Dim")):
            dyn_shapes = None
        training = None
        with contextlib.suppress(Exception):
            training = torch.onnx.TrainingMode.EVAL
        sig_keys = Exporter._export_sig_keys()
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
                        args = sample if isinstance(sample, (list, tuple)) else (sample,)
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
            return Exporter._OnnxLayer.export(model, onnx_path, **kwargs)
        return onnx_path


class _OrtLayer:
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


class Format(Protocol):
    name = None

    def save(self, model: nn.Module, dst: PathLike, *args: Any, **kwargs: Any) -> None: ...


class Builder:
    NATIVE_EXTS = {".pt", ".pth", ".safetensors"}

    @staticmethod
    def is_target_native(path: PathLike) -> bool:
        suffix = Path(path).suffix.lower()
        return not suffix or suffix in Builder.NATIVE_EXTS

    @staticmethod
    def save(
        model: object,
        path: PathLike,
        optimizer: object | None = None,
        extra: object | None = None,
        **opts: Any,
    ) -> object:
        p = Path(path)

        def _make_meta() -> dict[str, Any]:
            return {
                "version": 1,
                "in_dim": int(getattr(model, "in_dim", 0)),
                "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
                "config": _load_model_config(model),
                "pytorch_version": torch.__version__,
                "extra": coerce_json(extra or {}),
            }

        if not p.suffix and p.exists() and p.is_dir():
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint import save as dcp_save

            with _save_sync(p, barrier=True):
                dcp_save(
                    state_dict={
                        "model": get_model_state_dict(
                            model, options=StateDictOptions(full_state_dict=True)
                        )
                    },
                    storage_writer=FileSystemWriter(str(p)),
                )
                if _is_rank0_global():
                    write_json(p / "meta.json", {**_make_meta(), "format": "dcp-dir-v1"}, indent=2)
            return p
        if not p.suffix:
            p = p.with_suffix(".pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        with _save_sync(p, barrier=False):
            if not _is_rank0_global():
                return p
            if p.suffix == ".safetensors":
                is_required("safetensors", "pip install safetensors")
                from safetensors.torch import save_file as save_tensors

                fd, tmp_name = tempfile.mkstemp(
                    prefix=p.name + ".", suffix=p.suffix + ".tmp", dir=str(p.parent)
                )
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    save_tensors(
                        {k: _to_tensor(v) for k, v in model.state_dict().items()},
                        str(tmp_path),
                        metadata={"format": "safetensors-v1"},
                    )
                    tmp_path.replace(p)
                finally:
                    with contextlib.suppress(Exception):
                        tmp_path.unlink() if tmp_path.exists() else None
                write_json(p.with_suffix(".json"), _make_meta(), indent=2)
                return p
            payload = {**_make_meta(), "state_dict": model.state_dict()}
            if optimizer is not None:
                with contextlib.suppress(Exception):
                    payload["optimizer_state_dict"] = optimizer.state_dict()
            save_temp(p, payload, **opts)
            return p


class Exporter:
    _by_name: dict[str, Format] = {}
    _ext_map: dict[str, str] = {}
    _defaults_registered: bool = False
    _defaults_lock = threading.Lock()
    _OnnxLayer = _OnnxLayer
    _OrtLayer = _OrtLayer
    _export_sig_cache: object | None = None
    _export_sig_lock = threading.Lock()

    @classmethod
    def _export_sig(cls) -> object:
        cached = getattr(cls, "_export_sig_cache", None)
        if cached is not None:
            return cached
        with cls._export_sig_lock:
            cached = getattr(cls, "_export_sig_cache", None)
            if cached is not None:
                return cached
            try:
                sig = inspect.signature(torch.onnx.export)
            except Exception:
                sig = None
            cls._export_sig_cache = sig
            return sig

    @classmethod
    def _export_sig_keys(cls) -> set[str]:
        sig = cls._export_sig()
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

    @classmethod
    def register(cls, name: str, exts: tuple[str, ...], impl: Format) -> None:
        cls._by_name[name] = impl
        for ext in exts:
            cls._ext_map[ext.lower()] = name

    @classmethod
    def _ensure_defaults_registered(cls) -> None:
        if cls._defaults_registered:
            return
        with cls._defaults_lock:
            if cls._defaults_registered:
                return
            cls.register("onnx", (".onnx",), Onnx())
            cls.register("ort", (".ort",), Ort())
            cls.register("tensorrt", (".engine",), TensorRT())
            cls.register("nnef", (".nnef",), Nnef())
            cls.register("coreml", (".mlmodel",), CoreML())
            cls.register("litert", (".tflite",), LiteRT())
            cls.register("torchscript", (".ts", ".torchscript"), TorchScript())
            cls.register("executorch", (".pte",), ExecuTorch())
            cls.register("tensorflow", (".savedmodel", ".pb", ".tf"), TensorFlow())
            cls._defaults_registered = True

    @classmethod
    def for_export(cls, ext: str) -> Format | None:
        cls._ensure_defaults_registered()
        name = cls._ext_map.get(ext.lower())
        return cls._by_name.get(name) if name else None


class Onnx(Format):
    name = "onnx"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        with _onnx_model(model) as serving:
            out = Exporter._OnnxLayer.export(serving, dst, **_onnx_options(kwargs, target="onnx"))
        return (out,)


class Ort(Format):
    name = "ort"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        with _onnx_model(model) as serving:
            onnx_path = Exporter._OnnxLayer.coerce(
                serving,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="onnx"),
            )
            ort_path, optimized = Exporter._OrtLayer.to_ort(
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
            onnx_path = Exporter._OnnxLayer.coerce(
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
                min_shape = (1, *shape[1:])
                opt_shape = (int(kwargs.get("opt_batch", shape[0])), *shape[1:])
                max_shape = (int(kwargs.get("max_batch", 8)), *shape[1:])
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


class Nnef(Format):
    name = "nnef"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        with _onnx_model(model) as serving_model:
            onnx_path = Exporter._OnnxLayer.coerce(
                serving_model,
                _coerce_onnx_path(dst, kwargs),
                **_onnx_options(kwargs, target="nnef"),
            )
            try:
                import importlib

                importlib.import_module("nnef_tools.convert")
            except ImportError as exc:
                raise ImportError("nnef-tools[onnx] is required for this export.") from exc
            input_shapes = kwargs.get("input_shapes")
            if input_shapes is None:
                sample = _pad_sample(serving_model, kwargs.get("sample_input"))
                input_shapes = {"features": tuple((int(x) for x in sample.shape))}
            import json

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
            wrapper = _CompatLayer(serving_model).eval()
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
            onnx_path = Exporter._OnnxLayer.coerce(
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


class TorchScript(Format):
    name = "torchscript"

    def save(
        self,
        model: nn.Module,
        dst: PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        method = str(kwargs.get("method", "script")).lower()
        with _onnx_model(model) as serving_model:
            sample = kwargs.get("sample_input")
            wrapper = _CompatLayer(serving_model).eval()
            if "method" not in kwargs and hasattr(serving_model, "forward_export"):
                method = "trace"
            if method == "trace":
                if sample is None:
                    sample = _pad_sample(serving_model, None)
                with torch.no_grad():
                    scripted = torch.jit.trace(
                        wrapper,
                        sample,
                        check_trace=False,
                        strict=False,
                    )
            else:
                try:
                    with torch.no_grad():
                        scripted = torch.jit.script(wrapper)
                except Exception as exc:
                    if sample is None:
                        sample = _pad_sample(serving_model, None)
                    warnings.warn(
                        f"TorchScript scripting failed ({type(exc).__name__}: {exc}); falling back to tracing.",
                        RuntimeWarning,
                    )
                    with torch.no_grad():
                        scripted = torch.jit.trace(
                            wrapper,
                            sample,
                            check_trace=False,
                            strict=False,
                        )
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
            wrapper = _CompatLayer(serving_model).eval()
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
            onnx_path = Exporter._OnnxLayer.coerce(
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


_register_safe_globals()
