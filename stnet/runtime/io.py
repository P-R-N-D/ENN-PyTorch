# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
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
from types import ModuleType
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Protocol, Sequence, TypeAlias

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from tensordict import TensorDictBase

from ..core.compat import is_meta_or_fake_tensor
from ..core.graph import inference_mode
from ..data.schemas import save_temp, write_json, coerce_json
from ..nn.layers import Recorder

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None


PathLike: TypeAlias = str | os.PathLike[str] | Path
TorchDeviceLike: TypeAlias = torch.device | str | int

_IGNORED_WARNING_SENTENCES: tuple[str, ...] = (
    "torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.",
    "TypedStorage is deprecated. It will be removed in the future",
)

_IGNORED_WARNING_MESSAGE_RE: str = (
    r".*(?:" + "|".join((re.escape(str(s)) for s in _IGNORED_WARNING_SENTENCES)) + r").*"
)

_FORWARD_PARAM_CACHE: dict[object, object] = {}
_FORWARD_PARAM_CACHE_LOCK = threading.Lock()

_WARNINGS_FILTER_LOCK = threading.Lock()

_SAVE_LOCK_GUARD = threading.Lock()
_SAVE_PATH_LOCKS = weakref.WeakValueDictionary()


def _register_torchversion_safe_global() -> None:
    if add_safe_globals is None:
        return
    with contextlib.suppress(Exception):
        from torch.torch_version import TorchVersion
        add_safe_globals([TorchVersion])


@contextlib.contextmanager
def _filtered_warnings(ignored_sentences: Sequence[str] | None = None) -> Iterator[None]:
    sentences = _IGNORED_WARNING_SENTENCES if ignored_sentences is None else tuple(ignored_sentences)
    if not sentences:
        yield
        return
    msg_re = _IGNORED_WARNING_MESSAGE_RE
    if ignored_sentences is not None:
        parts = [re.escape(str(s)) for s in sentences if s]
        msg_re = r".*(?:" + "|".join(parts) + r").*" if parts else ""
    ctx_aware = False
    with contextlib.suppress(Exception):
        if sys.version_info >= (3, 14):
            ctx_aware = bool(getattr(getattr(sys, "flags", None), "context_aware_warnings", False))
    guard = contextlib.nullcontext() if ctx_aware else _WARNINGS_FILTER_LOCK
    with guard:
        with warnings.catch_warnings():
            with contextlib.suppress(Exception):
                if msg_re:
                    warnings.filterwarnings("ignore", message=str(msg_re))
            yield


@contextlib.contextmanager
def _temp_environ(updates: dict[str, str | None], *args: Any, only_if_unset: bool = True) -> Iterator[None]:
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
    if path is None:
        key = "__global__"
    else:
        try:
            key = str(Path(path).expanduser().resolve())
        except Exception:
            key = str(path)
    with _SAVE_LOCK_GUARD:
        lk = _SAVE_PATH_LOCKS.get(key)
        if lk is None:
            lk = threading.RLock()
            _SAVE_PATH_LOCKS[key] = lk
        return lk


def _get_dist() -> ModuleType | None:
    try:
        import torch.distributed as dist
    except Exception:
        return None
    return dist


def _is_rank0_global() -> bool:
    dist = _get_dist()
    if dist is None:
        return True
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
    except Exception:
        pass
    return True


def _dist_barrier() -> None:
    dist = _get_dist()
    if dist is None:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


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
        t = value
        with contextlib.suppress(Exception):
            from torch.distributed.tensor import DTensor
            if isinstance(t, DTensor):
                t = t.to_local()
        if materialize_meta and is_meta_or_fake_tensor(t):
            t = torch.zeros(tuple(t.shape), dtype=t.dtype, device="cpu")
        if t.device.type != "cpu":
            t = t.detach().to(device="cpu")
        else:
            t = t.detach()
        if make_contiguous:
            with contextlib.suppress(Exception):
                if hasattr(t, "is_contiguous") and (not bool(t.is_contiguous())):
                    t = t.contiguous()
        return t
    if isinstance(value, dict):
        return type(value)(
            (
                (k, _to_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous))
                for k, v in value.items()
            )
        )
    if isinstance(value, list):
        return [
            _to_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous)
            for v in value
        ]
    if isinstance(value, tuple):
        seq = tuple(
            (
                _to_tensor(v, materialize_meta=materialize_meta, make_contiguous=make_contiguous)
                for v in value
            )
        )
        if type(value) is tuple:
            return seq
        if hasattr(value, "_fields"):
            with contextlib.suppress(Exception):
                return type(value)(*seq)
        with contextlib.suppress(Exception):
            return type(value)(seq)
        return seq
    return value


def _torch_load_checkpoint(
    path: PathLike,
    *args: Any,
    map_location: TorchDeviceLike | None = None,
    weights_only: bool = True,
) -> object:
    p = Path(path)
    try:
        return torch.load(
            str(p), map_location=map_location or "cpu", weights_only=bool(weights_only)
        )
    except TypeError:
        return torch.load(str(p), map_location=map_location or "cpu")
    except Exception as exc:
        if weights_only:
            raise RuntimeError(
                "Failed to load checkpoint with weights_only=True. If you trust the checkpoint source, retry with weights_only=False."
            ) from exc
        raise


def _in_console(cmd: object, desc: object) -> None:
    try:
        subprocess.run(list(cmd), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{desc} failed with error: {exc}") from exc


@contextlib.contextmanager
def _onnx_model(model: object) -> None:
    was_training = bool(getattr(model, "training", False))
    candidate_attrs = (
        "optimizer",
        "optimizer_state",
        "optim",
        "training_history",
        "history",
        "logger",
        "metrics",
        "_training_history",
    )
    removed_top = {}
    removed_sub = []
    model.eval()
    for name in candidate_attrs:
        if hasattr(model, name):
            try:
                removed_top[name] = getattr(model, name)
                delattr(model, name)
            except Exception:
                pass
    with contextlib.suppress(Exception):
        for module in model.modules():
            for attr in ("logger", "history"):
                if hasattr(module, attr):
                    try:
                        v = getattr(module, attr)
                    except Exception:
                        continue
                    if isinstance(v, Recorder):
                        try:
                            removed_sub.append((module, attr, v))
                            delattr(module, attr)
                        except Exception:
                            pass
    try:
        with _temp_environ({"STNET_MSR_FORCE_TORCH": "1"}, only_if_unset=True):
            eager_ctx = getattr(model, "eager_for_export", None)
            if callable(eager_ctx):
                with eager_ctx():
                    yield model
            else:
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
        for v in out:
            if isinstance(v, torch.Tensor):
                return v
    raise RuntimeError("Model forward did not return a tensor output.")


def _get_forward_parameters(model_cls: object) -> object:
    with _FORWARD_PARAM_CACHE_LOCK:
        cached = _FORWARD_PARAM_CACHE.get(model_cls)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(model_cls.forward)
    except Exception:
        with _FORWARD_PARAM_CACHE_LOCK:
            _FORWARD_PARAM_CACHE[model_cls] = None
        return None
    names = set(sig.parameters.keys())
    accepts_kwargs = any(
        getattr(p, "kind", None) == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    info = (names, bool(accepts_kwargs))
    with _FORWARD_PARAM_CACHE_LOCK:
        _FORWARD_PARAM_CACHE[model_cls] = info
        if len(_FORWARD_PARAM_CACHE) > 512:
            _FORWARD_PARAM_CACHE.clear()
    return info

def _forward(model: object, x: object) -> object:
    fwd_export = getattr(model, "forward_export", None)
    if callable(fwd_export) and isinstance(x, torch.Tensor):
        return fwd_export(x)
    info = _get_forward_parameters(type(model))
    if info is None:
        return model(x)
    names, accepts_kwargs = info
    kwargs = {}
    if accepts_kwargs or "labels_flat" in names:
        kwargs["labels_flat"] = None
    if accepts_kwargs or "net_loss" in names:
        kwargs["net_loss"] = None
    return model(x, **kwargs) if kwargs else model(x)


def _get_tensor_shape(model: object, sample_input: object) -> object:
    in_dim = None
    out_shape = None
    if hasattr(model, "in_dim"):
        try:
            in_dim = int(getattr(model, "in_dim"))
        except (TypeError, ValueError):
            in_dim = None
    if hasattr(model, "out_shape"):
        try:
            _shape = getattr(model, "out_shape")
            out_shape = tuple((int(x) for x in _shape))
        except (TypeError, ValueError):
            out_shape = None
    if (in_dim is None or out_shape is None) and sample_input is not None:
        dev = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
        sample = sample_input.to(dev)
        with inference_mode(model.eval()):
            if sample.ndim == 1:
                sample = sample.unsqueeze(0)
            y_flat = _extract_tensor(_forward(model, sample))
        if in_dim is None:
            in_dim = int(sample.numel() // int(sample.shape[0]))
        if out_shape is None:
            out_shape = tuple((int(x) for x in y_flat.shape[1:]))
            if len(out_shape) == 1:
                out_shape = (out_shape[0],)
    if in_dim is None or out_shape is None:
        raise RuntimeError("Failed to infer input and output shapes.")
    return (int(in_dim), tuple(out_shape))


def _pad_sample(model: object, sample_input: object) -> object:
    if sample_input is not None:
        return sample_input
    in_dim, _ = _get_tensor_shape(model, sample_input)
    try:
        param = next(model.parameters())
        dtype, device = (param.dtype, param.device)
    except StopIteration:
        dtype, device = (torch.float32, torch.device("cpu"))
    return torch.zeros(1, in_dim, dtype=dtype, device=device)


def _onnx_options(kwargs: object, *, target: str = "onnx") -> object:
    """Centralized ONNX export defaults.

    목표:
    - ONNX/ORT: 최대한 동적(batch) 유지 + 최신 opset 우선
    - TF/LiteRT/NNEF: 변환 성공률 최우선(정적 batch, 보수적인 opset, legacy exporter 우선)
    """
    target_l = str(target or "onnx").strip().lower()
    if target_l in {"tensorrt", "trt", "tensor-rt", "tensor_rt"}:
        default_opset = 17
        default_dynamic_batch = True
        default_prefer_dynamo = False
        default_simplify = True
        default_fallback = [17, 16, 15, 14, 13]
    elif target_l in {"tensorflow", "tf", "litert", "tflite"}:
        default_opset = 15
        default_dynamic_batch = False
        default_prefer_dynamo = False
        default_simplify = True
        default_fallback = [15, 13]
    elif target_l in {"nnef"}:
        default_opset = 13
        default_dynamic_batch = False
        default_prefer_dynamo = False
        default_simplify = True
        default_fallback = [13]
    else:
        default_opset = 18
        default_dynamic_batch = True
        default_prefer_dynamo = True
        default_simplify = False
        default_fallback = [18, 17, 16, 15, 13]

    opset_version = int(kwargs.get("opset_version", default_opset))
    dynamic_batch = bool(kwargs.get("dynamic_batch", default_dynamic_batch))
    prefer_dynamo = bool(kwargs.get("prefer_dynamo", kwargs.get("dynamo", default_prefer_dynamo)))
    simplify = bool(kwargs.get("simplify_onnx", kwargs.get("onnx_simplify", default_simplify)))

    fb = kwargs.get("opset_fallback", None)
    opset_fallback: list[int]
    if fb is None:
        opset_fallback = list(default_fallback)
    elif isinstance(fb, str):
        opset_fallback = [int(x) for x in re.split(r"[\s,]+", fb) if x]
    elif isinstance(fb, (list, tuple)):
        opset_fallback = [int(x) for x in fb]
    else:
        opset_fallback = [int(fb)]

    # Ensure the requested opset is tried first.
    opset_try = [opset_version] + [v for v in opset_fallback if int(v) != int(opset_version)]
    return {
        "sample_input": kwargs.get("sample_input"),
        "opset_version": opset_version,
        "opset_fallback": opset_try,
        "dynamic_batch": dynamic_batch,
        "prefer_dynamo": prefer_dynamo,
        "simplify": simplify,
    }


def _coerce_onnx_path(dst: PathLike, kwargs: object) -> object:
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
        **kwargs: Any,
    ) -> object:
        """Export a model to ONNX with converter-friendly fallbacks.

        - TF/LiteRT/NNEF 쪽은 동적 axis/dynamo exporter가 실패 원인이 되는 경우가 많아서
          (정적 batch + legacy exporter 우선 + 낮은 opset) 조합을 기본값으로 둔다.
        - ONNX/ORT 쪽은 최신 opset + dynamo exporter 우선, 실패 시 legacy로 폴백한다.
        """
        del args, kwargs
        is_required("onnx", "pip install onnx")
        wrapper = _CompatLayer(model).eval()
        sample = _pad_sample(model, sample_input)
        if isinstance(sample, torch.Tensor) and sample.ndim == 1:
            sample = sample.unsqueeze(0)

        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        input_names = ["features"]
        output_names = ["preds_flat"]

        dynamic_axes = None
        dynamic_shapes = None
        if bool(dynamic_batch):
            dynamic_axes = {"features": {0: "batch"}, "preds_flat": {0: "batch"}}
            # Dynamo exporter uses dynamic_shapes + torch.export.Dim (if available).
            with contextlib.suppress(Exception):
                if hasattr(torch, "export") and hasattr(torch.export, "Dim"):
                    dynamic_shapes = (
                        {"features": {0: torch.export.Dim("batch")}},
                        {"preds_flat": {0: torch.export.Dim("batch")}},
                    )

        training_mode = None
        with contextlib.suppress(Exception):
            training_mode = torch.onnx.TrainingMode.EVAL

        def _base_kwargs(opset: int) -> dict[str, Any]:
            common_kwargs: dict[str, Any] = {
                "export_params": True,
                "opset_version": int(opset),
                "do_constant_folding": True,
                "keep_initializers_as_inputs": False,
                "input_names": input_names,
                "output_names": output_names,
            }
            if training_mode is not None:
                common_kwargs["training"] = training_mode
            if dynamic_axes is not None:
                common_kwargs["dynamic_axes"] = dynamic_axes
            base = inspect.signature(torch.onnx.export).parameters
            return {k: v for k, v in common_kwargs.items() if k in base}

        supports_dynamo = bool("dynamo" in inspect.signature(torch.onnx.export).parameters)
        if opset_fallback is None:
            opset_fallback = (opset_version,)
        # De-dup while keeping order.
        seen: set[int] = set()
        opset_try: list[int] = []
        for v in (int(opset_version), *[int(x) for x in opset_fallback]):
            if v not in seen and v > 0:
                seen.add(v)
                opset_try.append(v)
        if len(opset_try) == 0:
            opset_try = [int(opset_version)]

        # Prefer dynamo vs legacy depending on target, but always fallback.
        exporter_order: list[bool]
        if supports_dynamo:
            exporter_order = [True, False] if bool(prefer_dynamo) else [False, True]
        else:
            exporter_order = [False]

        errors: list[str] = []
        for opset in opset_try:
            base_kwargs = _base_kwargs(opset)
            for use_dynamo in exporter_order:
                try:
                    if supports_dynamo:
                        if use_dynamo:
                            dyn_kwargs = dict(base_kwargs)
                            if dynamic_shapes is not None:
                                dyn_kwargs["dynamic_shapes"] = dynamic_shapes
                            torch.onnx.export(
                                wrapper,
                                sample,
                                str(onnx_path),
                                dynamo=True,
                                **dyn_kwargs,
                            )
                        else:
                            torch.onnx.export(
                                wrapper,
                                sample,
                                str(onnx_path),
                                dynamo=False,
                                **base_kwargs,
                            )
                    else:
                        torch.onnx.export(wrapper, sample, str(onnx_path), **base_kwargs)

                    if bool(simplify):
                        with contextlib.suppress(Exception):
                            import onnx
                            model_onnx = onnx.load(str(onnx_path))
                            with contextlib.suppress(Exception):
                                import onnxsim  # type: ignore
                                model_simp, ok = onnxsim.simplify(model_onnx)
                                if bool(ok):
                                    onnx.save(model_simp, str(onnx_path))
                    return onnx_path
                except Exception as exc:
                    errors.append(
                        f"opset={opset} dynamo={use_dynamo} -> {type(exc).__name__}: {exc}"
                    )
                    continue

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
        p = Path(path)
        suffix = p.suffix.lower()
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
        suffix = p.suffix.lower()
        if not suffix and p.exists() and p.is_dir():
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint import save as dcp_save
            with _save_sync(p, barrier=True):
                opts_sd = StateDictOptions(full_state_dict=True)
                m_sd = get_model_state_dict(model, options=opts_sd)
                dcp_save(state_dict={"model": m_sd}, storage_writer=FileSystemWriter(str(p)))
                meta = {
                    "version": 1,
                    "format": "dcp-dir-v1",
                    "in_dim": int(getattr(model, "in_dim", 0)),
                    "out_shape": tuple((int(x) for x in getattr(model, "out_shape", ()))),
                    "config": _load_model_config(model),
                    "pytorch_version": torch.__version__,
                    "extra": coerce_json(extra or {}),
                }
                meta_path = p / "meta.json"
                if _is_rank0_global():
                    write_json(meta_path, coerce_json(meta), indent=2)
            return p
        if not suffix:
            p = p.with_suffix(".pt")
            suffix = ".pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        with _save_sync(p, barrier=False):
            if not _is_rank0_global():
                return p
            if suffix == ".safetensors":
                is_required("safetensors", "pip install safetensors")
                from safetensors.torch import save_file as save_tensors

                sd = model.state_dict()
                cpu_sd = {k: _to_tensor(v) for k, v in sd.items()}
                fd, tmp_name = tempfile.mkstemp(
                    prefix=p.name + ".", suffix=p.suffix + ".tmp", dir=str(p.parent)
                )
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    save_tensors(cpu_sd, str(tmp_path), metadata={"format": "safetensors-v1"})
                    tmp_path.replace(p)
                finally:
                    with contextlib.suppress(Exception):
                        if tmp_path.exists():
                            tmp_path.unlink()
                meta = {
                    "version": 1,
                    "in_dim": int(getattr(model, "in_dim", 0)),
                    "out_shape": tuple((int(x) for x in getattr(model, "out_shape", ()))),
                    "config": _load_model_config(model),
                    "pytorch_version": torch.__version__,
                    "extra": coerce_json(extra or {}),
                }
                meta_path = p.with_suffix(".json")
                write_json(meta_path, coerce_json(meta), indent=2)
                return p
            payload = {
                "version": 1,
                "in_dim": int(getattr(model, "in_dim", 0)),
                "out_shape": tuple((int(x) for x in getattr(model, "out_shape", ()))),
                "config": _load_model_config(model),
                "state_dict": model.state_dict(),
                "pytorch_version": torch.__version__,
            }
            if optimizer is not None and hasattr(optimizer, "state_dict"):
                with contextlib.suppress(Exception):
                    payload["optimizer_state_dict"] = optimizer.state_dict()
            if extra is not None:
                payload["extra"] = coerce_json(extra)
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
                serving, _coerce_onnx_path(dst, kwargs), **_onnx_options(kwargs, target="onnx")
            )
            ort_path, optimized = Exporter.OrtLayer.to_ort(
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
                serving_model, _coerce_onnx_path(dst, kwargs), **_onnx_options(kwargs, target="nnef")
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
            with inference_mode(wrapper):
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
                serving_model, _coerce_onnx_path(dst, kwargs), **_onnx_options(kwargs, target="litert")
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
                with inference_mode(wrapper):
                    scripted = torch.jit.trace(
                        wrapper,
                        sample,
                        check_trace=False,
                        strict=False,
                    )
            else:
                try:
                    with inference_mode(wrapper):
                        scripted = torch.jit.script(wrapper)
                except Exception as exc:
                    if sample is None:
                        sample = _pad_sample(serving_model, None)
                    warnings.warn(
                        f"TorchScript scripting failed ({type(exc).__name__}: {exc}); falling back to tracing.",
                        RuntimeWarning,
                    )
                    with inference_mode(wrapper):
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
            with inference_mode(wrapper):
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
                    # onnx2tf may nest the SavedModel; locate it if needed.
                    if (saved_model_dir / "saved_model.pb").exists():
                        return (saved_model_dir,)
                    found = list(saved_model_dir.rglob("saved_model.pb"))
                    if len(found) > 0:
                        return (found[0].parent,)

            # Fallback: onnx-tf
            is_required("onnx-tf", "pip install onnx-tf")
            from onnx_tf.backend import prepare

            import onnx

            model_onnx = onnx.load(str(onnx_path))
            prepare(model_onnx).export_graph(str(saved_model_dir))

        return (saved_model_dir,)


_register_torchversion_safe_global()
