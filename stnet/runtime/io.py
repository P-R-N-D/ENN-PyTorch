# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import os
import re
import sys
import tempfile
import threading
import warnings
import weakref
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from ..core.tensor import coerce_tensor, is_meta_or_fake_tensor
from ..core.datatypes import Mutex, PathLike, coerce_json, save_temp, write_json
from .wrappers import (
    CoreML,
    ExecuTorch,
    Format,
    LiteRT,
    ONNX,
    ORT,
    PathLike,
    TensorFlow,
    TensorRT,
    TorchExport,
    TorchInductor,
    _ONNXModule,
    _ORTModule,
)

try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None


_IGNORED_WARNINGS = ("torch.distributed is disabled", "TypedStorage is deprecated")
_IGNORED_RE = r".*(?:" + "|".join(re.escape(s) for s in _IGNORED_WARNINGS) + r").*"

_WARNINGS_FILTER_LOCK = Mutex()

_SAVE_LOCK_GUARD = Mutex()
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


def _save_lock(path: PathLike | None = None) -> Mutex:
    try:
        key = str(Path(path).expanduser().resolve()) if path else "__global__"
    except Exception:
        key = str(path)
    with _SAVE_LOCK_GUARD:
        return _SAVE_PATH_LOCKS.setdefault(key, Mutex(reentrant=True))


@contextlib.contextmanager
def _save_sync(path: PathLike | None = None, *args: Any, barrier: bool = False) -> None:
    with _save_lock(path):
        if barrier:
            distributed_barrier()
        try:
            yield
        finally:
            if barrier:
                distributed_barrier()


def _load_model_config(model: object) -> object:
    try:
        from ..config import _extract_model_config_dict

        return _extract_model_config_dict(model)
    except Exception:
        return {}


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


def is_required(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise ImportError(f"{module} is required for this operation{hint}") from err


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
                if is_rank0():
                    write_json(p / "meta.json", {**_make_meta(), "format": "dcp-dir-v1"}, indent=2)
            return p
        if not p.suffix:
            p = p.with_suffix(".pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        with _save_sync(p, barrier=False):
            if not is_rank0():
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
                        {k: coerce_tensor(v) for k, v in model.state_dict().items()},
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
    _defaults_lock = Mutex()
    _ONNXModule = _ONNXModule
    _ORTModule = _ORTModule
    _export_sig_cache: object | None = None
    _export_sig_lock = Mutex()

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
            cls.register("onnx", (".onnx",), ONNX())
            cls.register("ort", (".ort",), ORT())
            cls.register("tensorrt", (".engine",), TensorRT())
            cls.register("coreml", (".mlmodel",), CoreML())
            cls.register("litert", (".tflite",), LiteRT())
            cls.register("pt2", (".pt2", ".export"), TorchExport())
            cls.register("aoti", (".aoti",), TorchInductor())
            cls.register("executorch", (".pte",), ExecuTorch())
            cls.register("tensorflow", (".savedmodel", ".pb", ".tf"), TensorFlow())
            cls._defaults_registered = True

    @classmethod
    def for_export(cls, ext: str) -> Format | None:
        cls._ensure_defaults_registered()
        name = cls._ext_map.get(ext.lower())
        return cls._by_name.get(name) if name else None


_register_safe_globals()
