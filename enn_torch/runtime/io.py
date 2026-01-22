# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import inspect
import os
import re
import sys
import tempfile
import warnings
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence, Self

import torch
from torch import nn

from ..core.concurrency import Mutex
from ..core.datatypes import PathLike, coerce_json, save_temp, write_json
from ..core.distributed import distributed_barrier, is_rank0

if TYPE_CHECKING:
    from .wrappers import Format

_IGNORED_WARNINGS = (
    "torch.distributed is disabled",
    "TypedStorage is deprecated",
)
_IGNORED_RE = (
    r".*(?:" + "|".join(re.escape(s) for s in _IGNORED_WARNINGS) + r").*"
)
_SAVE_LOCK_GUARD = Mutex()
_SAVE_PATH_LOCKS = weakref.WeakValueDictionary()
_WARNINGS_FILTER_LOCK = Mutex()


def _register_safe_globals() -> None:
    with contextlib.suppress(Exception):
        if add_safe_globals:
            from torch.torch_version import TorchVersion

            add_safe_globals([TorchVersion])


@contextlib.contextmanager
def _filtered_warnings(
    sentences: Sequence[str] | None = None,
) -> Iterator[None]:
    msg_re = (
        _IGNORED_RE
        if sentences is None
        else (
            r".*(?:" + "|".join(re.escape(str(s)) for s in sentences) + r").*"
            if sentences
            else ""
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
def _save_sync(
    path: PathLike | None = None, *args: Any, barrier: bool = False
) -> Iterator[None]:
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
    path: PathLike,
    *args: Any,
    map_location: object = None,
    weights_only: bool = True,
) -> object:
    try:
        return torch.load(
            str(path),
            map_location=map_location or "cpu",
            weights_only=weights_only,
        )
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
        raise ImportError(
            f"{module} is required for this operation{hint}"
        ) from err


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
                "out_shape": tuple(
                    int(x) for x in getattr(model, "out_shape", ())
                ),
                "config": _load_model_config(model),
                "pytorch_version": torch.__version__,
                "extra": coerce_json(extra or {}),
            }

        if not p.suffix and p.exists() and p.is_dir():
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint import save as dcp_save
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            with _save_sync(p, barrier=True):
                dcp_save(
                    state_dict={
                        "model": get_model_state_dict(
                            model,
                            options=StateDictOptions(full_state_dict=True),
                        )
                    },
                    storage_writer=FileSystemWriter(str(p)),
                )
                if is_rank0():
                    write_json(
                        p / "meta.json",
                        {**_make_meta(), "format": "dcp-dir-v1"},
                        indent=2,
                    )
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

                from ..core.tensor import coerce_tensor

                fd, tmp_name = tempfile.mkstemp(
                    prefix=p.name + ".",
                    suffix=p.suffix + ".tmp",
                    dir=str(p.parent),
                )
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    save_tensors(
                        {
                            k: coerce_tensor(v)
                            for k, v in model.state_dict().items()
                        },
                        str(tmp_path),
                        metadata={"format": "safetensors-v1"},
                    )
                    tmp_path.replace(p)
                finally:
                    with contextlib.suppress(Exception):
                        tmp_path.unlink() if tmp_path.exists() else None
                meta = _make_meta()
                write_json(p.with_name(p.name + ".json"), meta, indent=2)
                with contextlib.suppress(Exception):
                    legacy = p.with_suffix(".json")
                    if legacy != p.with_name(p.name + ".json"):
                        write_json(legacy, meta, indent=2)
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
    _ONNXExporter: Any = None
    _ORTBuilder: Any = None
    _export_sig_cache: object | None = None
    _export_sig_lock = Mutex()

    @classmethod
    def _export_sig(cls: type[Self]) -> object:
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
    def _export_sig_keys(cls: type[Self]) -> set[str]:
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
            return set(sig.parameters.keys())
        except Exception:
            return set()

    @classmethod
    def register(
        cls: type[Self], name: str, exts: tuple[str, ...], impl: Format
    ) -> None:
        with cls._defaults_lock:
            cls._register_unlocked(name, exts, impl)

    @classmethod
    def _register_unlocked(
        cls: type[Self], name: str, exts: tuple[str, ...], impl: Format
    ) -> None:
        cls._by_name[name] = impl
        for ext in exts:
            cls._ext_map[ext.lower()] = name

    @classmethod
    def _ensure_defaults_registered(cls: type[Self]) -> None:
        if cls._defaults_registered:
            return
        with cls._defaults_lock:
            if cls._defaults_registered:
                return
            try:
                from . import wrappers as _w
            except Exception as exc:
                raise RuntimeError(
                    "Exporter backends live in enn_torch.runtime.wrappers, but it could not be imported. "
                    "Install the optional export dependencies (e.g. tensordict) or avoid calling Exporter.for_export()."
                ) from exc

            cls._ONNXExporter = getattr(_w, "_ONNXExporter", None)
            cls._ORTBuilder = getattr(_w, "_ORTBuilder", None)

            cls._register_unlocked("onnx", (".onnx",), _w.ONNX())
            cls._register_unlocked("ort", (".ort",), _w.ORT())
            cls._register_unlocked(
                "tensorrt", (".engine", ".plan"), _w.TensorRT()
            )
            cls._register_unlocked(
                "coreml", (".mlmodel", ".mlpackage"), _w.CoreML()
            )
            cls._register_unlocked("litert", (".tflite",), _w.LiteRT())
            cls._register_unlocked(
                "pt2", (".pt2", ".export"), _w.TorchExport()
            )
            cls._register_unlocked("aoti", (".aoti",), _w.TorchInductor())
            cls._register_unlocked("executorch", (".pte",), _w.ExecuTorch())
            cls._register_unlocked(
                "tensorflow", (".savedmodel", ".pb", ".tf"), _w.TensorFlow()
            )
            cls._defaults_registered = True

    @classmethod
    def for_export(cls: type[Self], ext: str) -> Format | None:
        cls._ensure_defaults_registered()
        with cls._defaults_lock:
            name = cls._ext_map.get(ext.lower())
            return cls._by_name.get(name) if name else None


try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None
_register_safe_globals()
