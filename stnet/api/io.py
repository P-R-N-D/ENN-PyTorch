# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import torch
from torch import nn

try:
    from torch.serialization import add_safe_globals as _add_safe_globals
except ImportError:  # pragma: no cover
    _add_safe_globals = None

# Best-effort: allowlist TorchVersion for weights_only loads when available.
# Keep this non-fatal; older torch won't have these modules or APIs.
if _add_safe_globals is not None:
    with contextlib.suppress(Exception):
        import torch.torch_version as _torch_version

        _add_safe_globals([_torch_version.TorchVersion])

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from ..model.nn import Root, resize_scaler_buffer
from .config import ModelConfig, coerce_model_config, model_config_to_dict

_LOGGER = logging.getLogger(__name__)

# Serialize/save operations should not overlap within a process.
# This does not stop training by itself, but prevents concurrent saves and is
# a good hook point if the training loop cooperates.
_SAVE_LOCK = threading.RLock()


class Format(Protocol):
    name: str

    def save(
        self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any
    ) -> Tuple[Path, ...]: ...


@lru_cache(maxsize=1)
def _export_backend() -> ModuleType:
    from ..backend import export as export_mod

    return export_mod


def _is_required(module: str, pip_hint: str | None = None) -> None:
    _export_backend().is_required(module, pip_hint)


def _to_cpu(value: Any) -> Any:
    """Detach tensors and move them to CPU recursively.

    This is used for formats that require CPU tensors (e.g. safetensors).
    It is defensive about tuple subclasses (e.g. namedtuple).
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return type(value)((k, _to_cpu(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        seq = tuple(_to_cpu(v) for v in value)
        if type(value) is tuple:
            return seq
        # namedtuple: constructor expects positional args.
        if hasattr(value, "_fields"):
            try:
                return type(value)(*seq)
            except Exception:
                return seq
        try:
            return type(value)(seq)
        except Exception:
            return seq
    return value


def _strip_legacy_wrapped_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Strip legacy keys like `m.0.module.<real_key>` -> `<real_key>`.

    This is done lazily: we only allocate a new dict if we actually change a key.
    """
    new_sd: Optional[Dict[str, Any]] = None
    prefix = ("m",)  # explicit marker for clarity

    for k, v in sd.items():
        # Fast checks before split:
        if not (k.startswith("m.") and ".module." in k):
            nk = k
        else:
            parts = k.split(".")
            if len(parts) >= 4 and parts[0] == prefix[0] and parts[1].isdigit() and parts[2] == "module":
                nk = ".".join(parts[3:])
            else:
                nk = k

        if nk != k and new_sd is None:
            new_sd = type(sd)()
            # Accumulate already processed entries: we must not copy `sd` blindly to keep O(n)
            # work bounded and to preserve dict subclass types if possible.
            # We'll rebuild from scratch in the second pass below.
            break

    if new_sd is None:
        return sd

    # Rebuild with the same rule (single pass).
    for k, v in sd.items():
        if not (k.startswith("m.") and ".module." in k):
            nk = k
        else:
            parts = k.split(".")
            if len(parts) >= 4 and parts[0] == prefix[0] and parts[1].isdigit() and parts[2] == "module":
                nk = ".".join(parts[3:])
            else:
                nk = k
        new_sd[nk] = v

    return new_sd


def _json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (torch.device, Path, torch.dtype)):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def _json_default(obj: Any) -> Any:
    return _json_sanitize(obj)


def _get_dist():
    try:
        import torch.distributed as dist  # type: ignore
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


def _local_rank_from_env() -> Optional[int]:
    # Common launchers:
    for key in ("LOCAL_RANK", "SLURM_LOCALID", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        v = os.environ.get(key)
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    return None


def _is_rank0_local() -> bool:
    # Prefer local rank env vars when present; fallback to global rank.
    lr = _local_rank_from_env()
    if lr is not None:
        return lr == 0
    return _is_rank0_global()


def _dist_barrier() -> None:
    dist = _get_dist()
    if dist is None:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        # Never fail saving because barrier isn't available/healthy.
        pass


@contextlib.contextmanager
def _save_sync() -> Any:
    """Synchronize saves within a process (lock) and across ranks (barrier)."""
    with _SAVE_LOCK:
        _dist_barrier()
        try:
            yield
        finally:
            _dist_barrier()


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=path.suffix + ".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(text, encoding=encoding)
        tmp_path.replace(path)
    finally:
        with contextlib.suppress(Exception):
            if tmp_path.exists():
                tmp_path.unlink()


def _atomic_torch_save(obj: Any, path: Path, **opts: Any) -> None:
    """Atomically write a torch checkpoint file.

    Writes to a temporary file in the same directory and then renames into place.
    This avoids corrupting checkpoints if the process is interrupted mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=path.suffix + ".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, str(tmp_path), **opts)
        tmp_path.replace(path)
    finally:
        with contextlib.suppress(Exception):
            if tmp_path.exists():
                tmp_path.unlink()


def _torch_load_checkpoint(
    path: Path,
    *,
    map_location: Optional[torch.device | str] = None,
    weights_only: bool = True,
) -> Any:
    """Load a torch checkpoint with a safe default.

    - Prefer `weights_only=True` when supported by the installed torch.
    - Fall back for older torch versions that don't support this argument.
    """
    try:
        return torch.load(
            str(path),
            map_location=map_location or "cpu",
            weights_only=bool(weights_only),
        )
    except TypeError:
        # Older PyTorch: no weights_only support.
        return torch.load(str(path), map_location=map_location or "cpu")
    except Exception as exc:
        if weights_only:
            raise RuntimeError(
                "Failed to load checkpoint with weights_only=True. "
                "If you trust the checkpoint source, retry with weights_only=False."
            ) from exc
        raise


def _read_checkpoint_dir_meta(p: Path) -> Dict[str, Any]:
    meta_path = p / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse checkpoint metadata at {str(meta_path)!r}") from exc


def _load_model_config(model: nn.Module) -> Dict[str, Any]:
    cfg_obj = getattr(model, "_Root__config", None)
    if cfg_obj is None:
        cfg_obj = getattr(model, "__stnet_instance_config__", None)
    if cfg_obj is None:
        for submodule in model.modules():
            cfg_obj = getattr(submodule, "_Root__config", None)
            if cfg_obj is not None:
                break
    candidate: ModelConfig | Dict[str, Any] | None
    if isinstance(cfg_obj, (ModelConfig, dict)):
        candidate = cfg_obj
    else:
        candidate = None
    try:
        return model_config_to_dict(candidate)
    except Exception:
        return {}


def new_model(
    in_dim: int,
    out_shape: Sequence[int],
    config: ModelConfig | Dict[str, Any] | None,
) -> nn.Module:
    cfg = coerce_model_config(config)
    core = Root(in_dim, tuple(int(x) for x in out_shape), config=cfg)
    return core


def load_model(
    checkpoint_path: str | Path,
    in_dim: Optional[int] = None,
    out_shape: Optional[Sequence[int]] = None,
    config: ModelConfig | Dict[str, Any] | None = None,
    map_location: Optional[torch.device | str] = None,
    weights_only: bool = True,
) -> nn.Module:
    p = Path(checkpoint_path)
    load_dev = torch.device(map_location) if map_location is not None else torch.device("cpu")

    # 1) Distributed Checkpoint directory
    if p.is_dir():
        meta: Dict[str, Any] = _read_checkpoint_dir_meta(p)

        use_in_dim = int(in_dim if in_dim is not None else (meta.get("in_dim") or 0))
        out_shape_meta = out_shape if out_shape is not None else (meta.get("out_shape") or ())
        use_out_shape = tuple(int(x) for x in out_shape_meta) if out_shape_meta else ()

        user_provided_config = config is not None
        raw_cfg = config if user_provided_config else meta.get("config")
        use_config = coerce_model_config(raw_cfg)
        if not user_provided_config:
            use_config.device = load_dev
        elif map_location is not None and use_config.device is None:
            use_config.device = load_dev

        if use_in_dim <= 0 or not use_out_shape:
            raise ValueError(
                "Loading from a checkpoint directory requires in_dim and out_shape, "
                "or a valid meta.json inside the directory."
            )

        model = new_model(use_in_dim, use_out_shape, use_config)
        opts = StateDictOptions(full_state_dict=True)
        m_sd = get_model_state_dict(model, options=opts)
        dcp_load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(str(p)))
        resize_scaler_buffer(model, m_sd)
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        return model

    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {str(p)!r}")

    suffix = p.suffix.lower()

    # 2) safetensors + sidecar json
    if suffix == ".safetensors":
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            raise RuntimeError("Missing sidecar JSON file for the safetensors checkpoint.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        use_in_dim = int(in_dim if in_dim is not None else meta.get("in_dim"))
        out_shape_meta = out_shape if out_shape is not None else (meta.get("out_shape") or ())
        use_out_shape = tuple(int(x) for x in out_shape_meta) if out_shape_meta else ()

        user_provided_config = config is not None
        use_config = coerce_model_config(config if user_provided_config else meta.get("config"))
        if not user_provided_config:
            use_config.device = load_dev
        elif map_location is not None and use_config.device is None:
            use_config.device = load_dev

        if use_in_dim <= 0 or not use_out_shape:
            raise RuntimeError(
                f"Invalid in_dim/out_shape metadata in {str(meta_path)!r}: "
                f"in_dim={use_in_dim}, out_shape={use_out_shape}"
            )

        model = new_model(use_in_dim, use_out_shape, use_config)

        _is_required("safetensors", "pip install safetensors")
        from safetensors.torch import load_file as load_tensors

        dev: str
        if map_location is None:
            dev = "cpu"
        elif isinstance(map_location, torch.device):
            dev = str(map_location)
        else:
            dev = str(map_location)

        sd = load_tensors(str(p), device=dev)
        sd = _strip_legacy_wrapped_keys(sd)
        resize_scaler_buffer(model, sd)
        model.load_state_dict(sd, strict=False)
        return model

    # 3) torch.save checkpoints (.pt/.pth or any other file)
    obj = _torch_load_checkpoint(p, map_location=map_location or "cpu", weights_only=weights_only)
    if isinstance(obj, dict):
        meta_in_dim = obj.get("in_dim")
        meta_out_shape = obj.get("out_shape")
        meta_cfg = obj.get("config")
        sd = obj["state_dict"] if "state_dict" in obj else obj
    else:
        meta_in_dim = None
        meta_out_shape = None
        meta_cfg = None
        sd = obj

    use_in_dim = int(in_dim if in_dim is not None else meta_in_dim)
    out_shape_meta = out_shape if out_shape is not None else (meta_out_shape or ())
    use_out_shape = tuple(int(x) for x in out_shape_meta)
    user_provided_config = config is not None

    use_config = coerce_model_config(config if user_provided_config else meta_cfg)
    if not user_provided_config:
        use_config.device = load_dev
    elif map_location is not None and use_config.device is None:
        use_config.device = load_dev

    if use_in_dim <= 0 or not use_out_shape:
        raise RuntimeError(
            f"Invalid or missing in_dim/out_shape when loading checkpoint {str(p)!r}: "
            f"in_dim={use_in_dim}, out_shape={use_out_shape}"
        )

    model = new_model(use_in_dim, use_out_shape, use_config)
    sd = _strip_legacy_wrapped_keys(sd) if isinstance(sd, dict) else sd
    # Keep buffer sizing consistent across formats.
    with contextlib.suppress(Exception):
        resize_scaler_buffer(model, sd)  # type: ignore[arg-type]
    model.load_state_dict(sd, strict=False)  # type: ignore[arg-type]
    return model


def save_model(
    model: nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
    *args: Any,
    ema_averager: Optional[Any] = None,
    swa_averager: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    p = Path(path)

    # Native torch checkpoint path (rank0-local only for single-file outputs).
    if TorchIO.is_native_target(p):
        if args:
            # Native path does not accept positional args (kept only for export converters).
            raise TypeError(
                "Positional args are only supported for export converters; "
                "use keyword arguments for TorchIO.save()."
            )

        merged_extra = dict(extra or {})
        if ema_averager is not None and hasattr(ema_averager, "state_dict"):
            try:
                merged_extra["ema_averager_state"] = ema_averager.state_dict()
            except Exception:
                _LOGGER.debug("Failed to serialize ema_averager_state; skipping", exc_info=True)
        if swa_averager is not None and hasattr(swa_averager, "state_dict"):
            try:
                merged_extra["swa_averager_state"] = swa_averager.state_dict()
            except Exception:
                _LOGGER.debug("Failed to serialize swa_averager_state; skipping", exc_info=True)

        out = TorchIO.save(model, p, optimizer=optimizer, extra=merged_extra or None, **kwargs)
        return str(out)

    # Export converter path (positional args supported here only).
    conv = _export_backend().OnnxIO.for_export(p.suffix)
    if conv is None:
        raise ValueError(f"Unknown export format for path '{path}'.")
    conv.save(model, p, *args, **kwargs)
    return str(p)


class TorchIO:
    NATIVE_EXTS = {".pt", ".pth", ".safetensors"}

    @staticmethod
    def is_native_target(path: str | Path) -> bool:
        p = Path(path)
        suffix = p.suffix.lower()
        return (not suffix) or (suffix in TorchIO.NATIVE_EXTS)

    @staticmethod
    def save(
        model: nn.Module,
        path: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None,
        **opts: Any,
    ) -> Path:
        p = Path(path)
        suffix = p.suffix.lower()

        # Distributed checkpoint directory case:
        # - All ranks participate in dcp_save
        # - Only local-rank0 writes meta.json
        if not suffix and p.exists() and p.is_dir():
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint import save as dcp_save

            with _save_sync():
                opts_sd = StateDictOptions(full_state_dict=True)
                m_sd = get_model_state_dict(model, options=opts_sd)
                dcp_save(state_dict={"model": m_sd}, storage_writer=FileSystemWriter(str(p)))

                meta = {
                    "version": 1,
                    "format": "dcp-dir-v1",
                    "in_dim": int(getattr(model, "in_dim", 0)),
                    "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
                    "config": _load_model_config(model),
                    "pytorch_version": torch.__version__,
                    "extra": _json_sanitize(extra or {}),
                }
                meta_path = p / "meta.json"
                if _is_rank0_local():
                    meta_text = json.dumps(meta, indent=2, default=_json_default)
                    _atomic_write_text(meta_path, meta_text, encoding="utf-8")
            return p

        # Single-file targets are local-rank0 only.
        if not _is_rank0_local():
            # Return the resolved destination path without writing.
            if not suffix:
                return p.with_suffix(".pt")
            return p

        if not suffix:
            p = p.with_suffix(".pt")
            suffix = ".pt"

        p.parent.mkdir(parents=True, exist_ok=True)

        with _save_sync():
            # safetensors: CPU tensors only; include sidecar json.
            if suffix == ".safetensors":
                _is_required("safetensors", "pip install safetensors")
                from safetensors.torch import save_file as save_tensors

                sd = model.state_dict()
                cpu_sd = {k: _to_cpu(v) for k, v in sd.items()}

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
                    "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
                    "config": _load_model_config(model),
                    "pytorch_version": torch.__version__,
                    "extra": _json_sanitize(extra or {}),
                }
                meta_path = p.with_suffix(".json")
                meta_text = json.dumps(meta, indent=2, default=_json_default)
                _atomic_write_text(meta_path, meta_text, encoding="utf-8")
                return p

            # torch.save payload: always sanitize extra for weights_only-friendly loads.
            payload: Dict[str, Any] = {
                "version": 1,
                "in_dim": int(getattr(model, "in_dim", 0)),
                "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
                "config": _load_model_config(model),
                "state_dict": model.state_dict(),
                "pytorch_version": torch.__version__,
            }
            if optimizer is not None and hasattr(optimizer, "state_dict"):
                try:
                    payload["optimizer_state_dict"] = optimizer.state_dict()
                except Exception:
                    _LOGGER.debug("Failed to serialize optimizer_state_dict; skipping", exc_info=True)
            if extra is not None:
                payload["extra"] = _json_sanitize(extra)

            _atomic_torch_save(payload, p, **opts)
            return p


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
