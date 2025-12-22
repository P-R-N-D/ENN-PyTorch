# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import logging
import os
import json
import tempfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

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

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     set_model_state_dict)

from ..model.nn import Root, resize_scaler_buffer
from .config import ModelConfig, coerce_model_config


_LOGGER = logging.getLogger(__name__)


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
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        seq = [_to_cpu(v) for v in value]
        return type(value)(seq) if isinstance(value, tuple) else seq
    return value


def _strip_legacy_wrapped_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    def _is_wrapped_key(key: str) -> bool:
        parts = key.split(".")
        return len(parts) >= 3 and parts[0] == "m" and parts[1].isdigit() and parts[2] == "module"

    if not any(_is_wrapped_key(k) for k in sd.keys()):
        return sd

    new_sd = sd.__class__() if hasattr(sd, "__class__") else {}
    for k, v in sd.items():
        if _is_wrapped_key(k):
            parts = k.split(".")
            try:
                module_idx = parts.index("module")
                new_key = ".".join(parts[module_idx + 1 :])
            except ValueError:
                new_key = k
        else:
            new_key = k
        new_sd[new_key] = v

    return new_sd


def _json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
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


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == 0
    except Exception:
        pass
    return True


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
    """Atomically write a torch checkpoint.

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
            str(path), map_location=map_location or "cpu", weights_only=bool(weights_only)
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
        raise RuntimeError(
            f"Failed to parse checkpoint metadata at {str(meta_path)!r}"
        ) from exc


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
        data = asdict(coerce_model_config(candidate))
        dev = data.get("device", None)
        if isinstance(dev, torch.device):
            data["device"] = str(dev)
        return data
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

    if p.suffix.lower() == ".safetensors":
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            raise RuntimeError(
                "Missing sidecar JSON file for the safetensors checkpoint."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        use_in_dim = int(in_dim if in_dim is not None else meta.get("in_dim"))
        out_shape_meta = (
            out_shape if out_shape is not None else meta.get("out_shape") or ()
        )
        use_out_shape = tuple(int(x) for x in out_shape_meta)
        user_provided_config = config is not None

        use_config = coerce_model_config(
            config if user_provided_config else meta.get("config")
        )
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

        dev = map_location or "cpu"
        # Normalize `map_location` to a string for `safetensors.torch.load_file`.
        # Passing a torch.device is usually accepted, but normalizing avoids
        # edge-cases across versions.
        if isinstance(dev, torch.device):
            dev = str(dev)
        else:
            dev = str(dev)

        sd = load_tensors(str(p), device=dev)
        sd = _strip_legacy_wrapped_keys(sd)
        resize_scaler_buffer(model, sd)
        model.load_state_dict(sd, strict=False)
        return model

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
    out_shape_meta = out_shape if out_shape is not None else meta_out_shape or ()
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
    sd = _strip_legacy_wrapped_keys(sd)
    model.load_state_dict(sd, strict=False)
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

    if TorchIO.is_native_target(p):
        merged_extra = dict(extra or {})
        if ema_averager is not None and hasattr(ema_averager, "state_dict"):
            try:
                merged_extra["ema_averager_state"] = ema_averager.state_dict()
            except Exception:
                _LOGGER.debug(
                    "Failed to serialize ema_averager_state; skipping",
                    exc_info=True,
                )
        if swa_averager is not None and hasattr(swa_averager, "state_dict"):
            try:
                merged_extra["swa_averager_state"] = swa_averager.state_dict()
            except Exception:
                _LOGGER.debug(
                    "Failed to serialize swa_averager_state; skipping",
                    exc_info=True,
                )
        out = TorchIO.save(
            model,
            p,
            optimizer=optimizer,
            extra=merged_extra or None,
            **kwargs,
        )
        return str(out)

    conv = _export_backend().OnnxIO.for_export(p.suffix)
    if conv is None:
        raise ValueError(f"Unknown export format for path '{path}'.")
    conv.save(model, p, **kwargs)
    return str(p)


class TorchIO:
    NATIVE_EXTS = {".pt", ".pth", ".safetensors"}

    @staticmethod
    def is_native_target(path: str | Path) -> bool:
        p = Path(path)
        suffix = p.suffix.lower()
        if not suffix:
            return True
        return suffix in TorchIO.NATIVE_EXTS

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

        if not suffix and p.exists() and p.is_dir():
            from torch.distributed.checkpoint import FileSystemWriter
            from torch.distributed.checkpoint import save as dcp_save

            opts_sd = StateDictOptions(full_state_dict=True)
            m_sd = get_model_state_dict(model, options=opts_sd)
            dcp_save(
                state_dict={"model": m_sd},
                storage_writer=FileSystemWriter(str(p)),
            )
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
            if _is_rank0():
                meta_text = json.dumps(meta, indent=2, default=_json_default)
                _atomic_write_text(meta_path, meta_text, encoding="utf-8")
            return p

        if not suffix:
            p = p.with_suffix(".pt")
            suffix = ".pt"

        p.parent.mkdir(parents=True, exist_ok=True)

        if suffix == ".safetensors":
            _is_required("safetensors", "pip install safetensors")
            from safetensors.torch import save_file as save_tensors

            sd = model.state_dict()
            cpu_sd = {k: _to_cpu(v) for k, v in sd.items()}
            # Write safetensors atomically to avoid partial files.
            fd, tmp_name = tempfile.mkstemp(
                prefix=p.name + ".", suffix=p.suffix + ".tmp", dir=str(p.parent)
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                save_tensors(
                    cpu_sd, str(tmp_path), metadata={"format": "safetensors-v1"}
                )
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
                _LOGGER.debug(
                    "Failed to serialize optimizer_state_dict; skipping",
                    exc_info=True,
                )
        if extra:
            payload["extra"] = extra
        _atomic_torch_save(payload, p, **opts)
        return p


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
