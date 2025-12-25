# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Sequence

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
from .config import ModelConfig, coerce_model_config

_LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _export_backend() -> ModuleType:
    from ..backend import export as export_mod

    return export_mod


def _is_required(module: str, pip_hint: str | None = None) -> None:
    _export_backend().is_required(module, pip_hint)


def _strip_legacy_wrapped_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Strip legacy keys like `m.0.module.<real_key>` -> `<real_key>`.

    This is done lazily: we only allocate a new dict if we actually change a key.
    """
    new_sd: Optional[Dict[str, Any]] = None
    prefix = ("m",)  # explicit marker for clarity

    for k, v in sd.items():
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
            break

    if new_sd is None:
        return sd

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


def _torch_load_checkpoint(
    path: str | Path,
    *,
    map_location: Optional[torch.device | str] = None,
    weights_only: bool = True,
) -> Any:
    """Load a torch checkpoint with a safe default."""
    p = Path(path)

    try:
        return torch.load(
            str(p),
            map_location=map_location or "cpu",
            weights_only=bool(weights_only),
        )
    except TypeError:
        # Older PyTorch: no weights_only support.
        return torch.load(str(p), map_location=map_location or "cpu")
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

    from ..backend.export import TorchIO as _TorchIO  # local import (breaks backend<->api cycles)

    # Native torch checkpoint path (global-rank0 only for single-file outputs).
    if _TorchIO.is_native_target(p):
        if args:
            raise TypeError(
                "Positional args are only supported for export converters; "
                "use keyword arguments for TorchIO.save()."
            )

        merged_extra = dict(extra or {})
        if ema_averager is not None and hasattr(ema_averager, "state_dict"):
            with contextlib.suppress(Exception):
                merged_extra["ema_averager_state"] = ema_averager.state_dict()
        if swa_averager is not None and hasattr(swa_averager, "state_dict"):
            with contextlib.suppress(Exception):
                merged_extra["swa_averager_state"] = swa_averager.state_dict()

        out = _TorchIO.save(model, p, optimizer=optimizer, extra=merged_extra or None, **kwargs)
        return str(out)

    # Export converter path (positional args supported here only).
    conv = _export_backend().OnnxIO.for_export(p.suffix)
    if conv is None:
        raise ValueError(f"Unknown export format for path '{path}'.")
    conv.save(model, p, *args, **kwargs)
    return str(p)


# TorchIO + Format were moved to stnet.backend.export to break backend<->api cycles.
# We still provide a lazy re-export for backward compatibility.
def __getattr__(name: str) -> Any:
    if name in {"TorchIO", "Format"}:
        from ..backend.export import TorchIO, Format  # local import (heavy)

        return TorchIO if name == "TorchIO" else Format
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
