# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
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

from torch.distributed.checkpoint import FileSystemReader, load as dcp_load
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from ..functional.fx import Fusion
from ..model import Root
from .config import ModelConfig, coerce_model_config

class Format(Protocol):

    name: str

    def save(self, model: nn.Module, dst: Path, *args: Any, **kwargs: Any) -> Tuple[Path, ...]: ...


@lru_cache(maxsize=1)
def _export_backend():
    from ..backend import export as export_mod

    return export_mod


def _is_required(module: str, pip_hint: str | None = None) -> None:
    _export_backend().is_required(module, pip_hint)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    return value


def _load_model_config(model: nn.Module) -> Dict[str, Any]:
    cfg_obj = getattr(model, "_Root__config", None)
    if cfg_obj is None:
        cfg_obj = getattr(model, "__stnet_root_config__", None)
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
    return asdict(coerce_model_config(candidate))


def new_model(
    in_dim: int,
    out_shape: Sequence[int],
    config: ModelConfig | Dict[str, Any] | None,
    *,
    wrap: bool = True,
) -> nn.Module:
    cfg = coerce_model_config(config)
    core = Root(in_dim, tuple(int(x) for x in out_shape), config=cfg)
    if not wrap:
        return core
    return Fusion.use_tensordict_layers(
        core, in_key="features", out_key="pred", add_loss=True
    )


def load_model(
    checkpoint_path: str | Path,
    in_dim: Optional[int] = None,
    out_shape: Optional[Sequence[int]] = None,
    config: ModelConfig | Dict[str, Any] | None = None,
    map_location: Optional[torch.device | str] = None,
    *,
    wrap: bool = True,
) -> nn.Module:

    p = Path(checkpoint_path)
    if p.is_dir():
        if in_dim is None or out_shape is None:
            raise ValueError("Loading from a checkpoint directory requires in_dim and out_shape.")
        model = new_model(int(in_dim), tuple(out_shape), config, wrap=wrap)
        opts = StateDictOptions(full_state_dict=True)
        m_sd = get_model_state_dict(model, options=opts)
        dcp_load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(str(p)))
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        return model

    if p.suffix.lower() == ".safetensors":
        meta_path = p.with_suffix(".json")
        if not meta_path.exists():
            raise RuntimeError("Missing sidecar JSON file for the safetensors checkpoint.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        use_in_dim = int(in_dim if in_dim is not None else meta.get("in_dim"))
        out_shape_meta = out_shape if out_shape is not None else meta.get("out_shape") or ()
        use_out_shape = tuple(int(x) for x in out_shape_meta)
        use_config = coerce_model_config(config if config is not None else meta.get("config"))
        model = new_model(use_in_dim, use_out_shape, use_config, wrap=wrap)
        _is_required("safetensors", "pip install safetensors")
        from safetensors.torch import load_file as load_tensors

        sd = load_tensors(str(p), device=map_location or "cpu")
        model.load_state_dict(sd)
        return model

    obj = torch.load(str(p), map_location=map_location or "cpu", weights_only=False)
    meta_in_dim = obj.get("in_dim") if isinstance(obj, dict) else None
    meta_out_shape = obj.get("out_shape") if isinstance(obj, dict) else None
    meta_cfg = obj.get("config") if isinstance(obj, dict) else None
    use_in_dim = int(in_dim if in_dim is not None else meta_in_dim)
    out_shape_meta = out_shape if out_shape is not None else meta_out_shape or ()
    use_out_shape = tuple(int(x) for x in out_shape_meta)
    use_config = coerce_model_config(config if config is not None else meta_cfg)
    model = new_model(use_in_dim, use_out_shape, use_config, wrap=wrap)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(sd)
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

    if Model.is_native_target(p):
        merged_extra = dict(extra or {})
        if ema_averager is not None and hasattr(ema_averager, "state_dict"):
            with contextlib.suppress(Exception):
                merged_extra["ema_averager_state"] = ema_averager.state_dict()
        if swa_averager is not None and hasattr(swa_averager, "state_dict"):
            with contextlib.suppress(Exception):
                merged_extra["swa_averager_state"] = swa_averager.state_dict()
        out = Model.save(
            model,
            p,
            optimizer=optimizer,
            extra=merged_extra or None,
            **kwargs,
        )
        return str(out)

    conv = _export_backend().Model.for_export(p.suffix)
    if conv is None:
        raise ValueError(f"Unknown export format for path '{path}'.")
    conv.save(model, p, **kwargs)
    return str(p)


class Model:

    NATIVE_EXTS = {".pt", ".pth", ".safetensors"}

    @staticmethod
    def is_native_target(path: str | Path) -> bool:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix:
            return suffix in Model.NATIVE_EXTS
        return True

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

        if not suffix:
            if p.exists() and p.is_dir():
                from torch.distributed.checkpoint import FileSystemWriter, save as dcp_save

                opts_sd = StateDictOptions(full_state_dict=True)
                m_sd = get_model_state_dict(model, options=opts_sd)
                dcp_save(
                    state_dict={"model": m_sd},
                    storage_writer=FileSystemWriter(str(p)),
                )
                return p
            p = p.with_suffix(".pt")
            suffix = ".pt"

        p.parent.mkdir(parents=True, exist_ok=True)

        if suffix == ".safetensors":
            _is_required("safetensors", "pip install safetensors")
            from safetensors.torch import save_file as save_tensors

            sd = model.state_dict()
            cpu_sd = {k: _to_cpu(v) for k, v in sd.items()}
            save_tensors(cpu_sd, str(p), metadata={"format": "safetensors-v1"})
            meta = {
                "version": 1,
                "in_dim": int(getattr(model, "in_dim", 0)),
                "out_shape": tuple(int(x) for x in getattr(model, "out_shape", ())),
                "config": _load_model_config(model),
                "pytorch_version": torch.__version__,
                "extra": extra or {},
            }
            p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
            with contextlib.suppress(Exception):
                payload["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            payload["extra"] = extra
        torch.save(payload, str(p), **opts)
        return p


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
