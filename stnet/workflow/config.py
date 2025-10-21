# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence, Tuple

OpsMode = Literal["train", "predict", "infer"]

__all__ = ["OpsMode", "OpsConfig", "coerce_ops_config", "ops_config"]


def _as_tuple_ints(xs: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in xs)


@dataclass(frozen=True)
class OpsConfig:
    mode: OpsMode
    in_dim: int
    out_shape: Tuple[int, ...]
    cfg_dict: Dict[str, Any]
    memmap_dir: Optional[str] = None
    ckpt_dir: Optional[str] = None
    init_ckpt_dir: Optional[str] = None
    epochs: int = 5
    batch_size: Optional[int] = None
    val_frac: float = 0.1
    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.0
    eta_min: float = 0.0
    seed: int = 42
    prefetch_factor: Optional[int] = 1
    grad_accum_steps: int = 1
    overlap_h2d: bool = True
    loss_tile_dim: Optional[int] = None
    loss_tile_size: Optional[int] = None
    loss_mask_mode: str = "none"
    loss_mask_value: Optional[float] = None
    model_ckpt_dir: Optional[str] = None
    keys: Optional[List[Tuple[Any, ...]]] = None
    TRAIN_POS_ORDER: ClassVar[Tuple[str, ...]] = (
        "epochs",
        "batch_size",
        "val_frac",
        "base_lr",
        "weight_decay",
        "warmup_ratio",
        "eta_min",
        "seed",
        "prefetch_factor",
        "grad_accum_steps",
        "overlap_h2d",
        "loss_tile_dim",
        "loss_tile_size",
        "loss_mask_mode",
        "loss_mask_value",
    )
    PRED_POS_ORDER: ClassVar[Tuple[str, ...]] = (
        "batch_size",
        "seed",
        "prefetch_factor",
    )

    @staticmethod
    def from_partial(mode: OpsMode, **kw: Any) -> "OpsConfig":
        kw = dict(kw)
        for k in ("in_dim", "out_shape", "cfg_dict"):
            if k not in kw or kw[k] is None:
                raise ValueError(f"OpsConfig missing required key: {k}")
        in_dim = int(kw["in_dim"])
        out_shape = _as_tuple_ints(kw["out_shape"])
        cfg_dict = dict(kw["cfg_dict"])
        common_keys = {
            "in_dim",
            "out_shape",
            "cfg_dict",
        }
        if mode == "train":
            for k in ("memmap_dir", "ckpt_dir"):
                if k not in kw or kw[k] is None:
                    raise ValueError(f"OpsConfig(train) missing required key: {k}")
            allowed = common_keys | {
                "memmap_dir",
                "ckpt_dir",
                "init_ckpt_dir",
                "epochs",
                "batch_size",
                "val_frac",
                "base_lr",
                "weight_decay",
                "warmup_ratio",
                "eta_min",
                "seed",
                "prefetch_factor",
                "grad_accum_steps",
                "overlap_h2d",
                "loss_tile_dim",
                "loss_tile_size",
                "loss_mask_mode",
                "loss_mask_value",
            }
            unsupported = set(kw) - allowed
            if unsupported:
                raise ValueError(
                    "OpsConfig(train) received unsupported parameters: "
                    f"{sorted(unsupported)}"
                )
            batch_size = int(kw.get("batch_size", 128))
            return OpsConfig(
                mode="train",
                in_dim=in_dim,
                out_shape=out_shape,
                cfg_dict=cfg_dict,
                memmap_dir=str(kw["memmap_dir"]),
                ckpt_dir=str(kw["ckpt_dir"]),
                init_ckpt_dir=kw.get("init_ckpt_dir"),
                epochs=int(kw.get("epochs", 5)),
                batch_size=batch_size,
                val_frac=float(kw.get("val_frac", 0.1)),
                base_lr=float(kw.get("base_lr", 1e-3)),
                weight_decay=float(kw.get("weight_decay", 1e-4)),
                warmup_ratio=float(kw.get("warmup_ratio", 0.0)),
                eta_min=float(kw.get("eta_min", 0.0)),
                seed=int(kw.get("seed", 42)),
                prefetch_factor=kw.get("prefetch_factor", 1),
                grad_accum_steps=int(kw.get("grad_accum_steps", 1)),
                overlap_h2d=bool(kw.get("overlap_h2d", True)),
                loss_tile_dim=kw.get("loss_tile_dim"),
                loss_tile_size=kw.get("loss_tile_size"),
                loss_mask_mode=str(kw.get("loss_mask_mode", "none")),
                loss_mask_value=kw.get("loss_mask_value"),
            )
        for k in ("memmap_dir", "keys"):
            if k not in kw or kw[k] is None:
                raise ValueError(f"OpsConfig({mode}) missing required key: {k}")
        allowed = common_keys | {
            "memmap_dir",
            "keys",
            "model_ckpt_dir",
            "batch_size",
            "seed",
            "prefetch_factor",
        }
        unsupported = set(kw) - allowed
        if unsupported:
            raise ValueError(
                f"OpsConfig({mode}) received unsupported parameters: {sorted(unsupported)}"
            )
        batch_size = int(kw.get("batch_size", 512))
        return OpsConfig(
            mode="predict" if mode == "predict" else "infer",
            in_dim=in_dim,
            out_shape=out_shape,
            cfg_dict=cfg_dict,
            memmap_dir=str(kw["memmap_dir"]),
            model_ckpt_dir=kw.get("model_ckpt_dir"),
            keys=list(kw["keys"]),
            batch_size=batch_size,
            seed=int(kw.get("seed", 7)),
            prefetch_factor=kw.get("prefetch_factor", 1),
        )


def coerce_ops_config(config: OpsConfig | Dict[str, Any]) -> OpsConfig:
    if isinstance(config, OpsConfig):
        data: Dict[str, Any] = asdict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError("ops configuration must be OpsConfig or dict")
    cfg_dict = data.get("cfg_dict")
    if cfg_dict is None:
        raise ValueError("cfg_dict is required in ops configuration")
    if not isinstance(cfg_dict, dict):
        try:
            data["cfg_dict"] = dict(cfg_dict)
        except Exception as exc:
            raise TypeError("cfg_dict must be dict-like") from exc
    mode_value = data.pop("mode", None)
    if mode_value is None:
        raise ValueError("ops configuration missing mode")
    mode = str(mode_value).lower()
    if mode not in ("train", "predict", "infer"):
        raise ValueError(f"invalid ops mode: {mode_value}")
    if "keys" in data and data["keys"] is not None:
        data["keys"] = list(data["keys"])
    return OpsConfig.from_partial(mode=mode, **data)


def ops_config(
    mode: OpsMode, base: Dict[str, Any] | None, /, *args: Any, **kwargs: Any
) -> OpsConfig:
    data: Dict[str, Any] = dict(base or {})
    order = OpsConfig.TRAIN_POS_ORDER if mode == "train" else OpsConfig.PRED_POS_ORDER
    for name, val in zip(order, args):
        data[name] = val
    data.update(kwargs)
    actual_mode = data.pop("mode", mode)
    return coerce_ops_config({"mode": actual_mode, **data})

