# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Literal,
                    Optional, Sequence, Tuple, TypeAlias, Union)

if TYPE_CHECKING:
    from ..data.nodes import Source
else:
    Source = Dict[str, Any]

import torch


def _coerce_bool(value: Any, *args: Any, name: str, **kwargs: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip()
        match normalized.lower():
            case "true" | "1":
                return True
            case "false" | "0":
                return False
    raise TypeError(f"{name} must be a boolean-compatible value")


def _coerce_int(
    value: Any,
    *args: Any,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    **kwargs: Any,
) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer-compatible value") from exc
    if minimum is not None and ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {ivalue}")
    if maximum is not None and ivalue > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {ivalue}")
    return ivalue


def _coerce_float(
    value: Any,
    *args: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    **kwargs: Any,
) -> float:
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a float-compatible value") from exc
    if minimum is not None and fvalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {fvalue}")
    if maximum is not None and fvalue > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {fvalue}")
    return fvalue


def _coerce_int_tuple(
    value: Any,
    *args: Any,
    name: str,
    dims: int,
    allow_none: bool = False,
    keep_scalar: bool = False,
    **kwargs: Any,
) -> Optional[Union[int, Tuple[int, ...]]]:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} cannot be None")
    if isinstance(value, int):
        ivalue = _coerce_int(value, name=name, minimum=1)
        if keep_scalar:
            return ivalue
        return tuple([ivalue] * dims)
    if isinstance(value, (list, tuple)):
        if len(value) != dims:
            raise ValueError(f"{name} must have length {dims}, got {len(value)}")
        items = tuple(_coerce_int(v, name=name, minimum=1) for v in value)
        return items
    raise TypeError(f"{name} must be an int or sequence of {dims} integers")


def _coerce_int_sequence(xs: Sequence[int]) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in xs)
    except Exception as exc:
        raise TypeError(f"out_shape must be a sequence of integers, got {xs!r}") from exc


@dataclass(frozen=True)
class PatchConfig:
    is_square: bool = False
    patch_size_1d: int = 16
    grid_size_2d: Optional[Union[int, Tuple[int, int], List[int]]] = None
    patch_size_2d: Union[int, Tuple[int, int], List[int]] = 4
    is_cube: bool = False
    grid_size_3d: Optional[Union[int, Tuple[int, int, int], List[int]]] = None
    patch_size_3d: Union[int, Tuple[int, int, int], List[int]] = (2, 2, 2)
    dropout: float = 0.0
    use_padding: bool = True


@dataclass
class ModelConfig:

    device: Optional[torch.device | str] = None
    dropout: float = 0.1
    normalization_method: str = "layernorm"
    depth: int = 128
    heads: int = 4
    spatial_depth: int = 4
    temporal_depth: int = 4
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    spatial_latents: int = 64
    temporal_latents: int = 64
    modeling_type: str = "spatiotemporal"
    patch: PatchConfig = field(default_factory=PatchConfig)
    use_linear_branch: bool = False
    compile_mode: str = "disabled"
    safety_margin_pow2: int = 3


def coerce_patch_config(
    config: PatchConfig | Dict[str, Any] | None,
) -> PatchConfig:
    if config is None:
        data: Dict[str, Any] = {}
    elif isinstance(config, PatchConfig):
        data = asdict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError("patch configuration must be PatchConfig, dict, or None")
    defaults = asdict(PatchConfig())
    allowed = set(inspect.signature(PatchConfig).parameters.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}
    resolved: Dict[str, Any] = {}
    resolved["is_square"] = _coerce_bool(
        filtered.get("is_square", defaults["is_square"]), name="is_square"
    )
    resolved["patch_size_1d"] = _coerce_int(
        filtered.get("patch_size_1d", defaults["patch_size_1d"]),
        name="patch_size_1d",
        minimum=1,
    )
    grid2d = filtered.get("grid_size_2d", defaults["grid_size_2d"])
    resolved["grid_size_2d"] = (
        _coerce_int_tuple(
            grid2d, name="grid_size_2d", dims=2, allow_none=True, keep_scalar=True
        )
        if grid2d is not None
        else None
    )
    patch2d = filtered.get("patch_size_2d", defaults["patch_size_2d"])
    resolved["patch_size_2d"] = _coerce_int_tuple(
        patch2d, name="patch_size_2d", dims=2, keep_scalar=True
    )
    resolved["is_cube"] = _coerce_bool(
        filtered.get("is_cube", defaults["is_cube"]), name="is_cube"
    )
    grid3d = filtered.get("grid_size_3d", defaults["grid_size_3d"])
    resolved["grid_size_3d"] = (
        _coerce_int_tuple(
            grid3d, name="grid_size_3d", dims=3, allow_none=True, keep_scalar=True
        )
        if grid3d is not None
        else None
    )
    patch3d = filtered.get("patch_size_3d", defaults["patch_size_3d"])
    resolved["patch_size_3d"] = _coerce_int_tuple(
        patch3d, name="patch_size_3d", dims=3, keep_scalar=True
    )
    resolved["dropout"] = _coerce_float(
        filtered.get("dropout", defaults["dropout"]),
        name="patch.dropout",
        minimum=0.0,
        maximum=1.0,
    )
    resolved["use_padding"] = _coerce_bool(
        filtered.get("use_padding", defaults["use_padding"]), name="use_padding"
    )
    if resolved["dropout"] < 0.0 or resolved["dropout"] > 1.0:
        raise ValueError(f"patch.dropout must be in [0,1], got {resolved['dropout']}")
    return PatchConfig(**resolved)


def coerce_model_config(
    config: ModelConfig | Dict[str, Any] | None,
) -> ModelConfig:
    if config is None:
        data: Dict[str, Any] = {}
    elif isinstance(config, ModelConfig):
        data = asdict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError("config must be ModelConfig, dict, or None")

    defaults = ModelConfig()
    allowed = set(inspect.signature(ModelConfig).parameters.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}

    patch_value = filtered.get("patch", getattr(defaults, "patch"))
    patch_cfg = coerce_patch_config(patch_value)

    resolved: Dict[str, Any] = {"patch": patch_cfg}

    device_val = filtered.get("device", getattr(defaults, "device"))
    if device_val is None or device_val == "":
        resolved["device"] = None
    else:
        try:
            resolved["device"] = torch.device(device_val)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid device specification: {device_val}") from exc

    resolved["dropout"] = _coerce_float(
        filtered.get("dropout", getattr(defaults, "dropout")),
        name="dropout",
        minimum=0.0,
        maximum=1.0,
    )
    norm_val = filtered.get(
        "normalization_method", getattr(defaults, "normalization_method")
    )
    resolved["normalization_method"] = str(norm_val)
    resolved["depth"] = _coerce_int(
        filtered.get("depth", getattr(defaults, "depth")),
        name="depth",
        minimum=1,
    )
    resolved["heads"] = _coerce_int(
        filtered.get("heads", getattr(defaults, "heads")),
        name="heads",
        minimum=1,
    )
    resolved["spatial_depth"] = _coerce_int(
        filtered.get("spatial_depth", getattr(defaults, "spatial_depth")),
        name="spatial_depth",
        minimum=1,
    )
    resolved["temporal_depth"] = _coerce_int(
        filtered.get("temporal_depth", getattr(defaults, "temporal_depth")),
        name="temporal_depth",
        minimum=1,
    )
    resolved["mlp_ratio"] = _coerce_float(
        filtered.get("mlp_ratio", getattr(defaults, "mlp_ratio")),
        name="mlp_ratio",
        minimum=0.0,
    )
    resolved["drop_path"] = _coerce_float(
        filtered.get("drop_path", getattr(defaults, "drop_path")),
        name="drop_path",
        minimum=0.0,
        maximum=1.0,
    )
    resolved["spatial_latents"] = _coerce_int(
        filtered.get("spatial_latents", getattr(defaults, "spatial_latents")),
        name="spatial_latents",
        minimum=1,
    )
    resolved["temporal_latents"] = _coerce_int(
        filtered.get("temporal_latents", getattr(defaults, "temporal_latents")),
        name="temporal_latents",
        minimum=1,
    )
    model_type_val = filtered.get(
        "modeling_type", getattr(defaults, "modeling_type")
    )
    resolved["modeling_type"] = str(model_type_val)
    resolved["use_linear_branch"] = _coerce_bool(
        filtered.get("use_linear_branch", getattr(defaults, "use_linear_branch")),
        name="use_linear_branch",
    )
    raw_compile_mode = filtered.get("compile_mode", getattr(defaults, "compile_mode"))
    if raw_compile_mode is None:
        normalized_compile_mode = getattr(defaults, "compile_mode")
    else:
        normalized_compile_mode = str(raw_compile_mode).strip()
        if not normalized_compile_mode:
            normalized_compile_mode = getattr(defaults, "compile_mode")
        else:
            normalized_compile_mode = normalized_compile_mode.lower()
    resolved["compile_mode"] = normalized_compile_mode
    raw_pow2 = filtered.get("safety_margin_pow2", getattr(defaults, "safety_margin_pow2", 3))
    resolved["safety_margin_pow2"] = _coerce_int(
        raw_pow2,
        name="safety_margin_pow2",
        minimum=0,
        maximum=30,
    )

    return ModelConfig(**resolved)


def patch_config(
    base: PatchConfig | Dict[str, Any] | None = None,
    /,
    **overrides: Any,
) -> PatchConfig:
    if base is None:
        data: Dict[str, Any] = {}
    elif isinstance(base, PatchConfig):
        data = asdict(base)
    elif isinstance(base, dict):
        data = dict(base)
    else:
        raise TypeError("base must be PatchConfig, dict, or None")
    data.update(overrides)
    return coerce_patch_config(data)


def model_config(
    base: ModelConfig | Dict[str, Any] | None = None,
    /,
    **overrides: Any,
) -> ModelConfig:
    if base is None:
        data: Dict[str, Any] = {}
    elif isinstance(base, ModelConfig):
        data = asdict(base)
    elif isinstance(base, dict):
        data = dict(base)
    else:
        raise TypeError("base must be ModelConfig, dict, or None")
    data.update(overrides)
    return coerce_model_config(data)


BuildConfig: TypeAlias = ModelConfig


build_config = model_config
coerce_build_config = coerce_model_config
OpsMode = Literal["train", "predict", "infer"]


@dataclass(frozen=True)
class RuntimeConfig:
    mode: OpsMode
    in_dim: int
    out_shape: Tuple[int, ...]
    cfg_dict: Dict[str, Any]
    sources: Optional[Union[Source, Sequence[Source], Dict[str, Source]]] = None
    ckpt_dir: Optional[str] = None
    init_ckpt_dir: Optional[str] = None
    epochs: int = 5
    val_frac: float = 0.1
    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.0
    eta_min: float = 0.0
    seed: int = 42
    shuffle: bool = True
    deterministic: bool = False
    loss_tile_dim: Optional[int] = None
    loss_tile_size: Optional[int] = None
    loss_mask_mode: str = "none"
    loss_mask_value: Optional[float] = None
    swa_enabled: bool = False
    swa_start_epoch: Optional[int] = None
    swa_update_batch_norm: bool = False
    model_ckpt_dir: Optional[str] = None
    keys: Optional[List[Tuple[Any, ...]]] = None
    loss_skew: bool = True
    TRAIN_POS_ORDER: ClassVar[Tuple[str, ...]] = (
        "epochs",
        "val_frac",
        "base_lr",
        "weight_decay",
        "warmup_ratio",
        "eta_min",
        "seed",
        "loss_tile_dim",
        "loss_tile_size",
        "loss_mask_mode",
        "loss_mask_value",
        "swa_enabled",
        "swa_start_epoch",
        "swa_update_batch_norm",
        "loss_skew",
    )
    PRED_POS_ORDER: ClassVar[Tuple[str, ...]] = ("seed",)

    @staticmethod
    def from_partial(mode: OpsMode, *args: Any, **kwargs: Any) -> "RuntimeConfig":
        kwargs = dict(kwargs)
        for k in ("in_dim", "out_shape", "cfg_dict"):
            if k not in kwargs or kwargs[k] is None:
                raise ValueError(f"RuntimeConfig missing required key: {k}")
        in_dim = _coerce_int(kwargs["in_dim"], name="in_dim", minimum=1)
        out_shape = _coerce_int_sequence(kwargs["out_shape"])
        if not out_shape:
            raise ValueError("RuntimeConfig.out_shape must be a non-empty sequence")
        cfg_dict = dict(kwargs["cfg_dict"])
        common_keys = {
            "in_dim",
            "out_shape",
            "cfg_dict",
        }
        if mode == "train":
            for k in ("sources", "ckpt_dir"):
                if k not in kwargs or kwargs[k] is None:
                    raise ValueError(f"RuntimeConfig(train) missing required key: {k}")
            allowed = common_keys | {
                "sources",
                "ckpt_dir",
                "init_ckpt_dir",
                "epochs",
                "val_frac",
                "base_lr",
                "weight_decay",
                "warmup_ratio",
                "eta_min",
                "seed",
                "shuffle",
                "deterministic",
                "loss_tile_dim",
                "loss_tile_size",
                "loss_mask_mode",
                "loss_mask_value",
                "swa_enabled",
                "swa_start_epoch",
                "swa_update_batch_norm",
                "loss_skew",
            }
            unsupported = set(kwargs) - allowed
            if unsupported:
                raise ValueError(
                    "RuntimeConfig(train) received unsupported parameters: "
                    f"{sorted(unsupported)}"
                )
            epochs = _coerce_int(kwargs.get("epochs", 5), name="epochs", minimum=1)
            val_frac = _coerce_float(kwargs.get("val_frac", 0.1), name="val_frac")
            if val_frac < 0.0 or val_frac > 1.0:
                raise ValueError(f"val_frac must be in [0,1], got {val_frac}")
            base_lr = _coerce_float(kwargs.get("base_lr", 1e-3), name="base_lr", minimum=0.0)
            weight_decay = _coerce_float(
                kwargs.get("weight_decay", 1e-4), name="weight_decay", minimum=0.0
            )

            return RuntimeConfig(
                mode="train",
                in_dim=in_dim,
                out_shape=out_shape,
                cfg_dict=cfg_dict,
                sources=kwargs["sources"],
                ckpt_dir=str(kwargs["ckpt_dir"]),
                init_ckpt_dir=kwargs.get("init_ckpt_dir"),
                epochs=epochs,
                val_frac=val_frac,
                base_lr=base_lr,
                weight_decay=weight_decay,
                warmup_ratio=float(kwargs.get("warmup_ratio", 0.0)),
                eta_min=float(kwargs.get("eta_min", 0.0)),
                seed=int(kwargs.get("seed", 42)),
                shuffle=bool(kwargs.get("shuffle", True)),
                deterministic=bool(kwargs.get("deterministic", False)),
                loss_tile_dim=kwargs.get("loss_tile_dim"),
                loss_tile_size=kwargs.get("loss_tile_size"),
                loss_mask_mode=str(kwargs.get("loss_mask_mode", "none")),
                loss_mask_value=kwargs.get("loss_mask_value"),
                swa_enabled=bool(kwargs.get("swa_enabled", False)),
                swa_start_epoch=kwargs.get("swa_start_epoch"),
                swa_update_batch_norm=bool(kwargs.get("swa_update_batch_norm", False)),
                loss_skew=bool(kwargs.get("loss_skew", True)),
            )
        for k in ("sources", "keys"):
            if k not in kwargs or kwargs[k] is None:
                raise ValueError(f"RuntimeConfig({mode}) missing required key: {k}")
        allowed = common_keys | {
            "sources",
            "keys",
            "model_ckpt_dir",
            "seed",
            "ckpt_dir",
            "loss_skew",
        }
        unsupported = set(kwargs) - allowed
        if unsupported:
            raise ValueError(
                f"RuntimeConfig({mode}) received unsupported parameters: {sorted(unsupported)}"
            )
        return RuntimeConfig(
            mode="predict" if mode == "predict" else "infer",
            in_dim=in_dim,
            out_shape=out_shape,
            cfg_dict=cfg_dict,
            sources=kwargs["sources"],
            ckpt_dir=kwargs.get("ckpt_dir"),
            model_ckpt_dir=kwargs.get("model_ckpt_dir"),
            keys=list(kwargs["keys"]),
            seed=int(kwargs.get("seed", 7)),
            loss_skew=bool(kwargs.get("loss_skew", True)),
        )


def coerce_runtime_config(config: RuntimeConfig | Dict[str, Any]) -> RuntimeConfig:
    if isinstance(config, RuntimeConfig):
        data: Dict[str, Any] = asdict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError("runtime configuration must be RuntimeConfig or dict")
    cfg_dict = data.get("cfg_dict")
    if cfg_dict is None:
        raise ValueError("cfg_dict is required in runtime configuration")
    if not isinstance(cfg_dict, dict):
        try:
            data["cfg_dict"] = dict(cfg_dict)
        except Exception as exc:
            raise TypeError("cfg_dict must be dict-like") from exc
    mode_value = data.pop("mode", None)
    if mode_value is None:
        raise ValueError("runtime configuration missing mode")
    mode = str(mode_value).lower()
    if mode not in ("train", "predict", "infer"):
        raise ValueError(f"invalid runtime mode: {mode_value}")
    if "keys" in data and data["keys"] is not None:
        try:
            data["keys"] = list(data["keys"])
        except Exception as exc:
            raise TypeError("RuntimeConfig.keys must be list-like") from exc
    return RuntimeConfig.from_partial(mode=mode, **data)


def runtime_config(
    mode: OpsMode, base: Dict[str, Any] | None, /, *args: Any, **kwargs: Any
) -> RuntimeConfig:
    data: Dict[str, Any] = dict(base or {})
    order = (
        RuntimeConfig.TRAIN_POS_ORDER
        if mode == "train"
        else RuntimeConfig.PRED_POS_ORDER
    )
    for name, val in zip(order, args):
        data[name] = val
    data.update(kwargs)
    actual_mode = data.pop("mode", mode)
    return coerce_runtime_config({"mode": actual_mode, **data})
