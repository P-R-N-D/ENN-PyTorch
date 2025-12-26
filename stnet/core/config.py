# -*- coding: utf-8 -*-
from __future__ import annotations

"""
stnet/core/config.py

Configuration dataclasses + coercion/validation helpers.

Design goals:
- Avoid dataclasses.asdict() on potentially large objects (it recurses and deepcopy()'s
  non-container values), to reduce memory peaks and improve thread scalability.
  See Python docs for dataclasses.asdict() behavior.  (Ref: docs.python.org)
- Provide strict, predictable coercion with clear error messages.
- Enforce PatchConfig.is_square / is_cube as value constraints.
"""

from dataclasses import dataclass, field, fields
import math
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import torch

if TYPE_CHECKING:
    from ..data.nodes import Source  # pragma: no cover
else:
    Source = Dict[str, Any]


# -----------------------------
# Small helpers (no deep-copy)
# -----------------------------

def _shallow_dataclass_dict(obj: Any) -> Dict[str, Any]:
    """Shallow conversion of a dataclass instance to a dict (no recursion, no deepcopy)."""
    return {f.name: getattr(obj, f.name) for f in fields(obj.__class__)}


def _shallow_copy_if_container(value: Any) -> Any:
    """Shallow-copy common containers (no deep recursion).

    This keeps the "no deepcopy" design goal, while avoiding surprising aliasing when a config
    field happens to contain a mutable container (e.g. list).
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, set):
        return set(value)
    return value



def _field_name_set(cls: type) -> frozenset[str]:
    return frozenset(f.name for f in fields(cls))


def _coerce_str(
    value: Any,
    *,
    name: str,
    default: str,
    lower: bool = False,
    strip: bool = True,
) -> str:
    """
    Coerce to string.
    - None or blank -> default
    - optionally lower-case
    """
    if value is None:
        return default
    s = str(value)
    if strip:
        s = s.strip()
    if not s:
        return default
    return s.lower() if lower else s


def _coerce_bool(value: Any, *args: Any, name: str, **kwargs: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        # Keep the accepted vocabulary small/predictable.
        match normalized:
            case "true" | "1" | "yes" | "y" | "on" | "t":
                return True
            case "false" | "0" | "no" | "n" | "off" | "f":
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
    # bool is a subclass of int -> reject explicitly.
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer-compatible value (bool not allowed)")
    try:
        ivalue = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
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
    finite: bool = True,
    **kwargs: Any,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float-compatible value (bool not allowed)")
    try:
        fvalue = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a float-compatible value") from exc
    if finite and not math.isfinite(fvalue):
        raise ValueError(f"{name} must be finite, got {fvalue}")
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

    # scalar -> broadcast or keep
    if isinstance(value, int) and not isinstance(value, bool):
        ivalue = _coerce_int(value, name=name, minimum=1)
        return ivalue if keep_scalar else (ivalue,) * dims

    # list/tuple -> validate
    if isinstance(value, (list, tuple)):
        if len(value) != dims:
            raise ValueError(f"{name} must have length {dims}, got {len(value)}")
        return tuple(_coerce_int(v, name=name, minimum=1) for v in value)

    raise TypeError(f"{name} must be an int or sequence of {dims} integers")


def _coerce_int_sequence(
    xs: Sequence[Any],
    *,
    name: str = "out_shape",
    minimum: Optional[int] = None,
) -> Tuple[int, ...]:
    try:
        return tuple(_coerce_int(x, name=name, minimum=minimum) for x in xs)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers, got {xs!r}") from exc


def _coerce_device(value: Any, *, name: str = "device") -> Optional[torch.device]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, torch.device):
        return value
    try:
        return torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"{name} has invalid device specification: {value!r}") from exc


def _as_tuple_nd(value: Union[int, Tuple[int, ...]], *, dims: int, name: str) -> Tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,) * dims
    if isinstance(value, tuple):
        if len(value) != dims:
            raise ValueError(f"{name} must have length {dims}, got {len(value)}")
        return value
    raise TypeError(f"{name} must be int or tuple of length {dims}")


def _enforce_equal_dims(value: Union[int, Tuple[int, ...]], *, dims: int, name: str) -> None:
    t = _as_tuple_nd(value, dims=dims, name=name)
    first = t[0]
    if any(v != first for v in t[1:]):
        raise ValueError(f"{name} must have equal dimensions (dims={dims}), got {t}")


# -----------------------------
# Patch / Model configuration
# -----------------------------

@dataclass(frozen=True)
class PatchConfig:
    is_square: bool = False
    patch_size_1d: int = 16

    # 2D
    grid_size_2d: Optional[Union[int, Tuple[int, int], list[int]]] = None
    patch_size_2d: Union[int, Tuple[int, int], list[int]] = 4

    # 3D
    is_cube: bool = False
    grid_size_3d: Optional[Union[int, Tuple[int, int, int], list[int]]] = None
    patch_size_3d: Union[int, Tuple[int, int, int], list[int]] = (2, 2, 2)

    dropout: float = 0.0
    use_padding: bool = True


@dataclass
class ModelConfig:
    device: Optional[torch.device | str] = None
    dropout: float = 0.1
    normalization_method: str = "layernorm"
    d_model: int = 128
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert this config into a JSON-friendly dict.

        - Shallow conversion only (no recursion / deepcopy).
        - Ensures `patch` is a plain dict.
        - Ensures `device` is serializable (string) when it's a torch.device.
        """
        data = _shallow_dataclass_dict(self)
        data["patch"] = patch_config_to_dict(getattr(self, "patch", None))

        dev = data.get("device")
        if isinstance(dev, torch.device):
            data["device"] = str(dev)

        return {k: _shallow_copy_if_container(v) for k, v in data.items()}


_PATCH_FIELDS = _field_name_set(PatchConfig)
_MODEL_FIELDS = _field_name_set(ModelConfig)
_PATCH_DEFAULTS = PatchConfig()
_MODEL_DEFAULTS = ModelConfig()


def coerce_patch_config(config: PatchConfig | Dict[str, Any] | None) -> PatchConfig:
    if config is None:
        data: Dict[str, Any] = {}
    elif isinstance(config, PatchConfig):
        data = _shallow_dataclass_dict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError("patch configuration must be PatchConfig, dict, or None")

    filtered = {k: v for k, v in data.items() if k in _PATCH_FIELDS}
    get = filtered.get

    is_square = _coerce_bool(get("is_square", _PATCH_DEFAULTS.is_square), name="is_square")
    patch_size_1d = _coerce_int(get("patch_size_1d", _PATCH_DEFAULTS.patch_size_1d), name="patch_size_1d", minimum=1)

    grid_size_2d = _coerce_int_tuple(
        get("grid_size_2d", _PATCH_DEFAULTS.grid_size_2d),
        name="grid_size_2d",
        dims=2,
        allow_none=True,
        keep_scalar=True,
    )
    patch_size_2d = _coerce_int_tuple(
        get("patch_size_2d", _PATCH_DEFAULTS.patch_size_2d),
        name="patch_size_2d",
        dims=2,
        keep_scalar=True,
    )

    is_cube = _coerce_bool(get("is_cube", _PATCH_DEFAULTS.is_cube), name="is_cube")
    grid_size_3d = _coerce_int_tuple(
        get("grid_size_3d", _PATCH_DEFAULTS.grid_size_3d),
        name="grid_size_3d",
        dims=3,
        allow_none=True,
        keep_scalar=True,
    )
    patch_size_3d = _coerce_int_tuple(
        get("patch_size_3d", _PATCH_DEFAULTS.patch_size_3d),
        name="patch_size_3d",
        dims=3,
        keep_scalar=True,
    )

    dropout = _coerce_float(
        get("dropout", _PATCH_DEFAULTS.dropout),
        name="patch.dropout",
        minimum=0.0,
        maximum=1.0,
    )
    use_padding = _coerce_bool(get("use_padding", _PATCH_DEFAULTS.use_padding), name="use_padding")

    # Value constraints:
    # - is_square => 2D sizes must be equal in each dimension
    # - is_cube   => 3D sizes must be equal in each dimension
    if is_square:
        if patch_size_2d is None:
            raise ValueError("patch_size_2d is required when is_square is True")
        _enforce_equal_dims(patch_size_2d, dims=2, name="patch_size_2d")
        if grid_size_2d is not None:
            _enforce_equal_dims(grid_size_2d, dims=2, name="grid_size_2d")

    if is_cube:
        if patch_size_3d is None:
            raise ValueError("patch_size_3d is required when is_cube is True")
        _enforce_equal_dims(patch_size_3d, dims=3, name="patch_size_3d")
        if grid_size_3d is not None:
            _enforce_equal_dims(grid_size_3d, dims=3, name="grid_size_3d")

    return PatchConfig(
        is_square=is_square,
        patch_size_1d=patch_size_1d,
        grid_size_2d=grid_size_2d,
        patch_size_2d=patch_size_2d,
        is_cube=is_cube,
        grid_size_3d=grid_size_3d,
        patch_size_3d=patch_size_3d,
        dropout=dropout,
        use_padding=use_padding,
    )


def coerce_model_config(config: ModelConfig | Dict[str, Any] | None) -> ModelConfig:
    if config is None:
        data: Dict[str, Any] = {}
    elif isinstance(config, ModelConfig):
        data = _shallow_dataclass_dict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError("config must be ModelConfig, dict, or None")

    filtered = {k: v for k, v in data.items() if k in _MODEL_FIELDS}
    get = filtered.get

    patch_cfg = coerce_patch_config(get("patch", _MODEL_DEFAULTS.patch))

    device = _coerce_device(get("device", _MODEL_DEFAULTS.device), name="device")
    dropout = _coerce_float(
        get("dropout", _MODEL_DEFAULTS.dropout),
        name="dropout",
        minimum=0.0,
        maximum=1.0,
    )

    normalization_method = _coerce_str(
        get("normalization_method", _MODEL_DEFAULTS.normalization_method),
        name="normalization_method",
        default=_MODEL_DEFAULTS.normalization_method,
        lower=True,
    )

    d_model = _coerce_int(get("d_model", _MODEL_DEFAULTS.d_model), name="d_model", minimum=1)
    heads = _coerce_int(get("heads", _MODEL_DEFAULTS.heads), name="heads", minimum=1)
    spatial_depth = _coerce_int(get("spatial_depth", _MODEL_DEFAULTS.spatial_depth), name="spatial_depth", minimum=1)
    temporal_depth = _coerce_int(get("temporal_depth", _MODEL_DEFAULTS.temporal_depth), name="temporal_depth", minimum=1)

    mlp_ratio = _coerce_float(get("mlp_ratio", _MODEL_DEFAULTS.mlp_ratio), name="mlp_ratio", minimum=0.0)
    drop_path = _coerce_float(
        get("drop_path", _MODEL_DEFAULTS.drop_path),
        name="drop_path",
        minimum=0.0,
        maximum=1.0,
    )

    spatial_latents = _coerce_int(get("spatial_latents", _MODEL_DEFAULTS.spatial_latents), name="spatial_latents", minimum=1)
    temporal_latents = _coerce_int(get("temporal_latents", _MODEL_DEFAULTS.temporal_latents), name="temporal_latents", minimum=1)

    modeling_type = _coerce_str(
        get("modeling_type", _MODEL_DEFAULTS.modeling_type),
        name="modeling_type",
        default=_MODEL_DEFAULTS.modeling_type,
        lower=True,
    )

    use_linear_branch = _coerce_bool(
        get("use_linear_branch", _MODEL_DEFAULTS.use_linear_branch),
        name="use_linear_branch",
    )

    compile_mode = _coerce_str(
        get("compile_mode", _MODEL_DEFAULTS.compile_mode),
        name="compile_mode",
        default=_MODEL_DEFAULTS.compile_mode,
        lower=True,
    )

    safety_margin_pow2 = _coerce_int(
        get("safety_margin_pow2", _MODEL_DEFAULTS.safety_margin_pow2),
        name="safety_margin_pow2",
        minimum=0,
        maximum=30,
    )

    # Keep ModelConfig.device type compatible with original annotation (device|str|None)
    device_out: Optional[torch.device | str] = device

    return ModelConfig(
        device=device_out,
        dropout=dropout,
        normalization_method=normalization_method,
        d_model=d_model,
        heads=heads,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        spatial_latents=spatial_latents,
        temporal_latents=temporal_latents,
        modeling_type=modeling_type,
        patch=patch_cfg,
        use_linear_branch=use_linear_branch,
        compile_mode=compile_mode,
        safety_margin_pow2=safety_margin_pow2,
    )


def patch_config_to_dict(config: PatchConfig | Dict[str, Any] | None) -> Dict[str, Any]:
    """Convert PatchConfig/dict/None into a JSON-friendly dict without dataclasses.asdict()."""
    cfg = coerce_patch_config(config)
    data = _shallow_dataclass_dict(cfg)
    # Copy list/dict/set containers one level deep (no recursion).
    return {k: _shallow_copy_if_container(v) for k, v in data.items()}


def model_config_to_dict(config: ModelConfig | Dict[str, Any] | None) -> Dict[str, Any]:
    """Convert ModelConfig/dict/None into a JSON-friendly dict.

    Prefer calling :meth:`ModelConfig.to_dict` directly.
    """
    if config is None:
        return {}
    cfg = config if isinstance(config, ModelConfig) else coerce_model_config(config)
    return cfg.to_dict()


def patch_config(base: PatchConfig | Dict[str, Any] | None = None, /, **overrides: Any) -> PatchConfig:
    if base is None:
        data: Dict[str, Any] = {}
    elif isinstance(base, PatchConfig):
        data = _shallow_dataclass_dict(base)
    elif isinstance(base, dict):
        data = dict(base)
    else:
        raise TypeError("base must be PatchConfig, dict, or None")
    if overrides:
        data.update(overrides)
    return coerce_patch_config(data)


def model_config(base: ModelConfig | Dict[str, Any] | None = None, /, **overrides: Any) -> ModelConfig:
    if base is None:
        data: Dict[str, Any] = {}
    elif isinstance(base, ModelConfig):
        data = _shallow_dataclass_dict(base)
    elif isinstance(base, dict):
        data = dict(base)
    else:
        raise TypeError("base must be ModelConfig, dict, or None")
    if overrides:
        data.update(overrides)
    return coerce_model_config(data)


BuildConfig: TypeAlias = ModelConfig
build_config = model_config
coerce_build_config = coerce_model_config


# -----------------------------
# Runtime configuration
# -----------------------------

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

    # Optional original sample keys for predict/infer.
    # Avoid eager list materialization (can be huge); keep as any sequence/range.
    keys: Optional[Sequence[Any]] = None

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

    # Allowed kw sets (excluding `mode` which is positional/explicit for this ctor helper)
    _COMMON_KEYS: ClassVar[frozenset[str]] = frozenset({"in_dim", "out_shape", "cfg_dict"})
    _TRAIN_KEYS: ClassVar[frozenset[str]] = _COMMON_KEYS | frozenset(
        {
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
    )
    _PRED_KEYS: ClassVar[frozenset[str]] = _COMMON_KEYS | frozenset(
        {
            "sources",
            "ckpt_dir",
            "model_ckpt_dir",
            "keys",
            "seed",
            "shuffle",
            "loss_skew",
        }
    )

    @staticmethod
    def from_partial(mode: OpsMode, *args: Any, **kwargs: Any) -> "RuntimeConfig":
        """
        Build RuntimeConfig from partial inputs.

        Positional args are supported for backward compatibility:
        - train: RuntimeConfig.TRAIN_POS_ORDER
        - predict/infer: RuntimeConfig.PRED_POS_ORDER
        """
        if "mode" in kwargs:
            raise TypeError("RuntimeConfig.from_partial() does not accept 'mode' in kwargs")

        mode_norm = str(mode).lower()
        if mode_norm not in ("train", "predict", "infer"):
            raise ValueError(f"invalid runtime mode: {mode}")

        order = RuntimeConfig.TRAIN_POS_ORDER if mode_norm == "train" else RuntimeConfig.PRED_POS_ORDER
        if len(args) > len(order):
            raise TypeError(f"too many positional args for mode={mode_norm}: got {len(args)}, max {len(order)}")

        # Map positional args onto kwargs (do not silently override existing kwargs)
        data = dict(kwargs)
        for name, val in zip(order, args):
            if name in data:
                raise TypeError(f"{name} specified both positionally and as a keyword")
            data[name] = val

        # Required common keys
        for k in ("in_dim", "out_shape", "cfg_dict"):
            if k not in data or data[k] is None:
                raise ValueError(f"RuntimeConfig missing required key: {k}")

        in_dim = _coerce_int(data["in_dim"], name="in_dim", minimum=1)
        out_shape = _coerce_int_sequence(data["out_shape"], name="out_shape", minimum=1)
        if not out_shape:
            raise ValueError("RuntimeConfig.out_shape must be a non-empty sequence")

        cfg_obj = data["cfg_dict"]
        if isinstance(cfg_obj, dict):
            cfg_dict: Dict[str, Any] = cfg_obj
        else:
            try:
                cfg_dict = dict(cfg_obj)
            except Exception as exc:
                raise TypeError("cfg_dict must be dict-like") from exc

        if mode_norm == "train":
            # Required train keys
            for k in ("sources", "ckpt_dir"):
                if k not in data or data[k] is None:
                    raise ValueError(f"RuntimeConfig(train) missing required key: {k}")

            unsupported = set(data) - set(RuntimeConfig._TRAIN_KEYS)
            if unsupported:
                raise ValueError(
                    "RuntimeConfig(train) received unsupported parameters: "
                    f"{sorted(unsupported)}"
                )

            epochs = _coerce_int(data.get("epochs", 5), name="epochs", minimum=1)
            val_frac = _coerce_float(data.get("val_frac", 0.1), name="val_frac", minimum=0.0, maximum=1.0)
            base_lr = _coerce_float(data.get("base_lr", 1e-3), name="base_lr", minimum=0.0)
            weight_decay = _coerce_float(data.get("weight_decay", 1e-4), name="weight_decay", minimum=0.0)
            warmup_ratio = _coerce_float(data.get("warmup_ratio", 0.0), name="warmup_ratio", minimum=0.0, maximum=1.0)
            eta_min = _coerce_float(data.get("eta_min", 0.0), name="eta_min", minimum=0.0)

            seed = _coerce_int(data.get("seed", 42), name="seed")
            shuffle = _coerce_bool(data.get("shuffle", True), name="shuffle")
            deterministic = _coerce_bool(data.get("deterministic", False), name="deterministic")

            loss_tile_dim = data.get("loss_tile_dim")
            if loss_tile_dim is not None:
                loss_tile_dim = _coerce_int(loss_tile_dim, name="loss_tile_dim", minimum=1)

            loss_tile_size = data.get("loss_tile_size")
            if loss_tile_size is not None:
                loss_tile_size = _coerce_int(loss_tile_size, name="loss_tile_size", minimum=1)

            loss_mask_mode = _coerce_str(
                data.get("loss_mask_mode", "none"),
                name="loss_mask_mode",
                default="none",
                lower=True,
            )

            loss_mask_value = data.get("loss_mask_value")
            if loss_mask_value is not None:
                loss_mask_value = _coerce_float(loss_mask_value, name="loss_mask_value", finite=True)

            swa_enabled = _coerce_bool(data.get("swa_enabled", False), name="swa_enabled")
            swa_start_epoch = data.get("swa_start_epoch")
            if swa_start_epoch is not None:
                swa_start_epoch = _coerce_int(swa_start_epoch, name="swa_start_epoch", minimum=0)
            swa_update_batch_norm = _coerce_bool(
                data.get("swa_update_batch_norm", False),
                name="swa_update_batch_norm",
            )

            loss_skew = _coerce_bool(data.get("loss_skew", True), name="loss_skew")

            ckpt_dir = str(data["ckpt_dir"])
            init_ckpt_dir = data.get("init_ckpt_dir")
            if init_ckpt_dir is not None:
                init_ckpt_dir = str(init_ckpt_dir)

            return RuntimeConfig(
                mode="train",
                in_dim=in_dim,
                out_shape=out_shape,
                cfg_dict=cfg_dict,
                sources=data["sources"],
                ckpt_dir=ckpt_dir,
                init_ckpt_dir=init_ckpt_dir,
                epochs=epochs,
                val_frac=val_frac,
                base_lr=base_lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                eta_min=eta_min,
                seed=seed,
                shuffle=shuffle,
                deterministic=deterministic,
                loss_tile_dim=loss_tile_dim,
                loss_tile_size=loss_tile_size,
                loss_mask_mode=loss_mask_mode,
                loss_mask_value=loss_mask_value,
                swa_enabled=swa_enabled,
                swa_start_epoch=swa_start_epoch,
                swa_update_batch_norm=swa_update_batch_norm,
                loss_skew=loss_skew,
            )

        # predict/infer
        for k in ("sources", "ckpt_dir"):
            if k not in data or data[k] is None:
                raise ValueError(f"RuntimeConfig({mode_norm}) missing required key: {k}")

        unsupported = set(data) - set(RuntimeConfig._PRED_KEYS)
        if unsupported:
            raise ValueError(
                f"RuntimeConfig({mode_norm}) received unsupported parameters: {sorted(unsupported)}"
            )

        seed = _coerce_int(data.get("seed", 7), name="seed")
        shuffle = _coerce_bool(data.get("shuffle", False), name="shuffle")
        loss_skew = _coerce_bool(data.get("loss_skew", True), name="loss_skew")

        ckpt_dir = str(data["ckpt_dir"])
        model_ckpt_dir = data.get("model_ckpt_dir")
        if model_ckpt_dir is not None:
            model_ckpt_dir = str(model_ckpt_dir)

        return RuntimeConfig(
            mode="predict" if mode_norm == "predict" else "infer",
            in_dim=in_dim,
            out_shape=out_shape,
            cfg_dict=cfg_dict,
            sources=data["sources"],
            ckpt_dir=ckpt_dir,
            model_ckpt_dir=model_ckpt_dir,
            keys=data.get("keys"),
            seed=seed,
            shuffle=shuffle,
            loss_skew=loss_skew,
        )


def coerce_runtime_config(config: RuntimeConfig | Dict[str, Any]) -> RuntimeConfig:
    if isinstance(config, RuntimeConfig):
        data: Dict[str, Any] = _shallow_dataclass_dict(config)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        raise TypeError("runtime configuration must be RuntimeConfig or dict")

    mode_value = data.pop("mode", None)
    if mode_value is None:
        raise ValueError("runtime configuration missing mode")
    mode = str(mode_value).lower()
    if mode not in ("train", "predict", "infer"):
        raise ValueError(f"invalid runtime mode: {mode_value}")

    # RuntimeConfig.from_partial handles cfg_dict dict-like conversion and validation.
    return RuntimeConfig.from_partial(mode=mode, **data)


def runtime_config(mode: OpsMode, base: Dict[str, Any] | None, /, *args: Any, **kwargs: Any) -> RuntimeConfig:
    """
    Convenience builder:
    - Merge base + kwargs
    - Determine mode (kwargs/base can override)
    - Map positional args according to that mode
    """
    data: Dict[str, Any] = dict(base or {})
    if kwargs:
        data.update(kwargs)

    actual_mode = data.pop("mode", mode)
    return RuntimeConfig.from_partial(actual_mode, *args, **data)
