# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, Literal, Optional,
                    Sequence, Tuple, TypeAlias, Union)

import torch

if TYPE_CHECKING:
    from .data.nodes import Source  # pragma: no cover
else:
    Source = Dict[str, Any]


# -----------------------------
# Small helpers (no deep-copy)
# -----------------------------

def _shallow_dataclass_dict(obj: Any) -> Dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj.__class__)}


def _shallow_copy_if_container(value: Any) -> Any:
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


    # Residual gating: z_hat = base + p * residue
    p_gate_enabled: bool = False
    p_gate_hidden_dim: int = 64
    p_gate_detach_inputs: bool = True

    # Tile-wise p gating (optional). When set, p is predicted per output tile
    # along the flattened y dimension, matching TiledLoss slicing.
    # - None or <=0: scalar p per sample (default).
    # - >0: tile size (number of y dims per tile).
    p_gate_tile_size: Optional[int] = None

    # Optional quantile-based bounds for p tightening. When enabled and
    # y_quantile bounds are available, p is constrained so z_hat stays within
    # the chosen per-dimension quantile range (more robust than min/max).
    p_gate_bounds_use_quantile: bool = False
    p_gate_bounds_q_low: float = 0.005
    p_gate_bounds_q_high: float = 0.995
    p_gate_bounds_q_max_samples: int = 8192
    p_gate_bounds_clip_to_minmax: bool = True
    p_gate_p_floor: float = 0.0
    p_gate_p_ceil: float = 1.0
    # If y_min/y_max bounds are unavailable (legacy scaler stats), fall back to
    # mean ± k * std bounds (in y-space). Set <=0 to disable fallback.
    p_gate_fallback_k: float = 6.0
    # Optional asymmetric fallback bounds: mean - k_low*std, mean + k_high*std.
    # When None, defaults to p_gate_fallback_k.
    p_gate_fallback_k_low: Optional[float] = None
    p_gate_fallback_k_high: Optional[float] = None

    # Optional auto-tuning for p_gate_fallback_k when y_min/y_max are missing.
    # Uses simple moving average (SMA) over a recent interval and an EMA
    # smoother to detect when the fallback bounds are overly restrictive.
    p_gate_auto_k_enabled: bool = False
    # Update cadence in *optimizer steps* (not micro-steps). Set <=0 to disable.
    p_gate_auto_k_interval: int = 100
    # Warmup steps before the first tuning update.
    p_gate_auto_k_warmup: int = 100
    # EMA smoothing factor for the constraint-activation metric.
    p_gate_auto_k_ema_alpha: float = 0.1
    # Target fraction of samples considered "tight" (i.e., fallback bounds
    # meaningfully clip the allowed [p_floor, p_ceil] interval on either side).
    p_gate_auto_k_target_tight: float = 0.02
    # Relative tolerance around the target (e.g., 0.5 -> [0.5x, 1.5x]).
    p_gate_auto_k_tolerance: float = 0.5
    # Multiplicative adjustment rates (symmetric default).
    p_gate_auto_k_step_up: float = 0.1
    p_gate_auto_k_step_down: float = 0.02
    # Optional per-side adjustment rates (override when provided).
    p_gate_auto_k_step_up_low: float = 0.1
    p_gate_auto_k_step_down_low: float = 0.02
    p_gate_auto_k_step_up_high: float = 0.1
    p_gate_auto_k_step_down_high: float = 0.02

    # Optional edge-based tuning: when constraint activation is within target,
    # but p frequently hugs dynamic endpoints, shrink bounds by reducing k.
    p_gate_auto_k_edge_enabled: bool = False
    p_gate_auto_k_target_edge: float = 0.05
    p_gate_auto_k_edge_tolerance: float = 0.5
    p_gate_auto_k_edge_ema_alpha: float = 0.1
    p_gate_auto_k_edge_step_down_low: float = 0.01
    p_gate_auto_k_edge_step_down_high: float = 0.01

    # Hard clamp for fallback_k.
    p_gate_auto_k_min: float = 1.0
    p_gate_auto_k_max: float = 16.0
    # A sample is "tight" when either side trims at least
    # width_frac * (p_ceil - p_floor), i.e.:
    #   (p_low - p_floor) >= width_frac * (p_ceil - p_floor)  OR
    #   (p_ceil - p_high) >= width_frac * (p_ceil - p_floor)
    p_gate_auto_k_width_frac: float = 0.05
    # For diagnostics: count samples where p is near the dynamic endpoints
    # [p_low, p_high] ("boundary hugging").
    p_gate_auto_k_edge_frac: float = 0.02
    # Log cadence (optimizer steps). Set <=0 to log only on actual k changes.
    p_gate_auto_k_log_interval: int = 200
    p_gate_clip_eps: float = 1e-6
    p_gate_eps: float = 1e-6

    # Optional edge-hugging regularizer for p: discourages p from sticking to
    # dynamic endpoints [p_low, p_high] when the allowed interval is wide enough.
    p_gate_edge_reg_weight: float = 0.0
    # Optional per-side weights (if None, use p_gate_edge_reg_weight).
    p_gate_edge_reg_weight_low: Optional[float] = None
    p_gate_edge_reg_weight_high: Optional[float] = None
    # If True, apply edge regularizer only when using fallback bounds (mean±kσ),
    # not when true y_min/y_max bounds are available.
    p_gate_edge_reg_fallback_only: bool = False
    p_gate_edge_reg_frac: float = 0.02
    p_gate_edge_reg_min_width_frac: float = 0.05
    p_gate_edge_reg_power: float = 2.0

    # Auxiliary losses (added to total loss inside Model.forward)
    unsup_xx_weight: float = 0.0
    unsup_yy_weight: float = 0.0

    # Prior regularization for p (optional; Beta prior on [0,1])
    p_prior_weight: float = 0.0
    p_prior_alpha: float = 2.0
    p_prior_beta: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
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

    # Canonicalize for stable serialization / CLI friendliness.
    # Accept common separators (space/_/-) and fold to known norms where possible.
    _norm_key = normalization_method.replace(" ", "").replace("_", "").replace("-", "")
    match _norm_key:
        case "ln" | "layernorm":
            normalization_method = "layernorm"
        case "bn" | "batchnorm":
            normalization_method = "batchnorm"
        case "rms" | "rmsnorm":
            normalization_method = "rmsnorm"
        case _:
            pass

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

    # Canonicalize modeling_type: normalize separators and map synonyms to {ss, tt, st}.
    _mt_key = modeling_type.replace("_", "-").replace(" ", "-")
    if "-" in _mt_key:
        _mt_key = "-".join(part for part in _mt_key.split("-") if part)

    match _mt_key:
        case "ss" | "spatial" | "sxs":
            modeling_type = "ss"
        case "tt" | "temporal" | "txt":
            modeling_type = "tt"
        case (
            "st"
            | "ts"
            | "sxt"
            | "txs"
            | "temporal-spatial"
            | "temporo-spatial"
            | "temporospatial"
            | "tempospatial"
            | "tempo-spatial"
            | "temporalspatial"
            | "spatiotemporal"
            | "spatio-temporal"
        ):
            modeling_type = "st"
        case _:
            # Keep normalized string for forward compat; blocks._coerce_modeling_types will validate.
            modeling_type = _mt_key

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

    # Canonicalize compile_mode.
    _cm_key = compile_mode.replace("_", "-").replace(" ", "-")
    if "-" in _cm_key:
        _cm_key = "-".join(part for part in _cm_key.split("-") if part)
    _cm_compact = _cm_key.replace("-", "")

    match _cm_key:
        case "" | "none" | "disabled" | "disable" | "off" | "false" | "0":
            compile_mode = "disabled"
        case (
            "default"
            | "reduce-overhead"
            | "max-autotune"
            | "max-autotune-no-cudagraphs"
            | "aot-eager"
        ):
            compile_mode = _cm_key
        case _:
            match _cm_compact:
                case "reduceoverhead":
                    compile_mode = "reduce-overhead"
                case "maxautotune":
                    compile_mode = "max-autotune"
                case "maxautotunenocudagraphs" | "maxautotunenocudagraph":
                    compile_mode = "max-autotune-no-cudagraphs"
                case "aoteager":
                    compile_mode = "aot-eager"
                case _:
                    # Keep normalized string; stnet.core.graph.compile will pass it through.
                    compile_mode = _cm_key or "disabled"

    safety_margin_pow2 = _coerce_int(
        get("safety_margin_pow2", _MODEL_DEFAULTS.safety_margin_pow2),
        name="safety_margin_pow2",
        minimum=0,
        maximum=30,
    )


    p_gate_enabled = _coerce_bool(
        get("p_gate_enabled", _MODEL_DEFAULTS.p_gate_enabled),
        name="p_gate_enabled",
    )
    p_gate_hidden_dim = _coerce_int(
        get("p_gate_hidden_dim", _MODEL_DEFAULTS.p_gate_hidden_dim),
        name="p_gate_hidden_dim",
        minimum=1,
    )
    p_gate_detach_inputs = _coerce_bool(
        get("p_gate_detach_inputs", _MODEL_DEFAULTS.p_gate_detach_inputs),
        name="p_gate_detach_inputs",
    )

    _raw_tile = get("p_gate_tile_size", None)
    p_gate_tile_size = None if _raw_tile is None else _coerce_int(
        _raw_tile,
        name="p_gate_tile_size",
        minimum=1,
    )

    p_gate_bounds_use_quantile = _coerce_bool(
        get("p_gate_bounds_use_quantile", _MODEL_DEFAULTS.p_gate_bounds_use_quantile),
        name="p_gate_bounds_use_quantile",
    )
    p_gate_bounds_q_low = _coerce_float(
        get("p_gate_bounds_q_low", _MODEL_DEFAULTS.p_gate_bounds_q_low),
        name="p_gate_bounds_q_low",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_bounds_q_high = _coerce_float(
        get("p_gate_bounds_q_high", _MODEL_DEFAULTS.p_gate_bounds_q_high),
        name="p_gate_bounds_q_high",
        minimum=0.0,
        maximum=1.0,
    )
    if p_gate_bounds_q_high < p_gate_bounds_q_low:
        p_gate_bounds_q_high = p_gate_bounds_q_low

    p_gate_bounds_q_max_samples = _coerce_int(
        get("p_gate_bounds_q_max_samples", _MODEL_DEFAULTS.p_gate_bounds_q_max_samples),
        name="p_gate_bounds_q_max_samples",
        minimum=0,
    )
    p_gate_bounds_clip_to_minmax = _coerce_bool(
        get("p_gate_bounds_clip_to_minmax", _MODEL_DEFAULTS.p_gate_bounds_clip_to_minmax),
        name="p_gate_bounds_clip_to_minmax",
    )
    p_gate_p_floor = _coerce_float(
        get("p_gate_p_floor", _MODEL_DEFAULTS.p_gate_p_floor),
        name="p_gate_p_floor",
        minimum=0.0,
        maximum=100.0,
    )
    p_gate_p_ceil = _coerce_float(
        get("p_gate_p_ceil", _MODEL_DEFAULTS.p_gate_p_ceil),
        name="p_gate_p_ceil",
        minimum=0.0,
        maximum=100.0,
    )
    if p_gate_p_ceil < p_gate_p_floor:
        p_gate_p_ceil = p_gate_p_floor

    p_gate_fallback_k = _coerce_float(
        get("p_gate_fallback_k", _MODEL_DEFAULTS.p_gate_fallback_k),
        name="p_gate_fallback_k",
    )

    _raw_k_low = get("p_gate_fallback_k_low", None)
    p_gate_fallback_k_low = None if _raw_k_low is None else _coerce_float(
        _raw_k_low,
        name="p_gate_fallback_k_low",
    )
    _raw_k_high = get("p_gate_fallback_k_high", None)
    p_gate_fallback_k_high = None if _raw_k_high is None else _coerce_float(
        _raw_k_high,
        name="p_gate_fallback_k_high",
    )

    p_gate_auto_k_enabled = _coerce_bool(
        get("p_gate_auto_k_enabled", _MODEL_DEFAULTS.p_gate_auto_k_enabled),
        name="p_gate_auto_k_enabled",
    )
    p_gate_auto_k_interval = _coerce_int(
        get("p_gate_auto_k_interval", _MODEL_DEFAULTS.p_gate_auto_k_interval),
        name="p_gate_auto_k_interval",
    )
    p_gate_auto_k_warmup = _coerce_int(
        get("p_gate_auto_k_warmup", _MODEL_DEFAULTS.p_gate_auto_k_warmup),
        name="p_gate_auto_k_warmup",
        minimum=0,
    )
    p_gate_auto_k_ema_alpha = _coerce_float(
        get("p_gate_auto_k_ema_alpha", _MODEL_DEFAULTS.p_gate_auto_k_ema_alpha),
        name="p_gate_auto_k_ema_alpha",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_target_tight = _coerce_float(
        get("p_gate_auto_k_target_tight", _MODEL_DEFAULTS.p_gate_auto_k_target_tight),
        name="p_gate_auto_k_target_tight",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_tolerance = _coerce_float(
        get("p_gate_auto_k_tolerance", _MODEL_DEFAULTS.p_gate_auto_k_tolerance),
        name="p_gate_auto_k_tolerance",
        minimum=0.0,
        maximum=10.0,
    )
    p_gate_auto_k_step_up = _coerce_float(
        get("p_gate_auto_k_step_up", _MODEL_DEFAULTS.p_gate_auto_k_step_up),
        name="p_gate_auto_k_step_up",
        minimum=0.0,
        maximum=10.0,
    )
    p_gate_auto_k_step_down = _coerce_float(
        get("p_gate_auto_k_step_down", _MODEL_DEFAULTS.p_gate_auto_k_step_down),
        name="p_gate_auto_k_step_down",
        minimum=0.0,
        maximum=1.0,
    )

    # Optional per-side step sizes (default to symmetric step_up/step_down).
    p_gate_auto_k_step_up_low = _coerce_float(
        get("p_gate_auto_k_step_up_low", p_gate_auto_k_step_up),
        name="p_gate_auto_k_step_up_low",
        minimum=0.0,
        maximum=10.0,
    )
    p_gate_auto_k_step_down_low = _coerce_float(
        get("p_gate_auto_k_step_down_low", p_gate_auto_k_step_down),
        name="p_gate_auto_k_step_down_low",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_step_up_high = _coerce_float(
        get("p_gate_auto_k_step_up_high", p_gate_auto_k_step_up),
        name="p_gate_auto_k_step_up_high",
        minimum=0.0,
        maximum=10.0,
    )
    p_gate_auto_k_step_down_high = _coerce_float(
        get("p_gate_auto_k_step_down_high", p_gate_auto_k_step_down),
        name="p_gate_auto_k_step_down_high",
        minimum=0.0,
        maximum=1.0,
    )

    p_gate_auto_k_edge_enabled = _coerce_bool(
        get("p_gate_auto_k_edge_enabled", _MODEL_DEFAULTS.p_gate_auto_k_edge_enabled),
        name="p_gate_auto_k_edge_enabled",
    )
    p_gate_auto_k_target_edge = _coerce_float(
        get("p_gate_auto_k_target_edge", _MODEL_DEFAULTS.p_gate_auto_k_target_edge),
        name="p_gate_auto_k_target_edge",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_edge_tolerance = _coerce_float(
        get("p_gate_auto_k_edge_tolerance", _MODEL_DEFAULTS.p_gate_auto_k_edge_tolerance),
        name="p_gate_auto_k_edge_tolerance",
        minimum=0.0,
        maximum=10.0,
    )
    p_gate_auto_k_edge_ema_alpha = _coerce_float(
        get("p_gate_auto_k_edge_ema_alpha", _MODEL_DEFAULTS.p_gate_auto_k_edge_ema_alpha),
        name="p_gate_auto_k_edge_ema_alpha",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_edge_step_down_low = _coerce_float(
        get("p_gate_auto_k_edge_step_down_low", _MODEL_DEFAULTS.p_gate_auto_k_edge_step_down_low),
        name="p_gate_auto_k_edge_step_down_low",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_edge_step_down_high = _coerce_float(
        get("p_gate_auto_k_edge_step_down_high", _MODEL_DEFAULTS.p_gate_auto_k_edge_step_down_high),
        name="p_gate_auto_k_edge_step_down_high",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_min = _coerce_float(
        get("p_gate_auto_k_min", _MODEL_DEFAULTS.p_gate_auto_k_min),
        name="p_gate_auto_k_min",
        minimum=0.0,
    )
    p_gate_auto_k_max = _coerce_float(
        get("p_gate_auto_k_max", _MODEL_DEFAULTS.p_gate_auto_k_max),
        name="p_gate_auto_k_max",
        minimum=0.0,
    )
    if p_gate_auto_k_max < p_gate_auto_k_min:
        p_gate_auto_k_max = p_gate_auto_k_min
    p_gate_auto_k_width_frac = _coerce_float(
        get("p_gate_auto_k_width_frac", _MODEL_DEFAULTS.p_gate_auto_k_width_frac),
        name="p_gate_auto_k_width_frac",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_edge_frac = _coerce_float(
        get("p_gate_auto_k_edge_frac", _MODEL_DEFAULTS.p_gate_auto_k_edge_frac),
        name="p_gate_auto_k_edge_frac",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_auto_k_log_interval = _coerce_int(
        get("p_gate_auto_k_log_interval", _MODEL_DEFAULTS.p_gate_auto_k_log_interval),
        name="p_gate_auto_k_log_interval",
    )

    p_gate_clip_eps = _coerce_float(
        get("p_gate_clip_eps", _MODEL_DEFAULTS.p_gate_clip_eps),
        name="p_gate_clip_eps",
        minimum=0.0,
        maximum=0.49,
    )
    p_gate_eps = _coerce_float(
        get("p_gate_eps", _MODEL_DEFAULTS.p_gate_eps),
        name="p_gate_eps",
        minimum=0.0,
        maximum=1.0,
    )

    p_gate_edge_reg_weight = _coerce_float(
        get("p_gate_edge_reg_weight", _MODEL_DEFAULTS.p_gate_edge_reg_weight),
        name="p_gate_edge_reg_weight",
        minimum=0.0,
    )

    raw_edge_w_low = get("p_gate_edge_reg_weight_low", _MODEL_DEFAULTS.p_gate_edge_reg_weight_low)
    if raw_edge_w_low is None:
        p_gate_edge_reg_weight_low = None
    else:
        p_gate_edge_reg_weight_low = _coerce_float(
            raw_edge_w_low,
            name="p_gate_edge_reg_weight_low",
            minimum=0.0,
        )

    raw_edge_w_high = get("p_gate_edge_reg_weight_high", _MODEL_DEFAULTS.p_gate_edge_reg_weight_high)
    if raw_edge_w_high is None:
        p_gate_edge_reg_weight_high = None
    else:
        p_gate_edge_reg_weight_high = _coerce_float(
            raw_edge_w_high,
            name="p_gate_edge_reg_weight_high",
            minimum=0.0,
        )

    p_gate_edge_reg_fallback_only = _coerce_bool(
        get("p_gate_edge_reg_fallback_only", _MODEL_DEFAULTS.p_gate_edge_reg_fallback_only),
        name="p_gate_edge_reg_fallback_only",
    )
    p_gate_edge_reg_frac = _coerce_float(
        get("p_gate_edge_reg_frac", _MODEL_DEFAULTS.p_gate_edge_reg_frac),
        name="p_gate_edge_reg_frac",
        minimum=0.0,
        maximum=0.49,
    )
    p_gate_edge_reg_min_width_frac = _coerce_float(
        get("p_gate_edge_reg_min_width_frac", _MODEL_DEFAULTS.p_gate_edge_reg_min_width_frac),
        name="p_gate_edge_reg_min_width_frac",
        minimum=0.0,
        maximum=1.0,
    )
    p_gate_edge_reg_power = _coerce_float(
        get("p_gate_edge_reg_power", _MODEL_DEFAULTS.p_gate_edge_reg_power),
        name="p_gate_edge_reg_power",
        minimum=1.0,
        maximum=8.0,
    )

    unsup_xx_weight = _coerce_float(
        get("unsup_xx_weight", _MODEL_DEFAULTS.unsup_xx_weight),
        name="unsup_xx_weight",
        minimum=0.0,
    )
    unsup_yy_weight = _coerce_float(
        get("unsup_yy_weight", _MODEL_DEFAULTS.unsup_yy_weight),
        name="unsup_yy_weight",
        minimum=0.0,
    )

    p_prior_weight = _coerce_float(
        get("p_prior_weight", _MODEL_DEFAULTS.p_prior_weight),
        name="p_prior_weight",
        minimum=0.0,
    )
    p_prior_alpha = _coerce_float(
        get("p_prior_alpha", _MODEL_DEFAULTS.p_prior_alpha),
        name="p_prior_alpha",
        minimum=0.0,
    )
    p_prior_beta = _coerce_float(
        get("p_prior_beta", _MODEL_DEFAULTS.p_prior_beta),
        name="p_prior_beta",
        minimum=0.0,
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
        p_gate_enabled=p_gate_enabled,
        p_gate_hidden_dim=p_gate_hidden_dim,
        p_gate_detach_inputs=p_gate_detach_inputs,
        p_gate_tile_size=p_gate_tile_size,
        p_gate_bounds_use_quantile=p_gate_bounds_use_quantile,
        p_gate_bounds_q_low=p_gate_bounds_q_low,
        p_gate_bounds_q_high=p_gate_bounds_q_high,
        p_gate_bounds_q_max_samples=p_gate_bounds_q_max_samples,
        p_gate_bounds_clip_to_minmax=p_gate_bounds_clip_to_minmax,
        p_gate_p_floor=p_gate_p_floor,
        p_gate_p_ceil=p_gate_p_ceil,
        p_gate_fallback_k=p_gate_fallback_k,
        p_gate_fallback_k_low=p_gate_fallback_k_low,
        p_gate_fallback_k_high=p_gate_fallback_k_high,
        p_gate_auto_k_enabled=p_gate_auto_k_enabled,
        p_gate_auto_k_interval=p_gate_auto_k_interval,
        p_gate_auto_k_warmup=p_gate_auto_k_warmup,
        p_gate_auto_k_ema_alpha=p_gate_auto_k_ema_alpha,
        p_gate_auto_k_target_tight=p_gate_auto_k_target_tight,
        p_gate_auto_k_tolerance=p_gate_auto_k_tolerance,
        p_gate_auto_k_step_up=p_gate_auto_k_step_up,
        p_gate_auto_k_step_down=p_gate_auto_k_step_down,
        p_gate_auto_k_step_up_low=p_gate_auto_k_step_up_low,
        p_gate_auto_k_step_down_low=p_gate_auto_k_step_down_low,
        p_gate_auto_k_step_up_high=p_gate_auto_k_step_up_high,
        p_gate_auto_k_step_down_high=p_gate_auto_k_step_down_high,
        p_gate_auto_k_edge_enabled=p_gate_auto_k_edge_enabled,
        p_gate_auto_k_target_edge=p_gate_auto_k_target_edge,
        p_gate_auto_k_edge_tolerance=p_gate_auto_k_edge_tolerance,
        p_gate_auto_k_edge_ema_alpha=p_gate_auto_k_edge_ema_alpha,
        p_gate_auto_k_edge_step_down_low=p_gate_auto_k_edge_step_down_low,
        p_gate_auto_k_edge_step_down_high=p_gate_auto_k_edge_step_down_high,
        p_gate_auto_k_min=p_gate_auto_k_min,
        p_gate_auto_k_max=p_gate_auto_k_max,
        p_gate_auto_k_width_frac=p_gate_auto_k_width_frac,
        p_gate_auto_k_edge_frac=p_gate_auto_k_edge_frac,
        p_gate_auto_k_log_interval=p_gate_auto_k_log_interval,
        p_gate_clip_eps=p_gate_clip_eps,
        p_gate_eps=p_gate_eps,
        p_gate_edge_reg_weight=p_gate_edge_reg_weight,
        p_gate_edge_reg_weight_low=p_gate_edge_reg_weight_low,
        p_gate_edge_reg_weight_high=p_gate_edge_reg_weight_high,
        p_gate_edge_reg_fallback_only=p_gate_edge_reg_fallback_only,
        p_gate_edge_reg_frac=p_gate_edge_reg_frac,
        p_gate_edge_reg_min_width_frac=p_gate_edge_reg_min_width_frac,
        p_gate_edge_reg_power=p_gate_edge_reg_power,
        unsup_xx_weight=unsup_xx_weight,
        unsup_yy_weight=unsup_yy_weight,
        p_prior_weight=p_prior_weight,
        p_prior_alpha=p_prior_alpha,
        p_prior_beta=p_prior_beta,
    )


def patch_config_to_dict(config: PatchConfig | Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = coerce_patch_config(config)
    data = _shallow_dataclass_dict(cfg)
    # Copy list/dict/set containers one level deep (no recursion).
    return {k: _shallow_copy_if_container(v) for k, v in data.items()}


def model_config_to_dict(config: ModelConfig | Dict[str, Any] | None) -> Dict[str, Any]:
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

            # Canonicalize/validate loss_mask_mode (used by TiledLoss).
            _lm_key = loss_mask_mode.replace("_", "-").replace(" ", "-")
            if "-" in _lm_key:
                _lm_key = "-".join(part for part in _lm_key.split("-") if part)
            _lm_compact = _lm_key.replace("-", "")

            match _lm_key:
                case "" | "none" | "disabled" | "off":
                    loss_mask_mode = "none"
                case "finite" | "isfinite" | "is-finite":
                    loss_mask_mode = "finite"
                case "neq" | "ne" | "not-equal" | "not-eq" | "!=":
                    loss_mask_mode = "neq"
                case _:
                    match _lm_compact:
                        case "isfinite":
                            loss_mask_mode = "finite"
                        case "notequal" | "notneq" | "ne":
                            loss_mask_mode = "neq"
                        case _:
                            raise ValueError(
                                "loss_mask_mode must be one of ('none', 'finite', 'neq'), got "
                                f"{loss_mask_mode!r}"
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
    data: Dict[str, Any] = dict(base or {})
    if kwargs:
        data.update(kwargs)

    actual_mode = data.pop("mode", mode)
    return RuntimeConfig.from_partial(actual_mode, *args, **data)
