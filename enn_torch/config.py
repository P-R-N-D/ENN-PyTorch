# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    Self,
)

import torch

from .core.graph import canonicalize_compile_mode

OpsMode = Literal["train", "predict", "infer"]


def _to_dict(value: Any) -> Optional[Dict[Any, Any] | List[Any] | Set[Any]]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, set)):
        return type(value)(value)
    return value


def _to_dict_strict(obj: Any) -> Dict[Any, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj.__class__)}


def _coerce_str(
    value: Any,
    name: str,
    default: str,
    lower: bool = False,
    strip: bool = True,
) -> str:
    s = (
        str(value).strip()
        if strip and value is not None
        else str(value)
        if value is not None
        else ""
    )
    return (s.lower() if lower else s) or default


def _coerce_bool(value: Any, *args: Any, name: str, **kwargs: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "on", "t"):
        return True
    if s in ("false", "0", "no", "n", "off", "f"):
        return False
    raise TypeError(f"{name} invalid bool")


def _coerce_num(
    val: Any,
    type_: type,
    name: str,
    min: Any = None,
    max: Any = None,
    finite: bool = True,
) -> Any:
    if isinstance(val, bool):
        raise TypeError(f"{name} cannot be bool")
    try:
        v = type_(val)
    except Exception as exc:
        raise TypeError(f"{name} invalid number") from exc
    if finite and not math.isfinite(v):
        raise ValueError(f"{name} not finite")
    if min is not None and v < min:
        raise ValueError(f"{name} < {min}")
    if max is not None and v > max:
        raise ValueError(f"{name} > {max}")
    return v


def _coerce_int(
    value: Any,
    *args: Any,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    **kwargs: Any,
) -> int:
    return _coerce_num(value, int, name, minimum, maximum)


def _coerce_float(
    value: Any,
    *args: Any,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    finite: bool = True,
    **kwargs: Any,
) -> float:
    return _coerce_num(value, float, name, minimum, maximum, finite)


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
    if isinstance(value, int) and not isinstance(value, bool):
        ivalue = _coerce_int(value, name=name, minimum=1)
        return ivalue if keep_scalar else (ivalue,) * dims
    if isinstance(value, (list, tuple)):
        if len(value) != dims:
            raise ValueError(
                f"{name} must have length {dims}, got {len(value)}"
            )
        return tuple(_coerce_num(v, int, name, min=1) for v in value)
    raise TypeError(f"{name} must be an int or sequence of {dims} integers")


def _coerce_int_sequence(
    xs: Sequence[Any],
    *args: Any,
    name: str = "out_shape",
    minimum: Optional[int] = None,
) -> Tuple[int, ...]:
    try:
        return tuple(_coerce_num(x, int, name, min=minimum) for x in xs)
    except TypeError as exc:
        raise TypeError(f"{name} sequence invalid: {xs}") from exc


def _coerce_weights_spec(
    value: Any,
    *args: Any,
    name: str,
) -> Optional[Mapping[str, float] | Sequence[float]]:
    _ = args
    if value is None:
        return None

    def _coerce_one(v: Any, where: str) -> float:
        return _coerce_num(v, float, f"{name}{where}", min=0.0)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [_coerce_num(value, float, f"{name}[0]", min=1e-9)]
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{name} empty map")
        out: Dict[str, float] = {}
        for k, v in dict(value).items():
            ks = str(k)
            if ks in out:
                raise ValueError(f"dup key {ks}")
            out[ks] = _coerce_one(v, f"[{ks}]")
        if not any(v > 0 for v in out.values()):
            raise ValueError(f"{name} needs >0 weight")
        return out
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if not value:
            raise ValueError(f"{name} empty seq")
        out_seq = [_coerce_one(v, f"[{i}]") for i, v in enumerate(value)]
        if not any(v > 0 for v in out_seq):
            raise ValueError(f"{name} needs >0 weight")
        return out_seq
    raise TypeError(f"{name} must be a Mapping[str, float] or Sequence[float]")


def _is_source_spec(obj: Any) -> bool:
    return (
        isinstance(obj, Mapping)
        and isinstance(obj.get("format"), str)
        and isinstance(obj.get("path"), (str, os.PathLike))
    )


def _effective_source_count(sources: Any) -> int:
    if not sources:
        return 0
    if isinstance(sources, (list, tuple)) or (
        isinstance(sources, Mapping) and not _is_source_spec(sources)
    ):
        return len(sources)
    return 1


def _validate_out_shape_dims(out_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if not out_shape or any(d <= 0 for d in out_shape):
        raise ValueError(f"Invalid shape {out_shape}")
    return out_shape


def _coerce_device(
    value: Any, *args: Any, name: str = "device"
) -> Optional[torch.device]:
    if not value or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, torch.device):
        return value
    try:
        return torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            f"{name} has invalid device specification: {value!r}"
        ) from exc


def _to_tuple(
    value: Union[int, Tuple[int, ...]], *args: Any, dims: int, name: str
) -> Tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,) * dims
    if isinstance(value, tuple):
        if len(value) != dims:
            raise ValueError(
                f"{name} must have length {dims}, got {len(value)}"
            )
        return value
    raise TypeError(f"{name} must be int or tuple of length {dims}")


def _validate_equal_dims(
    value: Union[int, Tuple[int, ...]], *args: Any, dims: int, name: str
) -> None:
    t = _to_tuple(value, dims=dims, name=name)
    first = t[0]
    if any(v != first for v in t[1:]):
        raise ValueError(
            f"{name} must have equal dimensions (dims={dims}), got {t}"
        )


def _extract_model_config_dict(model: Any) -> Dict[str, Any]:
    cfg_obj = getattr(model, "config", None) or getattr(
        model, "__enn_instance_config__", None
    )
    if not cfg_obj:
        for submodule in model.modules():
            cfg_obj = getattr(submodule, "config", None) or getattr(
                submodule, "__enn_instance_config__", None
            )
            if cfg_obj:
                break
    if isinstance(cfg_obj, ModelConfig):
        return cfg_obj.to_dict()
    if isinstance(cfg_obj, dict):
        return coerce_model_config(cfg_obj).to_dict()
    return {}


def coerce_patch_config(
    config: PatchConfig | Dict[str, Any] | None,
) -> PatchConfig:
    d = PatchConfig()
    data = (
        _to_dict_strict(config)
        if isinstance(config, PatchConfig)
        else dict(config)
        if isinstance(config, dict)
        else {}
    )
    get = lambda k, def_: data.get(k, def_)
    is_square = _coerce_bool(get("is_square", d.is_square), name="is_square")
    patch_size_1d = _coerce_int(
        get("patch_size_1d", d.patch_size_1d),
        name="patch_size_1d",
        minimum=1,
    )
    grid_size_2d = _coerce_int_tuple(
        get("grid_size_2d", d.grid_size_2d),
        name="grid_size_2d",
        dims=2,
        allow_none=True,
        keep_scalar=True,
    )
    patch_size_2d = _coerce_int_tuple(
        get("patch_size_2d", d.patch_size_2d),
        name="patch_size_2d",
        dims=2,
        keep_scalar=True,
    )
    is_cube = _coerce_bool(get("is_cube", d.is_cube), name="is_cube")
    grid_size_3d = _coerce_int_tuple(
        get("grid_size_3d", d.grid_size_3d),
        name="grid_size_3d",
        dims=3,
        allow_none=True,
        keep_scalar=True,
    )
    patch_size_3d = _coerce_int_tuple(
        get("patch_size_3d", d.patch_size_3d),
        name="patch_size_3d",
        dims=3,
        keep_scalar=True,
    )
    dropout = _coerce_float(
        get("dropout", d.dropout),
        name="patch.dropout",
        minimum=0.0,
        maximum=1.0,
    )
    use_padding = _coerce_bool(
        get("use_padding", d.use_padding), name="use_padding"
    )
    if is_square:
        if patch_size_2d is None:
            raise ValueError(
                "patch_size_2d is required when is_square is True"
            )
        _validate_equal_dims(patch_size_2d, dims=2, name="patch_size_2d")
    if is_cube:
        if patch_size_3d is None:
            raise ValueError("patch_size_3d is required when is_cube is True")
        _validate_equal_dims(patch_size_3d, dims=3, name="patch_size_3d")
    return PatchConfig(
        is_square,
        patch_size_1d,
        grid_size_2d,
        patch_size_2d,
        is_cube,
        grid_size_3d,
        patch_size_3d,
        dropout,
        use_padding,
    )


def coerce_model_config(
    config: ModelConfig | Dict[str, Any] | None,
) -> ModelConfig:
    _MODEL_DEFAULTS = ModelConfig()
    data = (
        _to_dict_strict(config)
        if isinstance(config, ModelConfig)
        else dict(config)
        if isinstance(config, dict)
        else {}
    )
    get = data.get
    patch_cfg = coerce_patch_config(get("patch", _MODEL_DEFAULTS.patch))
    device = _coerce_device(
        get("device", _MODEL_DEFAULTS.device), name="device"
    )
    params: Dict[str, Any] = {}

    def _f(
        key: str,
        min: float = 0.0,
        max: Optional[float] = None,
        allow_none: bool = False,
    ) -> None:
        value = get(key, getattr(_MODEL_DEFAULTS, key))
        if allow_none and value is None:
            params[key] = None
            return
        params[key] = _coerce_float(
            value,
            name=key,
            minimum=min,
            maximum=max,
        )

    def _i(key: str, min: int = 0) -> None:
        params[key] = _coerce_int(
            get(key, getattr(_MODEL_DEFAULTS, key)),
            name=key,
            minimum=min,
        )

    def _b(key: str) -> None:
        params[key] = _coerce_bool(
            get(key, getattr(_MODEL_DEFAULTS, key)),
            name=key,
        )

    _f("dropout", max=1.0)
    _f("mlp_ratio")
    _f("drop_path", max=1.0)
    _i("d_model", 1)
    _i("heads", 1)
    _i("spatial_depth", 1)
    _i("temporal_depth", 1)
    _i("spatial_latents", 1)
    _i("temporal_latents", 1)

    raw_fd = get("fuser_depth", getattr(_MODEL_DEFAULTS, "fuser_depth", None))
    if raw_fd is None:
        params["fuser_depth"] = None
    else:
        fd = _coerce_int(raw_fd, name="fuser_depth", minimum=0)
        params["fuser_depth"] = None if int(fd) <= 0 else int(fd)
    _i("fuser_self_attn_layers", 0)
    stream_task_id_raw = _coerce_str(
        get("stream_task_id", getattr(_MODEL_DEFAULTS, "stream_task_id", "") or ""),
        "stream_task_id",
        getattr(_MODEL_DEFAULTS, "stream_task_id", "") or "",
        lower=False,
    )
    stream_task_id = str(stream_task_id_raw).strip()
    params["stream_task_id"] = stream_task_id if stream_task_id else None
    stream_task_name = _coerce_str(
        get("stream_task_name", getattr(_MODEL_DEFAULTS, "stream_task_name", "") or ""),
        "stream_task_name",
        getattr(_MODEL_DEFAULTS, "stream_task_name", "") or "",
        lower=False,
    )
    params["stream_task_name"] = stream_task_name
    if params.get("stream_task_id") is None:
        alias = str(stream_task_name).strip()
        if alias:
            params["stream_task_id"] = alias

    _b("use_linear_branch")
    _i("safety_margin_pow2", 0)
    _i("delta_gate_hidden_dim", 1)
    _b("delta_gate_detach_inputs")
    _i("delta_gate_bounds_q_max_samples")
    _b("delta_gate_bounds_use_quantile")
    _b("delta_gate_bounds_clip_to_minmax")
    _f("delta_gate_bounds_q_low", max=1.0)
    _f("delta_gate_bounds_q_high", max=1.0)
    if params["delta_gate_bounds_q_high"] < params["delta_gate_bounds_q_low"]:
        params["delta_gate_bounds_q_high"] = params["delta_gate_bounds_q_low"]
    _f("delta_gate_p_floor", max=100.0)
    _f("delta_gate_p_ceil", max=100.0)
    if params["delta_gate_p_ceil"] < params["delta_gate_p_floor"]:
        params["delta_gate_p_ceil"] = params["delta_gate_p_floor"]

    normalization_method = _coerce_str(
        get("normalization_method", _MODEL_DEFAULTS.normalization_method),
        "normalization_method",
        _MODEL_DEFAULTS.normalization_method,
        lower=True,
    )
    norm_key = (
        normalization_method.replace(" ", "").replace("_", "").replace("-", "")
    )
    if norm_key in ("ln", "layernorm"):
        normalization_method = "layernorm"
    elif norm_key in ("bn", "batchnorm"):
        normalization_method = "batchnorm"
    elif norm_key in ("rms", "rmsnorm"):
        normalization_method = "rmsnorm"

    modeling_type = _coerce_str(
        get("modeling_type", _MODEL_DEFAULTS.modeling_type),
        "modeling_type",
        _MODEL_DEFAULTS.modeling_type,
        lower=True,
    )
    mt_key = modeling_type.replace("_", "-").replace(" ", "-")
    if mt_key in ("ss", "spatial", "sxs"):
        modeling_type = "ss"
    elif mt_key in ("tt", "temporal", "txt"):
        modeling_type = "tt"
    elif any(x in mt_key for x in ("st", "ts", "spatial", "temporal")):
        modeling_type = "st"

    compile_mode = canonicalize_compile_mode(
        _coerce_str(
            get("compile_mode", _MODEL_DEFAULTS.compile_mode),
            "compile_mode",
            _MODEL_DEFAULTS.compile_mode,
            lower=True,
        )
    )

    raw_tile_shape = get("delta_gate_tile_shape", None)
    delta_gate_tile_shape = None
    if isinstance(raw_tile_shape, int) and not isinstance(
        raw_tile_shape, bool
    ):
        delta_gate_tile_shape = (raw_tile_shape,)
    elif isinstance(raw_tile_shape, (list, tuple)):
        delta_gate_tile_shape = tuple(
            _coerce_int(v, name="delta_gate_tile_shape", minimum=1)
            for v in raw_tile_shape
        )
    elif isinstance(raw_tile_shape, str):
        parts = [
            p.strip()
            for p in raw_tile_shape.replace("x", ",")
            .replace("X", ",")
            .replace("*", ",")
            .replace(" ", ",")
            .split(",")
            if p.strip()
        ]
        if parts:
            delta_gate_tile_shape = tuple(
                _coerce_int(p, name="delta_gate_tile_shape", minimum=1)
                for p in parts
            )

    _f("delta_gate_fallback_k")
    _f("delta_gate_fallback_k_low", allow_none=True)
    _f("delta_gate_fallback_k_high", allow_none=True)
    _i("delta_gate_auto_k_interval")
    _i("delta_gate_auto_k_warmup")
    _f("delta_gate_auto_k_ema_alpha", max=1.0)
    _f("delta_gate_auto_k_target_tight", max=1.0)
    _f("delta_gate_auto_k_tolerance", max=10.0)
    _f("delta_gate_auto_k_step_up", max=10.0)
    _f("delta_gate_auto_k_step_down", max=1.0)
    for suffix in ("low", "high"):
        _f(f"delta_gate_auto_k_step_up_{suffix}", max=10.0)
        _f(f"delta_gate_auto_k_step_down_{suffix}", max=1.0)
        _f(f"delta_gate_auto_k_edge_step_down_{suffix}", max=1.0)
    _f("delta_gate_auto_k_target_edge", max=1.0)
    _f("delta_gate_auto_k_edge_tolerance", max=10.0)
    _f("delta_gate_auto_k_edge_ema_alpha", max=1.0)
    _f("delta_gate_auto_k_min")
    _f("delta_gate_auto_k_max")
    if params["delta_gate_auto_k_max"] < params["delta_gate_auto_k_min"]:
        params["delta_gate_auto_k_max"] = params["delta_gate_auto_k_min"]

    _f("delta_gate_auto_k_width_frac", max=1.0)
    _f("delta_gate_auto_k_edge_frac", max=1.0)
    _i("delta_gate_auto_k_log_interval")
    _f("delta_gate_clip_eps", max=0.49)
    _f("delta_gate_eps", max=1.0)
    _f("delta_gate_edge_reg_weight")
    _f("delta_gate_edge_reg_weight_low", allow_none=True)
    _f("delta_gate_edge_reg_weight_high", allow_none=True)
    _b("delta_gate_edge_reg_fallback_only")
    _f("delta_gate_edge_reg_frac", max=0.49)
    _f("delta_gate_edge_reg_min_width_frac", max=1.0)
    _f("delta_gate_edge_reg_power", min=1.0, max=8.0)
    _f("delta_gate_budget_weight")
    _f("delta_gate_budget_target")
    _f("delta_gate_tv_weight")
    _f("delta_gate_tv_power", min=1.0, max=8.0)
    _f("delta_gate_teacher_weight")
    _f("delta_gate_teacher_temp", min=1e-8, max=100.0)
    _f("delta_gate_teacher_tau")
    _b("delta_gate_teacher_relu")
    _f("unsup_xx_weight")
    _f("unsup_yy_weight")
    _f("p_prior_weight")
    _f("p_prior_alpha")
    _f("p_prior_beta")

    params.update(
        {
            "device": device,
            "normalization_method": normalization_method,
            "modeling_type": modeling_type,
            "patch": patch_cfg,
            "compile_mode": compile_mode,
            "delta_gate_tile_shape": delta_gate_tile_shape,
            "delta_gate_tile_size": (
                _coerce_int(
                    get("delta_gate_tile_size"), name="delta_gate_tile_size", minimum=1
                )
                if get("delta_gate_tile_size")
                else None
            ),
        }
    )
    return ModelConfig(**params)


def coerce_build_config(
    config: ModelConfig | Dict[str, Any] | None,
) -> ModelConfig:
    return coerce_model_config(config)


def patch_config_to_dict(
    config: PatchConfig | Dict[str, Any] | None,
) -> Dict[str, Any]:
    cfg = coerce_patch_config(config)
    data = _to_dict_strict(cfg)
    return {k: _to_dict(v) for k, v in data.items()}


def model_config_to_dict(
    config: ModelConfig | Dict[str, Any] | None,
) -> Dict[str, Any]:
    if config is None:
        return {}
    cfg = (
        config
        if isinstance(config, ModelConfig)
        else coerce_model_config(config)
    )
    return cfg.to_dict()


def patch_config(
    base: PatchConfig | Dict[str, Any] | None = None, /, **overrides: Any
) -> PatchConfig:
    if base is None:
        data: Dict[str, Any] = {}
    elif isinstance(base, PatchConfig):
        data = _to_dict_strict(base)
    elif isinstance(base, dict):
        data = dict(base)
    else:
        raise TypeError("base must be PatchConfig, dict, or None")
    if overrides:
        data.update(overrides)
    return coerce_patch_config(data)


def model_config(
    base: ModelConfig | Dict[str, Any] | None = None, /, **overrides: Any
) -> ModelConfig:
    if base is None:
        data: Dict[str, Any] = {}
    elif isinstance(base, ModelConfig):
        data = _to_dict_strict(base)
    elif isinstance(base, dict):
        data = dict(base)
    else:
        raise TypeError("base must be ModelConfig, dict, or None")
    if overrides:
        data.update(overrides)
    return coerce_model_config(data)


def build_config(
    base: ModelConfig | Dict[str, Any] | None = None, /, **overrides: Any
) -> ModelConfig:
    return model_config(base, **overrides)


def coerce_runtime_config(
    config: RuntimeConfig | Dict[str, Any],
) -> RuntimeConfig:
    if isinstance(config, RuntimeConfig):
        data: Dict[str, Any] = _to_dict_strict(config)
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
    return RuntimeConfig.from_partial(mode=mode, **data)


def runtime_config(
    mode: OpsMode, base: Dict[str, Any] | None, /, *args: Any, **kwargs: Any
) -> RuntimeConfig:
    data: Dict[str, Any] = dict(base or {})
    if kwargs:
        data.update(kwargs)
    actual_mode = data.pop("mode", mode)
    return RuntimeConfig.from_partial(actual_mode, *args, **data)


@dataclass
class PatchConfig:
    is_square: bool = False
    patch_size_1d: int = 16
    grid_size_2d: Optional[Union[int, Tuple[int, int], list[int]]] = None
    patch_size_2d: Union[int, Tuple[int, int], list[int]] = 4
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
    fuser_depth: Optional[int] = None
    fuser_self_attn_layers: int = 1
    stream_task_id: Optional[str] = None
    stream_task_name: str = ""
    fuser_blend_alpha: float = 0.0
    patch: PatchConfig = field(default_factory=PatchConfig)
    use_linear_branch: bool = False
    compile_mode: str = "disabled"
    safety_margin_pow2: int = 3
    delta_gate_hidden_dim: int = 64
    delta_gate_detach_inputs: bool = True
    delta_gate_tile_size: Optional[int] = None
    delta_gate_tile_shape: Optional[Tuple[int, ...]] = None
    delta_gate_bounds_use_quantile: bool = False
    delta_gate_bounds_q_low: float = 0.005
    delta_gate_bounds_q_high: float = 0.995
    delta_gate_bounds_q_max_samples: int = 8192
    delta_gate_bounds_clip_to_minmax: bool = True
    delta_gate_p_floor: float = 0.0
    delta_gate_p_ceil: float = 1.0
    delta_gate_fallback_k: float = 6.0
    delta_gate_fallback_k_low: Optional[float] = None
    delta_gate_fallback_k_high: Optional[float] = None
    delta_gate_auto_k_interval: int = 100
    delta_gate_auto_k_warmup: int = 100
    delta_gate_auto_k_ema_alpha: float = 0.1
    delta_gate_auto_k_target_tight: float = 0.02
    delta_gate_auto_k_tolerance: float = 0.5
    delta_gate_auto_k_step_up: float = 0.1
    delta_gate_auto_k_step_down: float = 0.02
    delta_gate_auto_k_step_up_low: float = 0.1
    delta_gate_auto_k_step_down_low: float = 0.02
    delta_gate_auto_k_step_up_high: float = 0.1
    delta_gate_auto_k_step_down_high: float = 0.02
    delta_gate_auto_k_target_edge: float = 0.05
    delta_gate_auto_k_edge_tolerance: float = 0.5
    delta_gate_auto_k_edge_ema_alpha: float = 0.1
    delta_gate_auto_k_edge_step_down_low: float = 0.01
    delta_gate_auto_k_edge_step_down_high: float = 0.01
    delta_gate_auto_k_min: float = 1.0
    delta_gate_auto_k_max: float = 16.0
    delta_gate_auto_k_width_frac: float = 0.05
    delta_gate_auto_k_edge_frac: float = 0.02
    delta_gate_auto_k_log_interval: int = 200
    delta_gate_clip_eps: float = 1e-6
    delta_gate_eps: float = 1e-6
    delta_gate_edge_reg_weight: float = 0.0
    delta_gate_edge_reg_weight_low: Optional[float] = None
    delta_gate_edge_reg_weight_high: Optional[float] = None
    delta_gate_edge_reg_fallback_only: bool = False
    delta_gate_edge_reg_frac: float = 0.02
    delta_gate_edge_reg_min_width_frac: float = 0.05
    delta_gate_edge_reg_power: float = 2.0
    delta_gate_budget_weight: float = 0.0
    delta_gate_budget_target: float = 0.5
    delta_gate_tv_weight: float = 0.0
    delta_gate_tv_power: float = 1.0
    delta_gate_teacher_weight: float = 0.0
    delta_gate_teacher_temp: float = 0.25
    delta_gate_teacher_tau: float = 0.0
    delta_gate_teacher_relu: bool = False
    unsup_xx_weight: float = 0.0
    unsup_yy_weight: float = 0.0
    p_prior_weight: float = 0.0
    p_prior_alpha: float = 2.0
    p_prior_beta: float = 2.0

    def to_dict(self: Self) -> Dict[str, Any]:
        data = _to_dict_strict(self)
        data["patch"] = patch_config_to_dict(getattr(self, "patch", None))
        dev = data.get("device")
        if isinstance(dev, torch.device):
            data["device"] = str(dev)
        return {k: _to_dict(v) for k, v in data.items()}


@dataclass
class RuntimeConfig:
    mode: OpsMode
    in_dim: int
    out_shape: Tuple[int, ...]
    cfg_dict: Dict[str, Any]
    sources: Optional[Union[Source, Sequence[Source], Dict[str, Source]]] = (
        None
    )
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
    train_weights: Optional[Mapping[str, float] | Sequence[float]] = None
    val_weights: Optional[Mapping[str, float] | Sequence[float]] = None
    loss_tile_dim: Optional[int] = None
    loss_tile_size: Optional[int] = None
    loss_mask_mode: str = "none"
    loss_mask_value: Optional[float] = None
    model_ckpt_dir: Optional[str] = None
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
        "loss_skew",
    )
    PRED_POS_ORDER: ClassVar[Tuple[str, ...]] = ("seed",)
    _COMMON_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"in_dim", "out_shape", "cfg_dict"}
    )
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
            "train_weights",
            "val_weights",
            "loss_tile_dim",
            "loss_tile_size",
            "loss_mask_mode",
            "loss_mask_value",
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
            "train_weights",
            "val_weights",
        }
    )

    @staticmethod
    def from_partial(
        mode: OpsMode, *args: Any, **kwargs: Any
    ) -> "RuntimeConfig":
        if "mode" in kwargs:
            raise TypeError("No mode in kwargs")
        mode_norm = str(mode).lower()
        if mode_norm not in ("train", "predict", "infer"):
            raise ValueError(f"Invalid mode {mode}")
        order = (
            RuntimeConfig.TRAIN_POS_ORDER
            if mode_norm == "train"
            else RuntimeConfig.PRED_POS_ORDER
        )
        if len(args) > len(order):
            raise TypeError(f"Too many args for {mode_norm}")
        data = dict(kwargs)
        for name, val in zip(order, args):
            if name in data:
                raise TypeError(f"Dup arg {name}")
            data[name] = val
        if "weights" in data:
            data.setdefault("train_weights", data.pop("weights"))
        for k in ("in_dim", "out_shape", "cfg_dict"):
            if k not in data:
                raise ValueError(f"Missing {k}")
        in_dim = _coerce_int(data["in_dim"], name="in_dim", minimum=1)
        out_shape = _validate_out_shape_dims(
            _coerce_int_sequence(
                data["out_shape"], name="out_shape", minimum=1
            )
        )
        cfg_dict = (
            data["cfg_dict"]
            if isinstance(data["cfg_dict"], dict)
            else dict(data["cfg_dict"])
        )

        def _get_val(
            key: str,
            typ: type,
            min: Optional[float] = None,
            max: Optional[float] = None,
            def_: Any = None,
        ) -> Any:
            return _coerce_num(data.get(key, def_), typ, key, min, max)

        if mode_norm == "train":
            for k in ("sources", "ckpt_dir"):
                if k not in data or data[k] is None:
                    raise ValueError(f"Missing {k}")
            if u := (set(data) - RuntimeConfig._TRAIN_KEYS):
                raise ValueError(f"Unsupported {u}")

            loss_mode = (
                _coerce_str(
                    data.get("loss_mask_mode", "none"),
                    "loss_mask_mode",
                    "none",
                    lower=True,
                )
                .replace("_", "")
                .replace("-", "")
            )
            if loss_mode in ("", "none", "disabled", "off"):
                loss_mode = "none"
            elif loss_mode in ("finite", "isfinite"):
                loss_mode = "finite"
            elif loss_mode in ("neq", "ne", "notequal", "noteq", "!="):
                loss_mode = "neq"
            else:
                raise ValueError(f"Invalid loss_mask_mode: {loss_mode}")

            src_n = _effective_source_count(data.get("sources"))
            return RuntimeConfig(
                mode="train",
                in_dim=in_dim,
                out_shape=out_shape,
                cfg_dict=cfg_dict,
                sources=data["sources"],
                ckpt_dir=str(data["ckpt_dir"]),
                init_ckpt_dir=(
                    str(data.get("init_ckpt_dir"))
                    if data.get("init_ckpt_dir")
                    else None
                ),
                epochs=_get_val("epochs", int, 1, def_=5),
                val_frac=_get_val("val_frac", float, 0.0, 1.0, 0.1),
                base_lr=_get_val("base_lr", float, 0.0, def_=1e-3),
                weight_decay=_get_val("weight_decay", float, 0.0, def_=1e-4),
                warmup_ratio=_get_val("warmup_ratio", float, 0.0, 1.0, 0.0),
                eta_min=_get_val("eta_min", float, 0.0, def_=0.0),
                seed=_get_val("seed", int, def_=42),
                shuffle=_coerce_bool(
                    data.get("shuffle", True), name="shuffle"
                ),
                deterministic=_coerce_bool(
                    data.get("deterministic", False), name="deterministic"
                ),
                train_weights=(
                    _coerce_weights_spec(
                        data.get("train_weights"), name="train_weights"
                    )
                    if src_n > 1
                    else None
                ),
                val_weights=(
                    _coerce_weights_spec(
                        data.get("val_weights"), name="val_weights"
                    )
                    if src_n > 1
                    else None
                ),
                loss_tile_dim=(
                    _get_val("loss_tile_dim", int, 1)
                    if data.get("loss_tile_dim")
                    else None
                ),
                loss_tile_size=(
                    _get_val("loss_tile_size", int, 1)
                    if data.get("loss_tile_size")
                    else None
                ),
                loss_mask_mode=loss_mode,
                loss_mask_value=(
                    _get_val("loss_mask_value", float)
                    if data.get("loss_mask_value")
                    else None
                ),
                loss_skew=_coerce_bool(
                    data.get("loss_skew", True), name="loss_skew"
                ),
            )
        for k in ("sources", "ckpt_dir"):
            if k not in data or data[k] is None:
                raise ValueError(f"Missing {k}")
        if u := (set(data) - RuntimeConfig._PRED_KEYS):
            raise ValueError(f"Unsupported {u}")
        src_n = _effective_source_count(data.get("sources"))
        return RuntimeConfig(
            mode="predict" if mode_norm == "predict" else "infer",
            in_dim=in_dim,
            out_shape=out_shape,
            cfg_dict=cfg_dict,
            sources=data["sources"],
            ckpt_dir=str(data["ckpt_dir"]),
            model_ckpt_dir=(
                str(data["model_ckpt_dir"])
                if data.get("model_ckpt_dir")
                else None
            ),
            keys=data.get("keys"),
            seed=_get_val("seed", int, def_=7),
            shuffle=_coerce_bool(data.get("shuffle", False), name="shuffle"),
            loss_skew=_coerce_bool(
                data.get("loss_skew", True), name="loss_skew"
            ),
            train_weights=(
                _coerce_weights_spec(
                    data.get("train_weights"), name="train_weights"
                )
                if src_n > 1
                else None
            ),
            val_weights=(
                _coerce_weights_spec(
                    data.get("val_weights"), name="val_weights"
                )
                if src_n > 1
                else None
            ),
        )


if TYPE_CHECKING:
    from .data.pipeline import Source
else:
    Source = Dict[str, Any]
