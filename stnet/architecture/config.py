# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

__all__ = [
    "PatchConfig",
    "ModelConfig",
    "coerce_patch_config",
    "coerce_model_config",
    "patch_config",
    "model_config",
]


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
    microbatch: int = 64
    dropout: float = 0.1
    normalize_method: str = "layernorm"
    depth: int = 128
    heads: int = 4
    spatial_depth: int = 4
    temporal_depth: int = 4
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    spatial_latent_tokens: int = 64
    temporal_latent_tokens: int = 64
    data_definition: str = "spatiotemporal"
    patch: PatchConfig = field(default_factory=PatchConfig)
    use_linear_branch: bool = False
    use_compilation: bool = False
    compile_mode: str = "default"
    loss_space: str = "z"


def _sanitize_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value in {"true", "True", "1"}:
        return True
    if value in {"false", "False", "0"}:
        return False
    raise TypeError(f"{name} must be a boolean-compatible value")


def _sanitize_int(
    value: Any,
    *,
    name: str,
    minimum: Optional[int] = None,
) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer-compatible value") from exc
    if minimum is not None and ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {ivalue}")
    return ivalue


def _sanitize_float(
    value: Any,
    *,
    name: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
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


def _sanitize_tuple_ints(
    value: Any,
    *,
    name: str,
    dims: int,
    allow_none: bool = False,
    keep_scalar: bool = False,
) -> Optional[Union[int, Tuple[int, ...]]]:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} cannot be None")
    if isinstance(value, int):
        ivalue = _sanitize_int(value, name=name, minimum=1)
        if keep_scalar:
            return ivalue  # type: ignore[return-value]
        return tuple([ivalue] * dims)
    if isinstance(value, (list, tuple)):
        if len(value) != dims:
            raise ValueError(f"{name} must have length {dims}, got {len(value)}")
        items = tuple(_sanitize_int(v, name=name, minimum=1) for v in value)
        return items
    raise TypeError(f"{name} must be an int or sequence of {dims} integers")


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
    args: Dict[str, Any] = {}
    args["is_square"] = _sanitize_bool(
        filtered.get("is_square", defaults["is_square"]), name="is_square"
    )
    args["patch_size_1d"] = _sanitize_int(
        filtered.get("patch_size_1d", defaults["patch_size_1d"]),
        name="patch_size_1d",
        minimum=1,
    )
    grid2d = filtered.get("grid_size_2d", defaults["grid_size_2d"])
    args["grid_size_2d"] = (
        _sanitize_tuple_ints(
            grid2d, name="grid_size_2d", dims=2, allow_none=True, keep_scalar=True
        )
        if grid2d is not None
        else None
    )
    patch2d = filtered.get("patch_size_2d", defaults["patch_size_2d"])
    args["patch_size_2d"] = _sanitize_tuple_ints(
        patch2d, name="patch_size_2d", dims=2, keep_scalar=True
    )
    args["is_cube"] = _sanitize_bool(
        filtered.get("is_cube", defaults["is_cube"]), name="is_cube"
    )
    grid3d = filtered.get("grid_size_3d", defaults["grid_size_3d"])
    args["grid_size_3d"] = (
        _sanitize_tuple_ints(
            grid3d, name="grid_size_3d", dims=3, allow_none=True, keep_scalar=True
        )
        if grid3d is not None
        else None
    )
    patch3d = filtered.get("patch_size_3d", defaults["patch_size_3d"])
    args["patch_size_3d"] = _sanitize_tuple_ints(
        patch3d, name="patch_size_3d", dims=3, keep_scalar=True
    )
    args["dropout"] = _sanitize_float(
        filtered.get("dropout", defaults["dropout"]),
        name="patch.dropout",
        minimum=0.0,
        maximum=1.0,
    )
    args["use_padding"] = _sanitize_bool(
        filtered.get("use_padding", defaults["use_padding"]), name="use_padding"
    )
    return PatchConfig(**args)


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

    args: Dict[str, Any] = {"patch": patch_cfg}

    device_val = filtered.get("device", getattr(defaults, "device"))
    if device_val is None or device_val == "":
        args["device"] = None
    else:
        try:
            args["device"] = torch.device(device_val)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid device specification: {device_val}") from exc

    args["microbatch"] = _sanitize_int(
        filtered.get("microbatch", getattr(defaults, "microbatch")),
        name="microbatch",
        minimum=1,
    )
    args["dropout"] = _sanitize_float(
        filtered.get("dropout", getattr(defaults, "dropout")),
        name="dropout",
        minimum=0.0,
        maximum=1.0,
    )
    args["normalize_method"] = str(
        filtered.get("normalize_method", getattr(defaults, "normalize_method"))
    )
    args["depth"] = _sanitize_int(
        filtered.get("depth", getattr(defaults, "depth")),
        name="depth",
        minimum=1,
    )
    args["heads"] = _sanitize_int(
        filtered.get("heads", getattr(defaults, "heads")),
        name="heads",
        minimum=1,
    )
    args["spatial_depth"] = _sanitize_int(
        filtered.get("spatial_depth", getattr(defaults, "spatial_depth")),
        name="spatial_depth",
        minimum=1,
    )
    args["temporal_depth"] = _sanitize_int(
        filtered.get("temporal_depth", getattr(defaults, "temporal_depth")),
        name="temporal_depth",
        minimum=1,
    )
    args["mlp_ratio"] = _sanitize_float(
        filtered.get("mlp_ratio", getattr(defaults, "mlp_ratio")),
        name="mlp_ratio",
        minimum=0.0,
    )
    args["drop_path"] = _sanitize_float(
        filtered.get("drop_path", getattr(defaults, "drop_path")),
        name="drop_path",
        minimum=0.0,
        maximum=1.0,
    )
    args["spatial_latent_tokens"] = _sanitize_int(
        filtered.get(
            "spatial_latent_tokens", getattr(defaults, "spatial_latent_tokens")
        ),
        name="spatial_latent_tokens",
        minimum=1,
    )
    args["temporal_latent_tokens"] = _sanitize_int(
        filtered.get(
            "temporal_latent_tokens", getattr(defaults, "temporal_latent_tokens")
        ),
        name="temporal_latent_tokens",
        minimum=1,
    )
    args["data_definition"] = str(
        filtered.get("data_definition", getattr(defaults, "data_definition"))
    )
    args["use_linear_branch"] = _sanitize_bool(
        filtered.get("use_linear_branch", getattr(defaults, "use_linear_branch")),
        name="use_linear_branch",
    )
    args["use_compilation"] = _sanitize_bool(
        filtered.get("use_compilation", getattr(defaults, "use_compilation")),
        name="use_compilation",
    )
    args["compile_mode"] = str(
        filtered.get("compile_mode", getattr(defaults, "compile_mode"))
    )
    args["loss_space"] = str(
        filtered.get("loss_space", getattr(defaults, "loss_space"))
    ).lower()

    return ModelConfig(**args)


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

