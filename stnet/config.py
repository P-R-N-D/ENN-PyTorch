# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import torch

from .utils.datatype import (
    ensure_bool,
    ensure_float,
    ensure_int,
    ensure_int_sequence,
    ensure_int_tuple,
)


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
    enable_compilation: bool = False
    compile_mode: str = "default"
    loss_space: str = "z"
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
    args["is_square"] = ensure_bool(
        filtered.get("is_square", defaults["is_square"]), name="is_square"
    )
    args["patch_size_1d"] = ensure_int(
        filtered.get("patch_size_1d", defaults["patch_size_1d"]),
        name="patch_size_1d",
        minimum=1,
    )
    grid2d = filtered.get("grid_size_2d", defaults["grid_size_2d"])
    args["grid_size_2d"] = (
        ensure_int_tuple(
            grid2d, name="grid_size_2d", dims=2, allow_none=True, keep_scalar=True
        )
        if grid2d is not None
        else None
    )
    patch2d = filtered.get("patch_size_2d", defaults["patch_size_2d"])
    args["patch_size_2d"] = ensure_int_tuple(
        patch2d, name="patch_size_2d", dims=2, keep_scalar=True
    )
    args["is_cube"] = ensure_bool(
        filtered.get("is_cube", defaults["is_cube"]), name="is_cube"
    )
    grid3d = filtered.get("grid_size_3d", defaults["grid_size_3d"])
    args["grid_size_3d"] = (
        ensure_int_tuple(
            grid3d, name="grid_size_3d", dims=3, allow_none=True, keep_scalar=True
        )
        if grid3d is not None
        else None
    )
    patch3d = filtered.get("patch_size_3d", defaults["patch_size_3d"])
    args["patch_size_3d"] = ensure_int_tuple(
        patch3d, name="patch_size_3d", dims=3, keep_scalar=True
    )
    args["dropout"] = ensure_float(
        filtered.get("dropout", defaults["dropout"]),
        name="patch.dropout",
        minimum=0.0,
        maximum=1.0,
    )
    args["use_padding"] = ensure_bool(
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

    args["microbatch"] = ensure_int(
        filtered.get("microbatch", getattr(defaults, "microbatch")),
        name="microbatch",
        minimum=1,
    )
    args["dropout"] = ensure_float(
        filtered.get("dropout", getattr(defaults, "dropout")),
        name="dropout",
        minimum=0.0,
        maximum=1.0,
    )
    args["normalization_method"] = str(
        filtered.get(
            "normalization_method", getattr(defaults, "normalization_method")
        )
    )
    args["depth"] = ensure_int(
        filtered.get("depth", getattr(defaults, "depth")),
        name="depth",
        minimum=1,
    )
    args["heads"] = ensure_int(
        filtered.get("heads", getattr(defaults, "heads")),
        name="heads",
        minimum=1,
    )
    args["spatial_depth"] = ensure_int(
        filtered.get("spatial_depth", getattr(defaults, "spatial_depth")),
        name="spatial_depth",
        minimum=1,
    )
    args["temporal_depth"] = ensure_int(
        filtered.get("temporal_depth", getattr(defaults, "temporal_depth")),
        name="temporal_depth",
        minimum=1,
    )
    args["mlp_ratio"] = ensure_float(
        filtered.get("mlp_ratio", getattr(defaults, "mlp_ratio")),
        name="mlp_ratio",
        minimum=0.0,
    )
    args["drop_path"] = ensure_float(
        filtered.get("drop_path", getattr(defaults, "drop_path")),
        name="drop_path",
        minimum=0.0,
        maximum=1.0,
    )
    args["spatial_latents"] = ensure_int(
        filtered.get("spatial_latents", getattr(defaults, "spatial_latents")),
        name="spatial_latents",
        minimum=1,
    )
    args["temporal_latents"] = ensure_int(
        filtered.get("temporal_latents", getattr(defaults, "temporal_latents")),
        name="temporal_latents",
        minimum=1,
    )
    args["modeling_type"] = str(
        filtered.get("modeling_type", getattr(defaults, "modeling_type"))
    )
    args["use_linear_branch"] = ensure_bool(
        filtered.get("use_linear_branch", getattr(defaults, "use_linear_branch")),
        name="use_linear_branch",
    )
    args["enable_compilation"] = ensure_bool(
        filtered.get(
            "enable_compilation", getattr(defaults, "enable_compilation")
        ),
        name="enable_compilation",
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
    def from_partial(mode: OpsMode, **kw: Any) -> "RuntimeConfig":
        kw = dict(kw)
        for k in ("in_dim", "out_shape", "cfg_dict"):
            if k not in kw or kw[k] is None:
                raise ValueError(f"RuntimeConfig missing required key: {k}")
        in_dim = int(kw["in_dim"])
        out_shape = ensure_int_sequence(kw["out_shape"])
        cfg_dict = dict(kw["cfg_dict"])
        common_keys = {
            "in_dim",
            "out_shape",
            "cfg_dict",
        }
        if mode == "train":
            for k in ("memmap_dir", "ckpt_dir"):
                if k not in kw or kw[k] is None:
                    raise ValueError(f"RuntimeConfig(train) missing required key: {k}")
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
                    "RuntimeConfig(train) received unsupported parameters: "
                    f"{sorted(unsupported)}"
                )
            batch_size = int(kw.get("batch_size", 128))
            return RuntimeConfig(
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
                raise ValueError(f"RuntimeConfig({mode}) missing required key: {k}")
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
                f"RuntimeConfig({mode}) received unsupported parameters: {sorted(unsupported)}"
            )
        batch_size = int(kw.get("batch_size", 512))
        return RuntimeConfig(
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
        data["keys"] = list(data["keys"])
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
