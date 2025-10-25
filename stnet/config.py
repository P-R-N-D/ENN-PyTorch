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

from .utils.dtypes import (
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
    loss_space: str = "logit"
    y_low: float = 0.0
    y_high: float = 100.0
    # 경계 여유: 절대/상대 병행(둘 중 큰 값 적용)
    y_eps_range: float = 1e-3
    y_eps_rel: float = 0.02
    auto_y_range: bool = True
    # 경계 타깃 줄이기: 분위수 폭 확대(저속 과소/고속 과대 억제)
    y_range_q_low: float = 0.001
    y_range_q_high: float = 0.999
    # A/B 여유 확장(비율)
    y_range_margin_low: float = 0.10
    y_range_margin_high: float = 0.05
    # 로짓 z 규제(과도 포화 억제)
    z_reg_lambda: float = 1e-3
    # 손실 함수 통계 설정
    loss_std_mode: str = "pooled"
    loss_ddof: int = 1
    loss_detach_stats: bool = True
    loss_clamp_max: float = 6.0
    loss_t_df_start: float = 3.0
    loss_t_df_end: float = 6.0
    loss_t_confidence: float = 0.995
    loss_z_penalty: str = "softmax"
    loss_z_tau: float = 1.5
    # 보조(비대칭) 손실: Quantile(τ>0.5면 과소예측 가중↑)
    aux_q_enable: bool = True
    aux_q_tau_lowspd: float = 0.7
    aux_q_tau_highspd: float = 0.45
    aux_q_weight: float = 0.05
    # 속도 가중(저/고속 불균형 보정)
    loss_low_thr: float = 30.0
    loss_high_thr: float = 90.0
    loss_w_low: float = 1.0
    loss_w_high: float = 1.0
    # 손실 가중(전역/지역/보조)
    w_global: float = 1.0
    w_local: float = 0.2
    w_aux: float = 0.0
    local_decay: bool = True

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
    args["y_low"] = ensure_float(
        filtered.get("y_low", getattr(defaults, "y_low")),
        name="y_low",
    )
    args["y_high"] = ensure_float(
        filtered.get("y_high", getattr(defaults, "y_high")),
        name="y_high",
    )
    if not args["y_high"] > args["y_low"]:
        raise ValueError(
            f"y_high must be greater than y_low (got {args['y_low']}, {args['y_high']})"
        )
    args["y_eps_range"] = ensure_float(
        filtered.get("y_eps_range", getattr(defaults, "y_eps_range")),
        name="y_eps_range",
        minimum=0.0,
    )
    if args["y_eps_range"] <= 0.0:
        raise ValueError("y_eps_range must be positive")
    args["y_eps_rel"] = ensure_float(
        filtered.get("y_eps_rel", getattr(defaults, "y_eps_rel")),
        name="y_eps_rel",
        minimum=0.0,
    )
    args["auto_y_range"] = ensure_bool(
        filtered.get("auto_y_range", getattr(defaults, "auto_y_range")),
        name="auto_y_range",
    )
    args["y_range_q_low"] = ensure_float(
        filtered.get("y_range_q_low", getattr(defaults, "y_range_q_low")),
        name="y_range_q_low",
        minimum=0.0,
        maximum=1.0,
    )
    args["y_range_q_high"] = ensure_float(
        filtered.get("y_range_q_high", getattr(defaults, "y_range_q_high")),
        name="y_range_q_high",
        minimum=0.0,
        maximum=1.0,
    )
    if not args["y_range_q_high"] > args["y_range_q_low"]:
        raise ValueError(
            "y_range_q_high must be greater than y_range_q_low"
        )
    args["y_range_margin_low"] = ensure_float(
        filtered.get("y_range_margin_low", getattr(defaults, "y_range_margin_low")),
        name="y_range_margin_low",
        minimum=0.0,
    )
    args["y_range_margin_high"] = ensure_float(
        filtered.get("y_range_margin_high", getattr(defaults, "y_range_margin_high")),
        name="y_range_margin_high",
        minimum=0.0,
    )
    args["z_reg_lambda"] = ensure_float(
        filtered.get("z_reg_lambda", getattr(defaults, "z_reg_lambda")),
        name="z_reg_lambda",
        minimum=0.0,
    )
    args["loss_std_mode"] = str(
        filtered.get("loss_std_mode", getattr(defaults, "loss_std_mode"))
    ).lower()
    if args["loss_std_mode"] not in {"pooled", "target"}:
        raise ValueError("loss_std_mode must be 'pooled' or 'target'")
    args["loss_ddof"] = ensure_int(
        filtered.get("loss_ddof", getattr(defaults, "loss_ddof")),
        name="loss_ddof",
        minimum=0,
    )
    args["loss_detach_stats"] = ensure_bool(
        filtered.get("loss_detach_stats", getattr(defaults, "loss_detach_stats")),
        name="loss_detach_stats",
    )
    args["loss_clamp_max"] = ensure_float(
        filtered.get("loss_clamp_max", getattr(defaults, "loss_clamp_max")),
        name="loss_clamp_max",
        minimum=0.0,
    )
    args["loss_t_df_start"] = ensure_float(
        filtered.get("loss_t_df_start", getattr(defaults, "loss_t_df_start")),
        name="loss_t_df_start",
        minimum=1e-06,
    )
    args["loss_t_df_end"] = ensure_float(
        filtered.get("loss_t_df_end", getattr(defaults, "loss_t_df_end")),
        name="loss_t_df_end",
        minimum=1e-06,
    )
    if args["loss_t_df_end"] < args["loss_t_df_start"]:
        raise ValueError("loss_t_df_end must be >= loss_t_df_start")
    args["loss_t_confidence"] = ensure_float(
        filtered.get("loss_t_confidence", getattr(defaults, "loss_t_confidence")),
        name="loss_t_confidence",
        minimum=1e-06,
        maximum=0.999999,
    )
    pz = str(filtered.get("loss_z_penalty", getattr(defaults, "loss_z_penalty"))).lower()
    allowed = {"hinge","tau","soft","softplus"}
    if pz not in allowed:
        raise ValueError(f"loss_z_penalty must be one of {sorted(allowed)}")
    args["loss_z_penalty"] = pz
    args["loss_z_tau"] = ensure_float(
        filtered.get("loss_z_tau", getattr(defaults, "loss_z_tau")),
        name="loss_z_tau",
        minimum=0.0,
    )
    args["aux_q_enable"] = ensure_bool(
        filtered.get("aux_q_enable", getattr(defaults, "aux_q_enable")),
        name="aux_q_enable",
    )
    args["aux_q_tau_lowspd"] = ensure_float(
        filtered.get("aux_q_tau_lowspd", getattr(defaults, "aux_q_tau_lowspd")),
        name="aux_q_tau_lowspd",
        minimum=0.0,
        maximum=1.0,
    )
    args["aux_q_tau_highspd"] = ensure_float(
        filtered.get("aux_q_tau_highspd", getattr(defaults, "aux_q_tau_highspd")),
        name="aux_q_tau_highspd",
        minimum=0.0,
        maximum=1.0,
    )
    args["aux_q_weight"] = ensure_float(
        filtered.get("aux_q_weight", getattr(defaults, "aux_q_weight")),
        name="aux_q_weight",
        minimum=0.0,
    )
    args["loss_low_thr"] = ensure_float(
        filtered.get("loss_low_thr", getattr(defaults, "loss_low_thr")),
        name="loss_low_thr",
    )
    args["loss_high_thr"] = ensure_float(
        filtered.get("loss_high_thr", getattr(defaults, "loss_high_thr")),
        name="loss_high_thr",
    )
    if args["loss_high_thr"] < args["loss_low_thr"]:
        raise ValueError("loss_high_thr must be >= loss_low_thr")
    args["loss_w_low"] = ensure_float(
        filtered.get("loss_w_low", getattr(defaults, "loss_w_low")),
        name="loss_w_low",
        minimum=0.0,
    )
    args["loss_w_high"] = ensure_float(
        filtered.get("loss_w_high", getattr(defaults, "loss_w_high")),
        name="loss_w_high",
        minimum=0.0,
    )
    args["w_global"] = ensure_float(
        filtered.get("w_global", getattr(defaults, "w_global")),
        name="w_global",
        minimum=0.0,
    )
    args["w_local"] = ensure_float(
        filtered.get("w_local", getattr(defaults, "w_local")),
        name="w_local",
        minimum=0.0,
    )
    args["w_aux"] = ensure_float(
        filtered.get("w_aux", getattr(defaults, "w_aux")),
        name="w_aux",
        minimum=0.0,
    )
    args["local_decay"] = ensure_bool(
        filtered.get("local_decay", getattr(defaults, "local_decay")),
        name="local_decay",
    )

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
