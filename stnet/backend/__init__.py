# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from ..api.config import (
    ModelConfig,
    PatchConfig,
    RuntimeConfig,
    coerce_runtime_config,
    runtime_config,
)
from ..api.io import load_model, new_model, save_model
from ..functional.losses import (
    DataFidelityLoss,
    LinearCombinationLoss,
    MultipleQuantileLoss,
    StandardNormalLoss,
    StudentsTLoss,
    TiledLoss,
)
from ..functional.optimizers import (
    AdamW,
    SWALR,
    StochasticWeightAverage,
    stochastic_weight_average,
)
from .distributed import (
    Join,
    JoinableModel,
    broadcast_model_states,
    coerce_host,
    distributed_barrier,
    format_endpoint_host,
    get_available_addr,
    get_preferred_ip,
    get_world_size,
    initialize_master_addr,
    is_dist_avail_and_initialized,
    is_port_available,
    joining,
    no_synchronization,
    normalize_endpoint,
    normalize_ip_literal,
    probe_stack_support,
    resolve_host_ip,
    sync_model_states,
    wrap_ddp_if_needed,
    wrap_fsdp_module,
)
from .environment import (
    cpu_count,
    cuda_compute_capability,
    default_temp,
    get_device,
    get_runtime_config,
    initialize_python_path,
    initialize_sdpa_backends,
    is_cpu_bf16_supported,
    is_cuda_bf16_supported,
    is_float8_supported,
    is_int4_supported,
    is_int8_supported,
    is_main_loadable,
    new_dir,
    optimal_optimizer_params,
    optimal_procs,
    optimal_start_method,
    optimal_threads,
    optimize_threads,
    set_multiprocessing_env,
)

__all__ = [
    "AdamW",
    "StochasticWeightAverage",
    "SWALR",
    "stochastic_weight_average",
    "DataFidelityLoss",
    "Join",
    "JoinableModel",
    "LinearCombinationLoss",
    "ModelConfig",
    "MultipleQuantileLoss",
    "PatchConfig",
    "RuntimeConfig",
    "StandardNormalLoss",
    "StudentsTLoss",
    "TiledLoss",
    "broadcast_model_states",
    "coerce_host",
    "coerce_runtime_config",
    "cpu_count",
    "cuda_compute_capability",
    "default_temp",
    "distributed_barrier",
    "format_endpoint_host",
    "get_available_addr",
    "get_device",
    "get_preferred_ip",
    "get_runtime_config",
    "get_world_size",
    "infer",
    "initialize_master_addr",
    "initialize_python_path",
    "initialize_sdpa_backends",
    "is_cpu_bf16_supported",
    "is_cuda_bf16_supported",
    "is_dist_avail_and_initialized",
    "is_float8_supported",
    "is_int4_supported",
    "is_int8_supported",
    "is_main_loadable",
    "is_port_available",
    "joining",
    "launch",
    "learn",
    "load_model",
    "new_dir",
    "new_model",
    "no_synchronization",
    "normalize_endpoint",
    "normalize_ip_literal",
    "optimal_optimizer_params",
    "optimal_procs",
    "optimal_start_method",
    "optimal_threads",
    "optimize_threads",
    "predict",
    "probe_stack_support",
    "resolve_host_ip",
    "runtime_config",
    "save_model",
    "set_multiprocessing_env",
    "sync_model_states",
    "train",
    "wrap_ddp_if_needed",
    "wrap_fsdp_module",
]


def __getattr__(name: str) -> Any:
    if name in {"train", "predict", "launch", "learn", "infer"}:
        from ..api.run import launch as _launch, predict as _predict, train as _train

        mapping = {
            "train": _train,
            "predict": _predict,
            "launch": _launch,
            "learn": _train,
            "infer": _predict,
        }
        return mapping[name]
    raise AttributeError(f"module 'stnet.backend' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__))
