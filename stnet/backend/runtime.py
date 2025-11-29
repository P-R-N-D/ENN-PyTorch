# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from dataclasses import replace
from functools import partial
import json
import logging
import math
import os
import re
import platform
import sys
import threading
import time
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from tensordict import TensorDictBase
import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor import DTensor, Placement, Replicate
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from tqdm.auto import tqdm
try:
    import pynvml as _nvml

    try:
        _nvml.nvmlInit()
        _NVML_READY = True
    except Exception:
        _NVML_READY = False
except Exception:
    _nvml = None
    _NVML_READY = False
try:
    import psutil as _psutil
except Exception:
    _psutil = None

from ..api.config import RuntimeConfig, coerce_model_config
from ..data.datatype import to_tensordict, to_torch_tensor
from ..data.stats import Metadata
from ..data.transforms import postprocess, preprocess
from ..functional.fx import Autocast, Fusion, Gradient
from ..functional.losses import (
    LossWeightController,
    LinearCombinationLoss,
    CRPSLoss,
    DataFidelityLoss,
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
from ..model.layers import History, Instance, resize_scaler_buffer
from .compat import (
    cudagraph_step_end,
    is_meta_or_fake_tensor,
    torch_no_compile,
    torch_compile_safe,
    torch_safe_distributed,
)
from .distributed import (
    distributed_barrier,
    distributed_sync,
    get_world_size,
    is_distributed,
    joining,
    no_sync,
    to_ddp,
    to_fsdp,
)
from .profiler import FlopCounter
from .system import Memory, get_device, get_tlb, initialize_python_path, is_float8_supported, new_dir, posix_time

if TYPE_CHECKING:
    import numpy as _np
    from numpy.typing import NDArray as _NDArray

    Float64Array = _NDArray[_np.float64]
else:
    Float64Array = Any

_LOGGER = logging.getLogger(__name__)

torch_safe_distributed()


try:
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
except ImportError:

    def precompute_float8_dynamic_scale_for_fsdp(*args: Any, **kwargs: Any) -> Any:
        return None


ignored_sentences = [
    "External init callback must run in same thread as registerClient",
    "Initializing zero-element tensors is a no-op",
    "gpuGetDeviceCount failed with code",
    "torch.distributed is disabled",
    "TypedStorage is deprecated",
    "flex_attention called without torch.compile",
    "SOLUTION: Use torch.compile",
    "Not enough SMs to use max_autotune_gemm mode",
]
ignored_pattern = "|".join([re.escape(s) for s in ignored_sentences])
_DL_STATE_FILE = "dataloader.json"


def _num_batches(loader: Any) -> int:
    if loader is None:
        return 0
    try:
        n = len(loader)
        if isinstance(n, int) and n >= 0:
            return n
    except Exception:
        pass
    if hasattr(loader, "state_dict") and hasattr(loader, "load_state_dict"):
        state = None
        with contextlib.suppress(Exception):
            state = loader.state_dict()
        if state is not None:
            count = 0
            try:
                for _ in loader:
                    count += 1
            finally:
                with contextlib.suppress(Exception):
                    loader.load_state_dict(state)
            return count
    return 0


def _float8_log(
    msg: str, *args: Any, only_main_rank: bool = True, **kwargs: Any
) -> None:
    try:
        if only_main_rank and torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
    except Exception:
        pass
    _LOGGER.info(msg, *args)


def _assert_no_meta_tensors(module: torch.nn.Module) -> None:
    hits: list[str] = []
    for name, param in module.named_parameters(recurse=True):
        if is_meta_or_fake_tensor(param):
            hits.append(f"param {name} shape={tuple(param.shape)}")
    for name, buffer in module.named_buffers(recurse=True):
        if is_meta_or_fake_tensor(buffer):
            hits.append(f"buffer {name} shape={tuple(buffer.shape)}")
    if hits:
        raise RuntimeError("Found meta tensors in model:\n" + "\n".join(hits))


def _meta_monitor_pre_hook(
    module: torch.nn.Module, inputs: Tuple[Any, ...], warn_only: bool
) -> None:
    for arg in inputs:
        if isinstance(arg, torch.Tensor) and is_meta_or_fake_tensor(arg):
            message = f"[META] {module.__class__.__name__} got meta input"
            if warn_only:
                warnings.warn(message, stacklevel=3)
                return
            raise RuntimeError(message)


def _enable_meta_monitor(model: torch.nn.Module) -> None:
    hook_mode = os.environ.get("STNET_META_MONITOR", "off").strip().lower()
    if hook_mode in {"0", "", "false", "off"}:
        return
    warn_only = hook_mode in {"warn", "warning"}
    for submodule in model.modules():
        submodule.register_forward_pre_hook(
            partial(_meta_monitor_pre_hook, warn_only=warn_only), with_kwargs=False
        )


def _assert_no_fake_dtensor(
    root: nn.Module, *args: Any, allow_dtensor: bool = False, **kwargs: Any
) -> None:
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except Exception:
        dtensor_types: Tuple[type, ...] = tuple()
    else:
        dtensor_types = tuple() if allow_dtensor else (_DTensor,)
    bad: list[str] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        for attr in ("weight", "bias"):
            tensor = getattr(module, attr, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            data_attr = getattr(tensor, "data", None)
            is_meta_or_fake = is_meta_or_fake_tensor(tensor)
            is_dtensor = isinstance(tensor, dtensor_types) or isinstance(
                data_attr, dtensor_types
            )
            if is_meta_or_fake or is_dtensor:
                module_name = name or module.__class__.__name__
                bad.append(f"{module_name}.{attr}{tuple(tensor.shape)}")
    if bad:
        raise RuntimeError(
            "LayerNorm params must be real/local tensors:\n  " + "\n  ".join(bad)
        )


def _reset_layernorm_parameter(
    module: nn.LayerNorm, name: str, data: torch.Tensor, *, requires_grad: bool
) -> None:
    setattr(module, name, nn.Parameter(data, requires_grad=requires_grad))


def _preload_layers(model: torch.nn.Module, device: torch.device) -> None:
    for module in model.modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        requires_grad_w = bool(getattr(weight, "requires_grad", True))
        requires_grad_b = bool(getattr(bias, "requires_grad", True))
        if device.type == "cpu":
            target_dtype = torch.float32
        else:
            target_dtype = None
            for tensor in (weight, bias):
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    if not is_meta_or_fake_tensor(tensor):
                        target_dtype = tensor.dtype
                        break
            if target_dtype is None:
                target_dtype = torch.get_default_dtype()
        if module.elementwise_affine:
            if (
                not isinstance(weight, torch.Tensor)
                or is_meta_or_fake_tensor(weight)
                or isinstance(weight, DTensor)
                or isinstance(getattr(weight, "data", None), DTensor)
            ):
                data = torch.ones(
                    module.normalized_shape, device=device, dtype=target_dtype
                )
                _reset_layernorm_parameter(
                    module, "weight", data, requires_grad=requires_grad_w
                )
                weight = module.weight
            if (
                not isinstance(bias, torch.Tensor)
                or is_meta_or_fake_tensor(bias)
                or isinstance(bias, DTensor)
                or isinstance(getattr(bias, "data", None), DTensor)
            ):
                data = torch.zeros(
                    module.normalized_shape, device=device, dtype=target_dtype
                )
                _reset_layernorm_parameter(
                    module, "bias", data, requires_grad=requires_grad_b
                )
                bias = module.bias
        if device.type == "cpu":
            if isinstance(weight, torch.Tensor) and weight.dtype != torch.float32:
                data = weight.to(device=device, dtype=torch.float32)
                _reset_layernorm_parameter(
                    module, "weight", data, requires_grad=requires_grad_w
                )
                weight = module.weight
            if isinstance(bias, torch.Tensor) and bias.dtype != torch.float32:
                data = bias.to(device=device, dtype=torch.float32)
                _reset_layernorm_parameter(
                    module, "bias", data, requires_grad=requires_grad_b
                )
                bias = module.bias
        else:
            if (
                isinstance(weight, torch.Tensor)
                and isinstance(bias, torch.Tensor)
                and weight.is_floating_point()
                and bias.is_floating_point()
                and bias.dtype != weight.dtype
            ):
                data = bias.to(device=device, dtype=weight.dtype)
                _reset_layernorm_parameter(
                    module, "bias", data, requires_grad=requires_grad_b
                )
                bias = module.bias


def _assert_unified_layer_dtype(model: torch.nn.Module, device: torch.device) -> None:
    mismatches: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        tensors = [
            ("weight", getattr(module, "weight", None)),
            ("bias", getattr(module, "bias", None)),
        ]
        expected: Optional[torch.dtype]
        if device.type == "cpu":
            expected = torch.float32
        else:
            expected = None
        for label, tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                continue
            if expected is None:
                expected = tensor.dtype
            elif tensor.dtype != expected:
                module_name = name or module.__class__.__name__
                mismatches.append(
                    f"{module_name}.{label} has dtype {tensor.dtype} (expected {expected})"
                )
        if expected is not None and device.type != "cpu":
            dtypes = {
                tensor.dtype
                for _, tensor in tensors
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
            }
            if len(dtypes) > 1:
                module_name = name or module.__class__.__name__
                mismatches.append(
                    f"{module_name} parameters disagree on dtype: {sorted(dtypes)}"
                )
    if mismatches:
        raise RuntimeError(
            "LayerNorm parameter dtype mismatch detected:\n" + "\n".join(mismatches)
        )


def _trim_dcp_keys(state: Any) -> Any:
    if isinstance(state, dict):
        keys = []
        for key, value in list(state.items()):
            key_str = str(key)
            if (
                key_str.endswith("._extra_state")
                or key_str.endswith("_extra_state")
                or key_str.endswith("output_baked_flag")
            ):
                keys.append(key)
                continue
            state[key] = _trim_dcp_keys(value)
        for key in keys:
            state.pop(key, None)
    return state


def _backend_type(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    if device.type == "xpu":
        return "xccl"
    return "gloo"


def _set_backend(device: torch.device) -> None:
    with contextlib.suppress(Exception):
        with contextlib.suppress(Exception):
            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.backends.fp32_precision = "tf32"
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.fp32_precision = "tf32"
                if hasattr(torch.backends, "cudnn"):
                    with contextlib.suppress(Exception):
                        torch.backends.cudnn.fp32_precision = "tf32"
                    if hasattr(torch.backends.cudnn, "conv"):
                        torch.backends.cudnn.conv.fp32_precision = "tf32"
                    if hasattr(torch.backends.cudnn, "rnn"):
                        torch.backends.cudnn.rnn.fp32_precision = "tf32"
            else:
                with contextlib.suppress(Exception):
                    torch.backends.fp32_precision = "ieee"
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.fp32_precision = "ieee"
                if hasattr(torch.backends, "cudnn"):
                    with contextlib.suppress(Exception):
                        torch.backends.cudnn.fp32_precision = "ieee"
                    if hasattr(torch.backends.cudnn, "conv"):
                        torch.backends.cudnn.conv.fp32_precision = "ieee"
                    if hasattr(torch.backends.cudnn, "rnn"):
                        torch.backends.cudnn.rnn.fp32_precision = "ieee"
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = True
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if device.type == "cuda":
        torch.cuda.set_device(rank)
    elif device.type == "xpu":
        torch.xpu.set_device(rank)
    else:
        try:
            import netifaces

            gws = netifaces.gateways()
            iface: str | None = None
            default_gateways = gws.get("default", {}) if isinstance(gws, dict) else {}
            families = []
            with contextlib.suppress(AttributeError):
                families.append(netifaces.AF_INET6)
            families.append(netifaces.AF_INET)
            for family in families:
                info = default_gateways.get(family)
                if info and len(info) >= 2:
                    iface = info[1]
                    if iface:
                        break
            if iface:
                os.environ.setdefault("GLOO_SOCKET_IFNAME", iface)
                os.environ.setdefault("TP_SOCKET_IFNAME", iface)
        except (ImportError, KeyError, OSError):
            pass


def _from_meta(memmap_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(memmap_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unify_param_dtype(
    model: Any, prefer: Optional[torch.dtype] = None
) -> Optional[torch.dtype]:
    dtypes = set((p.dtype for p in model.parameters() if p is not None))
    if len(dtypes) <= 1:
        return None
    if prefer is not None:
        tgt = prefer
    elif torch.bfloat16 in dtypes:
        tgt = torch.bfloat16
    elif torch.float16 in dtypes:
        tgt = torch.float16
    else:
        tgt = torch.float32
    for mod in model.modules():
        params = getattr(mod, "_parameters", None)
        if not params:
            continue
        for name, p in list(params.items()):
            if p is None or p.dtype == tgt:
                continue
            new_p = torch.nn.Parameter(
                p.detach().to(tgt), requires_grad=p.requires_grad
            )
            setattr(mod, name, new_p)
    return tgt


def _first_source_path(obj: Any) -> str:
    if isinstance(obj, dict):
        if "path" in obj and "kind" in obj:
            return os.fspath(obj["path"])
        if obj:
            first = next(iter(obj.values()))
            return _first_source_path(first)
    if isinstance(obj, (list, tuple)) and obj:
        return _first_source_path(obj[0])
    raise RuntimeError("sources is empty or invalid")


def _expand(sources: Any) -> Any:
    def _expand_from_root(spec: Any) -> Tuple[Any, bool]:
        if not isinstance(spec, dict) or "path" not in spec or "kind" not in spec:
            return spec, False
        root = os.fspath(spec.get("path") or "")
        mn_path = os.path.join(root, "multinode.json")
        if not os.path.isfile(mn_path):
            return spec, False
        with open(mn_path, "r", encoding="utf-8") as _f:
            _spec = json.load(_f)
        if isinstance(_spec, dict):
            resolved = {
                str(k): {"kind": "memmap", "path": os.path.join(root, str(v))}
                for k, v in _spec.items()
            }
            return resolved, True
        if isinstance(_spec, list):
            resolved = [
                {"kind": "memmap", "path": os.path.join(root, str(v))} for v in _spec
            ]
            return resolved, True
        return spec, False

    expanded, ok = _expand_from_root(sources)
    if ok:
        return expanded
    if isinstance(sources, (list, tuple)) and len(sources) == 1:
        expanded, ok = _expand_from_root(sources[0])
        if ok:
            return expanded
    return sources


@torch.no_grad()
def _calibrate_per_sample_mem(
    model: Instance,
    device: torch.device,
    ops: RuntimeConfig,
    max_probe_batch: int = 32,
) -> None:
    from ..data.nodes import Dataset

    dev_type = getattr(device, "type", "")
    if dev_type != "cuda":
        return

    try:
        memmap_root = _first_source_path(ops.sources)
        ds = Dataset(memmap_root, split="train", val_frac=float(getattr(ops, "val_frac", 0.0) or 0.0))
    except Exception:
        return

    try:
        N = int(len(ds))
    except Exception:
        N = 0
    if N <= 0:
        return

    B0 = max(1, min(int(max_probe_batch), N))
    try:
        batch = ds.get(0, B0)
    except Exception:
        return

    try:
        feats, labels, keys, label_shape = preprocess(batch)
    except Exception:
        return

    X = to_torch_tensor(feats)
    X = torch.atleast_2d(X)
    if X.dim() != 2 or X.shape[1] != int(ops.in_dim):
        return

    X = X.to(device, non_blocking=True)

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass

    with Gradient.inference(model), Autocast.float(device):
        td = to_tensordict({"features": X})
        _ = model(
            td,
            global_loss=None,
            local_loss=None,
            loss_weights=None,
            calibrate_output=True,
        )
    try:
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return

    if peak_bytes <= 0:
        return

    per_sample = max(1, peak_bytes // B0)
    try:
        Dataset._per_sample_mem_bytes = int(per_sample)
    except Exception:
        pass


def _coerce_dtensor(
    param: torch.nn.Parameter,
    mesh: Any,
    *args: Any,
    placements: Optional[Sequence[Placement]] = None,
    **kwargs: Any,
) -> None:
    if not isinstance(param, torch.nn.Parameter):
        return
    current_dtensor: Optional[DTensor]
    current_dtensor = param.data if isinstance(param.data, DTensor) else None
    placements_tuple: Optional[Tuple[Placement, ...]]
    placements_tuple = tuple(placements) if placements is not None else None
    if current_dtensor is None:
        if (
            getattr(param, "_is_sharded", False)
            or hasattr(param, "_sharding_spec")
            or param.__class__.__name__ == "FlatParameter"
        ):
            return
        placements_tuple = placements_tuple or (Replicate(),)
        try:
            current_dtensor = DTensor.from_local(
                param.data, mesh, placements_tuple, run_check=False
            )
        except Exception:
            return
        param.data = current_dtensor
    else:
        if (
            placements_tuple is not None
            and tuple(current_dtensor.placements) != placements_tuple
        ):
            try:
                current_dtensor = current_dtensor.redistribute(
                    device_mesh=current_dtensor.device_mesh, placements=placements_tuple
                )
                param.data = current_dtensor
            except Exception:
                placements_tuple = tuple(current_dtensor.placements)
    if placements_tuple is None and current_dtensor is not None:
        placements_tuple = tuple(current_dtensor.placements)
    grad = param.grad
    if grad is not None and not isinstance(grad, DTensor):
        target_mesh = mesh if current_dtensor is None else current_dtensor.device_mesh
        try:
            param.grad = DTensor.from_local(
                grad, target_mesh, placements_tuple or (Replicate(),), run_check=False
            )
        except Exception:
            param.grad = None


def _wrap_fsdp(
    target: Optional[torch.nn.Module],
    mesh: Any,
    mp_policy: MixedPrecisionPolicy,
    wrapped: set[int],
    ignored_param_registry: "_IdentityParamSet",
) -> Optional[torch.nn.Module]:
    if target is None or id(target) in wrapped:
        return target
    wrapped.add(id(target))
    per_mod_ignored = _ignored_params(target, ignored_param_registry)
    return to_fsdp(
        target,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=False,
        sync_module_states=True,
        ignored_params=per_mod_ignored or None,
    )


def _get_layers(root: Optional[torch.nn.Module]) -> List[torch.nn.Module]:
    if root is None:
        return []
    blocks: List[torch.nn.Module] = []
    seen: set[int] = set()
    for module in root.modules():
        block_list = getattr(module, "blocks", None)
        if isinstance(block_list, torch.nn.ModuleList):
            for block in block_list:
                if isinstance(block, torch.nn.Module) and id(block) not in seen:
                    seen.add(id(block))
                    blocks.append(block)
    return blocks


def _initialize_tensor(
    value: Any,
    *args: Any,
    param: torch.Tensor,
    capturable: bool,
    fused: bool,
    **kwargs: Any,
) -> torch.Tensor:
    desired_device = param.device if (capturable or fused) else torch.device("cpu")
    desired_dtype = param.dtype if torch.is_floating_point(param) else torch.float32
    if isinstance(value, torch.Tensor):
        step_tensor = value.detach()
        if step_tensor.ndim != 0:
            step_tensor = step_tensor.reshape(())
        if step_tensor.device != desired_device:
            step_tensor = step_tensor.to(desired_device)
        if step_tensor.dtype != desired_dtype:
            step_tensor = step_tensor.to(desired_dtype)
    else:
        base = float(value) if value is not None else 0.0
        step_tensor = torch.tensor(base, dtype=desired_dtype, device=desired_device)
    return step_tensor


def _initialize_adamw(optim: torch.optim.Optimizer) -> None:
    for group in optim.param_groups:
        amsgrad = group.get("amsgrad", False)
        capturable = bool(group.get("capturable", False))
        fused = bool(group.get("fused", False))
        for param in group.get("params", []):
            if not getattr(param, "requires_grad", False):
                continue
            state = optim.state.get(param)
            state = {} if state is None else state
            step_value = state.get("step")
            state["step"] = _initialize_tensor(
                step_value, param=param, capturable=capturable, fused=fused
            )
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(param)
            if amsgrad and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(param)
            optim.state[param] = state


def _scheduler(
    step: int,
    *args: Any,
    warmup_steps: int,
    start_factor: float,
    base: float,
    main_steps: int,
    emin: float,
    **kwargs: Any,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps))
    t = step - warmup_steps
    frac_min = emin / base if base > 0.0 else 0.0
    return frac_min + (1.0 - frac_min) * 0.5 * (
        1.0 + math.cos(math.pi * t / max(1, main_steps))
    )


def _initialize_group(backend: str, device: torch.device, local_rank: int) -> None:
    dev_id: Optional[Union[int, torch.device]] = None
    dev_type = getattr(device, "type", "cpu")
    if dev_type in ("cuda", "xpu", "mps"):
        index = (
            device.index
            if getattr(device, "index", None) is not None
            else int(os.environ.get("LOCAL_RANK", local_rank))
        )
        try:
            dev_id = torch.device(dev_type, index)
        except Exception:
            dev_id = index
    try:
        if dev_id is not None:
            torch.distributed.init_process_group(backend=backend, device_id=dev_id)
        else:
            torch.distributed.init_process_group(backend=backend)
    except TypeError:
        torch.distributed.init_process_group(backend=backend)


def _ignored_params(
    module: torch.nn.Module, registry: "_IdentityParamSet"
) -> Optional["_IdentityParamSet"]:
    if len(registry) == 0:
        return None
    params = [param for param in module.parameters(recurse=True) if param in registry]
    return _IdentityParamSet(tuple(params)) if params else None


class _IdentityParamSet(Sequence[torch.nn.Parameter]):
    def __init__(self, params: Sequence[torch.nn.Parameter]) -> None:
        self._params = tuple(params)
        self._ids = {id(p) for p in self._params}

    def __len__(self) -> int:
        return len(self._params)

    def __iter__(self) -> Iterator[torch.nn.Parameter]:
        return iter(self._params)

    def __getitem__(self, index: int) -> torch.nn.Parameter:
        return self._params[index]

    def __contains__(self, item: object) -> bool:
        return isinstance(item, torch.nn.Parameter) and (id(item) in self._ids)


def loader_state_path(directory: str) -> str:
    return os.path.join(directory, _DL_STATE_FILE)


def get_tqdm(
    *args: Any, title: str, total: int, device: torch.device, **kwargs: Any
) -> Optional[tqdm]:
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return None
    except Exception:
        pass
    if int(total) <= 0:
        return None
    bar = tqdm(
        total=int(total),
        desc=f"{title} ({device.type.upper()})",
        unit="I/O < 0.01 MB/s, COM < 0.01 TFLOPS",
        bar_format="{desc}"
        + "{bar} {percentage:3.0f}% "
        + "({unit}) Elapsed: {elapsed}, Remaining: {remaining}",
        colour="green",
        position=0,
        leave=False,
        file=sys.stdout,
    )
    return bar


def _gpu_nvml_utils(device: torch.device) -> Tuple[Optional[float], Optional[float]]:
    if not _NVML_READY or getattr(device, "type", "") != "cuda":
        return None, None
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        h = _nvml.nvmlDeviceGetHandleByIndex(int(idx))
        u = _nvml.nvmlDeviceGetUtilizationRates(h)
        mi = _nvml.nvmlDeviceGetMemoryInfo(h)
        gpu_util = float(getattr(u, "gpu", 0.0))
        mem_util = (
            (100.0 * float(mi.used) / float(mi.total)) if getattr(mi, "total", 0) else None
        )
        return gpu_util, mem_util
    except Exception:
        return None, None


def _cpu_percent_now() -> Optional[float]:
    if _psutil is None:
        return None
    try:
        return float(_psutil.cpu_percent(interval=0.0))
    except Exception:
        return None


def update_tqdm(
    bar: Optional[tqdm],
    finish: int,
    *args: Any,
    mbps: Optional[float] = None,
    tflops: Optional[float] = None,
    **kwargs: Any,
) -> None:
    if bar is None:
        return
    try:
        mbps_val = float(mbps) if mbps is not None else 0.0
    except Exception:
        mbps_val = 0.0
    try:
        tflops_val = float(tflops) if tflops is not None else 0.0
    except Exception:
        tflops_val = 0.0
    io_expr = f"I/O = {mbps_val:.2f} MB/s" if mbps_val >= 0.01 else "I/O < 0.01 MB/s"
    com_expr = (
        f"COM = {tflops_val:.2f} TFLOPS" if tflops_val >= 0.01 else "COM < 0.01 TFLOPS"
    )
    bar.unit = ", ".join([io_expr, com_expr])
    try:
        inc = int(finish)
    except Exception:
        inc = 1
    if inc > 0:
        bar.update(inc)


def epochs(
    model: Instance,
    device: torch.device,
    local_rank: int,
    ops: RuntimeConfig,
    *args: Any,
    param_dtype: torch.dtype,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: torch.optim.lr_scheduler.LRScheduler,
    loss_controller: LossWeightController,
    top_loss: nn.Module,
    bottom_loss: TiledLoss,
    train_loader: Any,
    val_loader: Any,
    total_epochs: int,
    scheduler_step_per_batch: bool = True,
    swa_helper: Optional[StochasticWeightAverage] = None,
    swa_start_epoch: int = 0,
    buffers_dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> None:
    from ..data.nodes import Dataset

    if train_loader is None:
        raise RuntimeError("epochs requires a training dataloader")

    per_batch = getattr(train_loader, "batch_size", None)
    if per_batch is None or int(per_batch) <= 0:
        try:
            sample = next(iter(train_loader))
            if isinstance(sample, (list, tuple)) and sample:
                feats = sample[0]
            elif isinstance(sample, dict) and sample:
                feats = next(iter(sample.values()))
            else:
                feats = sample
            per_batch = int(getattr(feats, "shape", [len(feats)])[0])
        except Exception:
            per_batch = 1

    if per_batch <= 0:
        per_batch = 1

    factor = 2
    with contextlib.suppress(Exception):
        avail_bytes = Memory.available()
        if avail_bytes and avail_bytes > 0:
            avail_gb = avail_bytes / float(1024 ** 3)
            est = int(max(1, min(8, avail_gb / 4.0)))
            factor = max(factor, est)

    target_global_batch = per_batch * factor
    auto_steps = target_global_batch // per_batch
    if auto_steps < 1:
        auto_steps = 1
    auto_steps = min(auto_steps, 64)

    grad_accum_steps: int = int(auto_steps)
    logging.info(
        f"[epochs] auto grad_accum_steps={grad_accum_steps} "
        f"(per_batch={per_batch}, target_global_batch={target_global_batch}, factor={factor})"
    )

    def _cast_fp_buffers(module: torch.nn.Module, dtype: torch.dtype) -> None:
        for buf in module.buffers(recurse=True):
            if isinstance(buf, torch.Tensor) and buf.is_floating_point():
                try:
                    if buf.dtype != dtype:
                        buf.data = buf.data.to(dtype=dtype)
                except Exception:
                    pass

    if buffers_dtype is not None:
        target_for_buffers = model.module if hasattr(model, "module") else model
        _cast_fp_buffers(target_for_buffers, buffers_dtype)

    model_for_hist = model.module if hasattr(model, "module") else model
    hist: Optional[History] = None
    maybe_hist = getattr(model_for_hist, "logger", None)
    if isinstance(maybe_hist, History):
        hist = maybe_hist
    if hist is None:
        maybe_hist = getattr(model_for_hist, "history", None)
        if isinstance(maybe_hist, History):
            hist = maybe_hist
    if hist is None:
        hist = History().to(device)
        try:
            setattr(model_for_hist, "logger", hist)
        except Exception:
            pass

    if isinstance(hist, History):
        start_ns = posix_time("Asia/Seoul")
        start_sec = round(float(start_ns) / 1e9, 6)
        hist.start_session(start_sec)
        hist.set_epochs(total_epochs)

        os_name = platform.system()
        if os_name == "Linux":
            pretty = None
            with contextlib.suppress(Exception):
                if os.path.exists("/etc/os-release"):
                    with open("/etc/os-release", "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("PRETTY_NAME="):
                                pretty = line.strip().split("=", 1)[1].strip().strip('"')
                                break
            os_full = pretty or f"{os_name} {platform.release()}"
        elif os_name == "Darwin":
            ver, _, _ = platform.mac_ver()
            os_full = f"macOS {ver or platform.release()}"
        elif os_name == "Windows":
            ver = platform.version()
            rel = platform.release()
            os_full = f"Windows {rel} {ver}"
        else:
            os_full = f"{os_name} {platform.release()}"

        kernel = platform.release()
        arch_list = [platform.machine(), platform.processor() or ""]
        cpu_list: List[str] = []
        proc = platform.processor()
        if proc:
            cpu_list.append(proc)

        try:
            ram_bytes = Memory.total()
            ram_gb = int(round(float(ram_bytes) / (1024 ** 3)))
        except Exception:
            ram_gb = 0

        py_ver = platform.python_version()

        backend_list: List[str] = []
        if torch.cuda.is_available():
            backend_list.append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            backend_list.append("xpu")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            backend_list.append("mps")
        backend_list.append("cpu")

        hist.set_system_info(
            os_name=os_full,
            kernel=kernel,
            cpu_list=cpu_list,
            arch_list=arch_list,
            ram_gb=ram_gb,
            python_version=py_ver,
            backends=backend_list,
        )

    model_for_scaler = model.module if hasattr(model, "module") else model
    scaler_x_device = model_for_scaler.scaler.x_mean.device
    scaler_y_device = model_for_scaler.scaler.y_mean.device
    with torch.no_grad():
        x_count: int = 0
        x_sum: Optional[torch.Tensor] = None
        x_sum_sq: Optional[torch.Tensor] = None
        y_count: int = 0
        y_sum: Optional[torch.Tensor] = None
        y_sum_sq: Optional[torch.Tensor] = None

        for batch in train_loader:
            feats = batch["features"]
            labs = batch["labels"]
            if feats.ndim == 3 and feats.shape[1] == 1:
                feats = feats.reshape(feats.shape[0], -1)

            xf = feats.to(device=scaler_x_device, dtype=torch.float32)
            n_x = xf.shape[0]
            if n_x > 0:
                x_count += n_x
                sx = xf.sum(dim=0)
                sx2 = (xf * xf).sum(dim=0)
                if x_sum is None:
                    x_sum = sx
                    x_sum_sq = sx2
                else:
                    x_sum += sx
                    x_sum_sq += sx2

            yf = labs.to(device=scaler_y_device, dtype=torch.float32)
            if yf.ndim >= 2:
                yf = yf.reshape(yf.shape[0], -1)
            n_y = yf.shape[0]
            if n_y > 0:
                y_count += n_y
                sy = yf.sum(dim=0)
                sy2 = (yf * yf).sum(dim=0)
                if y_sum is None:
                    y_sum = sy
                    y_sum_sq = sy2
                else:
                    y_sum += sy
                    y_sum_sq += sy2
        if is_distributed():
            x_count_t = torch.tensor(
                float(x_count), device=scaler_x_device, dtype=torch.float64
            )
            torch.distributed.all_reduce(x_count_t, op=torch.distributed.ReduceOp.SUM)
            x_count = int(x_count_t.item())
            if x_sum is not None:
                torch.distributed.all_reduce(x_sum, op=torch.distributed.ReduceOp.SUM)
            if x_sum_sq is not None:
                torch.distributed.all_reduce(x_sum_sq, op=torch.distributed.ReduceOp.SUM)

            y_count_t = torch.tensor(
                float(y_count), device=scaler_y_device, dtype=torch.float64
            )
            torch.distributed.all_reduce(y_count_t, op=torch.distributed.ReduceOp.SUM)
            y_count = int(y_count_t.item())
            if y_sum is not None:
                torch.distributed.all_reduce(y_sum, op=torch.distributed.ReduceOp.SUM)
            if y_sum_sq is not None:
                torch.distributed.all_reduce(y_sum_sq, op=torch.distributed.ReduceOp.SUM)

        eps = float(model_for_scaler.scaler.eps)
        if x_count > 0 and x_sum is not None and x_sum_sq is not None:
            mean_x = x_sum / float(x_count)
            var_x = (x_sum_sq / float(x_count)) - mean_x * mean_x
            std_x = torch.sqrt(var_x.clamp_min(eps))
            if model_for_scaler.scaler.x_mean.shape != mean_x.shape:
                model_for_scaler.scaler.x_mean.resize_(mean_x.shape)
            if model_for_scaler.scaler.x_std.shape != std_x.shape:
                model_for_scaler.scaler.x_std.resize_(std_x.shape)
            model_for_scaler.scaler.x_mean.copy_(mean_x)
            model_for_scaler.scaler.x_std.copy_(std_x)

        if y_count > 0 and y_sum is not None and y_sum_sq is not None:
            mean_y = y_sum / float(y_count)
            var_y = (y_sum_sq / float(y_count)) - mean_y * mean_y
            std_y = torch.sqrt(var_y.clamp_min(eps))
            if model_for_scaler.scaler.y_mean.shape != mean_y.shape:
                model_for_scaler.scaler.y_mean.resize_(mean_y.shape)
            if model_for_scaler.scaler.y_std.shape != std_y.shape:
                model_for_scaler.scaler.y_std.resize_(std_y.shape)
            model_for_scaler.scaler.y_mean.copy_(mean_y)
            model_for_scaler.scaler.y_std.copy_(std_y)

    start_kst_ns = posix_time("Asia/Seoul")
    in_dim = int(ops.in_dim)
    use_timer = (
        (device.type == "cuda" and hasattr(torch.cuda, "Event")) or
        (device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "Event"))
    )
    train_steps = _num_batches(train_loader)
    val_steps = _num_batches(val_loader)
    total_updates = int(total_epochs) * (int(train_steps) + int(val_steps))
    status_bar = (
        get_tqdm(title="Training", total=total_updates, device=device)
        if local_rank == 0
        else None
    )
    scheduler_step_per_batch = bool(scheduler_step_per_batch)
    swa_enabled = swa_helper is not None
    swa_start_epoch = max(0, int(swa_start_epoch))
    swa_has_updated = False
    prev_io_time = 0.0
    prev_comp_time = 0.0
    prev_io_bytes = 0.0
    prev_flops = 0.0

    join_context = joining(model=model, optimizer=optimizer)
    with join_context:

        with contextlib.suppress(Exception):
            get_tlb().pin_thread()
            Memory.prefer_local_numa()
        cpu_pool = Memory.Pool(capacity=8) if device.type in {"cuda", "xpu"} else None
        from typing import Dict

        pool_handles: Dict[int, object] = {}

        def _pin_tensor(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            if device.type not in ("cuda", "xpu") or cpu_pool is None:
                return tensors
            from typing import List
            out: List[torch.Tensor] = []
            for t in tensors:
                if torch.is_tensor(t) and t.device.type == "cpu":
                    if hasattr(t, "is_pinned") and t.is_pinned():
                        out.append(t)
                        continue
                    buf, h = cpu_pool.get(tuple(t.shape), t.dtype, return_handle=True)
                    buf.copy_(t, non_blocking=False)
                    if h is not None:
                        pool_handles[id(buf)] = h
                    out.append(buf)
                else:
                    out.append(t)
            return tuple(out)

        def _to_device_with_stream(tensor: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(tensor):
                return tensor
            if device.type in ("cuda", "xpu") and tensor.device.type == "cpu":
                try:
                    if not (hasattr(tensor, "is_pinned") and tensor.is_pinned()):
                        pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
                        pinned.copy_(tensor, non_blocking=False)
                    else:
                        pinned = tensor
                    backend = getattr(torch, device.type, None)
                    if backend is None or not hasattr(backend, "current_stream") or not hasattr(backend, "Event"):
                        tensor_dev = pinned.to(device, non_blocking=False)
                        h = pool_handles.pop(id(tensor), None)
                        if h is not None:
                            with contextlib.suppress(Exception):
                                cpu_pool.release(h)
                        return tensor_dev
                    tensor_dev = pinned.to(device, non_blocking=True)
                    stream = backend.current_stream(device)
                    pinned.record_stream(stream)
                    h = pool_handles.pop(id(tensor), None)
                    if h is not None:
                        with contextlib.suppress(Exception):
                            evt = backend.Event()
                            evt.record(stream)
                            cpu_pool.release_after(h, evt)
                    return tensor_dev
                except Exception:
                    return tensor.to(device, non_blocking=(device.type in ("cuda", "xpu")))
            return tensor.to(device, non_blocking=(device.type in ("cuda", "xpu")))

        for epoch_idx in range(int(total_epochs)):
            if is_distributed():
                target_module = model.module if hasattr(model, "module") else model
                distributed_sync(target_module, device=device)
            flop_breakdown_epoch: Dict[str, float] = {}
            io_time: float = 0.0
            comp_time: float = 0.0
            io_bytes: float = 0.0
            flops: float = 0.0
            flop_counter_train = FlopCounter(model, mode="train", device=device)
            with flop_counter_train:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                t_fetch_start = time.perf_counter_ns()
                total_batches = len(train_loader)
                train_accum_since_last = 0
                for step_idx, _raw in enumerate(train_loader):
                    try:
                        train_accum_since_last += 1
                        feat, label, *_ = preprocess(_raw)
                        X = to_torch_tensor(feat)
                        X = torch.atleast_2d(X)
                        if X.dim() != 2:
                            raise RuntimeError(
                                f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                            )
                        if X.shape[1] != in_dim:
                            raise RuntimeError(
                                f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                            )
                        Y = to_torch_tensor(label)
                        t_ready = time.perf_counter_ns()
                        X, Y = _pin_tensor(X, Y)
                        if use_timer:
                            h2d_s_ev, h2d_e_ev = (
                                torch.Event(device=device, enable_timing=True),
                                torch.Event(device=device, enable_timing=True),
                            )
                            h2d_s_ev.record()
                            X = _to_device_with_stream(X)
                            Y = _to_device_with_stream(Y)
                            h2d_e_ev.record()
                            h2d_e_ev.synchronize()
                            h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                        else:
                            t_h2d_s = time.perf_counter_ns()
                            X = _to_device_with_stream(X)
                            Y = _to_device_with_stream(Y)
                            t_h2d_e = time.perf_counter_ns()
                            h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                        wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                        io_time += float(wait_s + h2d_s)
                        with contextlib.suppress(Exception):
                            io_bytes += float(
                                X.element_size() * X.nelement()
                                + Y.element_size() * Y.nelement()
                            )
                        should_sync = ((step_idx + 1) % max(1, grad_accum_steps) == 0) or (
                            step_idx + 1 == total_batches
                        )
                        if use_timer:
                            ev_s, ev_e = (
                                torch.Event(device=device, enable_timing=True),
                                torch.Event(device=device, enable_timing=True),
                            )
                            ev_s.record()
                        else:
                            t_comp_s = time.perf_counter_ns()
                        with no_sync(
                            model, enable=(grad_accum_steps > 1 and (not should_sync))
                        ):
                            with flop_counter_train.step(display=False) as train_counter:
                                with Autocast.float(device):
                                    Y_flat = Y.reshape(Y.shape[0], -1).to(
                                        device, dtype=param_dtype, non_blocking=True
                                    )
                                    td = to_tensordict(
                                        {"features": X, "labels_flat": Y_flat}
                                    )
                                    model_out = model(
                                        td,
                                        global_loss=top_loss,
                                        local_loss=bottom_loss,
                                        loss_weights=loss_controller.weights(),
                                    )
                                    if isinstance(model_out, TensorDictBase):
                                        td = model_out
                                        y_hat = td.get("pred")
                                        loss_val = td.get("loss_total", None)
                                        if (
                                            isinstance(loss_val, torch.Tensor)
                                            and loss_val.ndim > 0
                                        ):
                                            loss_val = loss_val.mean()
                                    else:
                                        y_hat, loss_val = model_out
                                if loss_val is None:
                                    raise RuntimeError(
                                        "Model returned no loss value during training. "
                                        "Ensure loss functions are provided and returning valid outputs."
                                    )
                                if not isinstance(loss_val, torch.Tensor):
                                    loss_val = torch.as_tensor(
                                        loss_val, device=device, dtype=param_dtype
                                    )
                                else:
                                    loss_val = loss_val.to(
                                        device=device, dtype=param_dtype
                                    )
                                accum_scale = max(1, grad_accum_steps)
                                loss_for_backprop = loss_val / float(accum_scale)
                                scaler.scale(loss_for_backprop).backward()
                                if should_sync:
                                    scaler.unscale_(optimizer)
                                    scaler.step(optimizer)
                                    scaler.update()
                                    optimizer.zero_grad(set_to_none=True)
                                    if scheduler_step_per_batch:
                                        with contextlib.suppress(Exception):
                                            sched.step()
                                with contextlib.suppress(Exception):
                                    flops += max(0.0, float(train_counter.get_total_flops()))
                                breakdown_getter = getattr(
                                    train_counter, "get_manual_breakdown", None
                                )
                                if callable(breakdown_getter):
                                    for name, value in breakdown_getter().items():
                                        with contextlib.suppress(Exception):
                                            flop_breakdown_epoch[name] = (
                                                flop_breakdown_epoch.get(name, 0.0)
                                                + float(value)
                                            )
                        if use_timer:
                            ev_e.record()
                            ev_e.synchronize()
                            comp_time += float(ev_s.elapsed_time(ev_e)) / 1000.0
                        else:
                            comp_time += (time.perf_counter_ns() - t_comp_s) / 1_000_000_000.0
                        with contextlib.suppress(Exception):
                            cudagraph_step_end()
                        if local_rank == 0 and should_sync:
                            io_elapsed = prev_io_time + float(io_time)
                            io_transferred = prev_io_bytes + float(io_bytes)
                            comp_elapsed = prev_comp_time + float(comp_time)
                            flop_total = prev_flops + float(flops)
                            mbps_cur = io_transferred / max(io_elapsed, 1e-06) / 1_000_000.0
                            tflops_cur = (
                                flop_total / max(comp_elapsed, 1e-06) / 1_000_000_000_000.0
                            )
                            update_tqdm(
                                status_bar,
                                finish=train_accum_since_last,
                                mbps=mbps_cur,
                                tflops=tflops_cur,
                            )
                            train_accum_since_last = 0
                        if isinstance(hist, History):
                            try:
                                if train_steps <= 0 or step_idx % max(1, int(train_steps * 0.01)) == 0:
                                    hist.record_batch(X, Y)
                            except Exception:
                                pass
                        t_fetch_start = time.perf_counter_ns()
                        if cpu_pool is not None and ((step_idx + 1) & 255) == 0:
                            with contextlib.suppress(Exception):
                                cpu_pool.collect()
                    finally:
                        pool_handles.clear()
            if val_loader is not None:
                flop_counter_val = FlopCounter(model, mode="eval", device=device)
                with flop_counter_val:
                    model.eval()
                    with Gradient.inference(model), Autocast.float(device):
                        t_fetch_start = time.perf_counter_ns()
                        for _vstep, _raw in enumerate(val_loader):
                            try:
                                feat, label, *_ = preprocess(_raw)
                                X = to_torch_tensor(feat)
                                X = torch.atleast_2d(X)
                                if X.dim() != 2:
                                    raise RuntimeError(
                                        f"features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}"
                                    )
                                if X.shape[1] != in_dim:
                                    raise RuntimeError(
                                        f"feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}"
                                    )
                                Y = to_torch_tensor(label)
                                t_ready = time.perf_counter_ns()
                                if use_timer:
                                    h2d_s_ev, h2d_e_ev = (
                                        torch.Event(device=device, enable_timing=True),
                                        torch.Event(device=device, enable_timing=True),
                                    )
                                    h2d_s_ev.record()
                                    X = _to_device_with_stream(X)
                                    Y = _to_device_with_stream(Y)
                                    h2d_e_ev.record()
                                    h2d_e_ev.synchronize()
                                    h2d_s = float(h2d_s_ev.elapsed_time(h2d_e_ev)) / 1000.0
                                else:
                                    t_h2d_s = time.perf_counter_ns()
                                    X = _to_device_with_stream(X)
                                    Y = _to_device_with_stream(Y)
                                    t_h2d_e = time.perf_counter_ns()
                                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                                wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                                io_time += float(wait_s + h2d_s)
                                with contextlib.suppress(Exception):
                                    io_bytes += float(
                                        X.element_size() * X.nelement()
                                        + Y.element_size() * Y.nelement()
                                    )
                                if use_timer:
                                    ev_s, ev_e = (
                                        torch.Event(device=device, enable_timing=True),
                                        torch.Event(device=device, enable_timing=True),
                                    )
                                    ev_s.record()
                                else:
                                    t_comp_s = time.perf_counter_ns()
                                with flop_counter_val.step(display=False) as val_counter:
                                    Yv_flat = Y.reshape(Y.shape[0], -1).to(
                                        device, dtype=param_dtype, non_blocking=True
                                    )
                                    tdv = to_tensordict(
                                        {"features": X, "labels_flat": Yv_flat}
                                    )
                                    model_out_val = model(
                                        tdv,
                                        global_loss=top_loss,
                                        local_loss=bottom_loss,
                                        loss_weights=loss_controller.weights(),
                                    )
                                    if isinstance(model_out_val, TensorDictBase):
                                        tdv = model_out_val
                                        _y = tdv.get("pred")
                                        _loss_val = tdv.get("loss_total", None)
                                        if (
                                            isinstance(_loss_val, torch.Tensor)
                                            and _loss_val.ndim > 0
                                        ):
                                            _loss_val = _loss_val.mean()
                                    else:
                                        _y, _loss_val = model_out_val
                                if _loss_val is None:
                                    raise RuntimeError(
                                        "Model returned no loss value during validation. "
                                        "Ensure loss functions are configured correctly."
                                    )
                                if not isinstance(_loss_val, torch.Tensor):
                                    _loss_val = torch.as_tensor(
                                        _loss_val, device=device, dtype=param_dtype
                                    )
                                else:
                                    _loss_val = _loss_val.to(
                                        device=device, dtype=param_dtype
                                    )
                                if use_timer:
                                    ev_e.record()
                                    ev_e.synchronize()
                                    comp_time += float(ev_s.elapsed_time(ev_e)) / 1000.0
                                else:
                                    comp_time += (
                                        time.perf_counter_ns() - t_comp_s
                                    ) / 1_000_000_000.0
                                with contextlib.suppress(Exception):
                                    flops += max(0.0, float(val_counter.get_total_flops()))
                                breakdown_getter = getattr(
                                    val_counter, "get_manual_breakdown", None
                                )
                                if callable(breakdown_getter):
                                    for name, value in breakdown_getter().items():
                                        with contextlib.suppress(Exception):
                                            flop_breakdown_epoch[name] = (
                                                flop_breakdown_epoch.get(name, 0.0)
                                                + float(value)
                                            )
                                if local_rank == 0:
                                    io_elapsed = prev_io_time + float(io_time)
                                    io_transferred = prev_io_bytes + float(io_bytes)
                                    comp_elapsed = prev_comp_time + float(comp_time)
                                    flop_total = prev_flops + float(flops)
                                    mbps_cur = (
                                        io_transferred
                                        / max(io_elapsed, 1e-06)
                                        / 1_000_000.0
                                    )
                                    tflops_cur = (
                                        flop_total
                                        / max(comp_elapsed, 1e-06)
                                        / 1_000_000_000_000.0
                                    )
                                    update_tqdm(
                                        status_bar,
                                        finish=1,
                                        mbps=mbps_cur,
                                        tflops=tflops_cur,
                                    )
                                t_fetch_start = time.perf_counter_ns()
                                if cpu_pool is not None and ((_vstep + 1) & 255) == 0:
                                    with contextlib.suppress(Exception):
                                        cpu_pool.collect()
                            finally:
                                pool_handles.clear()
            if is_distributed():
                stats = torch.tensor(
                    [comp_time, io_time, flops, io_bytes],
                    device=device,
                    dtype=torch.float64,
                )
                torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                world = max(1, get_world_size(device))
                stats /= world
                comp_time, io_time, flops, io_bytes = [float(x) for x in stats.tolist()]
                distributed_barrier(device)
            updated_this_epoch = False
            if swa_enabled and epoch_idx >= swa_start_epoch:
                with contextlib.suppress(Exception):
                    swa_helper.update_weight()
                    updated_this_epoch = True
            if not scheduler_step_per_batch:
                with contextlib.suppress(Exception):
                    sched.step()
            if updated_this_epoch:
                swa_has_updated = True
            prev_comp_time += float(comp_time)
            prev_io_time += float(io_time)
            prev_flops += float(flops)
            prev_io_bytes += float(io_bytes)
    model_for_scaler = model.module if hasattr(model, "module") else model
    scaler_y_device = model_for_scaler.scaler.y_mean.device
    with torch.no_grad():
        sum_x: Optional[torch.Tensor] = None
        sum_y: Optional[torch.Tensor] = None
        sum_x2: Optional[torch.Tensor] = None
        sum_xy: Optional[torch.Tensor] = None
        total_n: int = 0

        for batch in train_loader:
            x_raw = batch["features"].to(device)
            y_raw = batch["labels"].to(scaler_y_device)
            if y_raw.ndim >= 2:
                y_flat = y_raw.reshape(y_raw.shape[0], -1)
            else:
                y_flat = y_raw
            out = model(
                x_raw,
                labels_flat=None,
                net_loss=None,
                global_loss=None,
                local_loss=None,
                calibrate_output=False,
            )
            if isinstance(out, tuple):
                z_pred_raw, _ = out
            else:
                z_pred_raw = out

            z_pred = z_pred_raw.detach().to(
                device=scaler_y_device, dtype=torch.float64
            )
            if z_pred.ndim >= 2:
                z_pred = z_pred.reshape(z_pred.shape[0], -1)
            else:
                z_pred = z_pred.view(-1, 1)

            z_true = model_for_scaler.scaler.normalize_y(
                y_flat.detach()
            ).to(dtype=torch.float64)
            if z_true.ndim >= 2:
                z_true = z_true.reshape(z_true.shape[0], -1)
            else:
                z_true = z_true.view(-1, 1)

            if z_pred.shape[-1] != z_true.shape[-1]:
                f_pred = z_pred.shape[-1]
                f_true = z_true.shape[-1]
                if f_true % f_pred == 0:
                    group = f_true // f_pred
                    z_true = z_true.view(z_true.shape[0], group, f_pred).mean(dim=1)
                elif f_pred % f_true == 0:
                    group = f_pred // f_true
                    z_true = z_true.repeat_interleave(group, dim=1)
                else:
                    raise RuntimeError(
                        "Calibration: feature dimension mismatch between prediction and target "
                        f"that cannot be reconciled generically. "
                        f"z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
                    )

            if z_pred.shape[0] != z_true.shape[0]:
                raise RuntimeError(
                    "Calibration: batch dimension mismatch between prediction and target. "
                    f"z_pred.shape={tuple(z_pred.shape)}, z_true.shape={tuple(z_true.shape)}"
                )

            if z_pred.numel() == 0 or z_true.numel() == 0:
                continue

            n_batch = z_pred.shape[0]
            total_n += n_batch

            sx = z_pred.sum(dim=0)
            sy = z_true.sum(dim=0)
            sx2 = (z_pred * z_pred).sum(dim=0)
            sxy = (z_pred * z_true).sum(dim=0)

            if sum_x is None:
                sum_x = sx
                sum_y = sy
                sum_x2 = sx2
                sum_xy = sxy
            else:
                sum_x += sx
                sum_y += sy
                sum_x2 += sx2
                sum_xy += sxy
        if is_distributed():
            n_t = torch.tensor(
                float(total_n), device=scaler_y_device, dtype=torch.float64
            )
            torch.distributed.all_reduce(n_t, op=torch.distributed.ReduceOp.SUM)
            total_n = int(n_t.item())

            if sum_x is not None:
                torch.distributed.all_reduce(sum_x, op=torch.distributed.ReduceOp.SUM)
            if sum_y is not None:
                torch.distributed.all_reduce(sum_y, op=torch.distributed.ReduceOp.SUM)
            if sum_x2 is not None:
                torch.distributed.all_reduce(sum_x2, op=torch.distributed.ReduceOp.SUM)
            if sum_xy is not None:
                torch.distributed.all_reduce(sum_xy, op=torch.distributed.ReduceOp.SUM)

        if total_n > 0 and sum_x is not None and sum_y is not None and sum_x2 is not None and sum_xy is not None:
            N = float(total_n)
            mean_x = sum_x / N
            mean_y = sum_y / N
            Ex2 = sum_x2 / N
            Exy = sum_xy / N
            var_x = Ex2 - mean_x * mean_x
            cov_xy = Exy - mean_x * mean_y

            eps = float(model_for_scaler.scaler.eps)
            denom = var_x.clone()
            tiny_mask = denom.abs() < eps
            denom[tiny_mask] = 1.0

            a = (cov_xy / denom).to(dtype=torch.float32)
            b = (mean_y - a.to(dtype=torch.float64) * mean_x).to(dtype=torch.float32)
            a[tiny_mask] = 1.0
            b[tiny_mask] = 0.0

            model_for_scaler.scaler.set_affine(a, b)

    if local_rank == 0 and status_bar is not None:
        mbps = prev_io_bytes / max(prev_io_time, 1e-06) / 1_000_000.0
        tflops = prev_flops / max(prev_comp_time, 1e-06) / 1_000_000_000_000.0
        status_bar.set_postfix_str(
            f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=True
        )
        status_bar.close()
    end_kst_ns = posix_time("Asia/Seoul")
    try:
        dev_t = getattr(device, "type", "")
        total_t = prev_io_time + prev_comp_time
        if dev_t != "cpu":
            gpu_util, mem_util = _gpu_nvml_utils(device)
            if gpu_util is not None:
                if gpu_util < 95.0:
                    Dataset.request_scale_up(1.25)
                elif gpu_util > 98.0 or (mem_util is not None and mem_util >= 90.0):
                    Dataset.request_scale_down(0.90)
            else:
                if total_t > 0:
                    util_fallback = prev_comp_time / total_t
                    if util_fallback < 0.95:
                        Dataset.request_scale_up(1.25)
                    elif util_fallback > 0.99:
                        Dataset.request_scale_down(0.90)
        else:
            cpu_pct = _cpu_percent_now()
            if cpu_pct is not None:
                if cpu_pct > 80.0:
                    time.sleep(min(0.005, 0.001 * (cpu_pct - 80.0)))
            else:
                if total_t > 0:
                    util_fallback = prev_comp_time / total_t
                    if util_fallback > 0.80:
                        time.sleep(min(0.005, total_t * (util_fallback - 0.80)))
        if isinstance(hist, History):
            try:
                end_sec = round(float(end_kst_ns) / 1e9, 6)
                world = max(1, get_world_size(device)) if is_distributed() else 1
                hist.end_session(end_sec, peers=world)
                if ops.ckpt_dir and int(local_rank) == 0:
                    history_path = os.path.join(ops.ckpt_dir, "history.json")
                    recs = hist.save()
                    print(
                        f"[HIST-DUMP] rank={local_rank}, path={history_path}, records={len(recs)}",
                        flush=True,
                    )
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(recs, f)
            except Exception as e:
                print(f"[HIST-DUMP-ERROR] {type(e).__name__}: {e}", flush=True)
    except Exception:
        pass


def infer(
    model: Instance,
    device: torch.device,
    local_rank: int,
    ops: RuntimeConfig,
    *args: Any,
    data_loader: Any,
    chunk_dir: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Dict[Tuple, torch.Tensor]]:
    from typing import Dict, List

    out_shape = tuple(ops.out_shape)

    SPILL_TARGET_RATIO = 0.80
    RESERVE_MB = 512
    MIN_LIMIT_MB = 128
    MAX_LIMIT_MB = 4096
    CHUNK_MIN_MB = 64
    CHUNK_MAX_MB = 512

    BYTES_PER_MB = 1024.0 * 1024.0
    latest_avail_mb: float = float(Memory.available()) / BYTES_PER_MB
    total_mb: float = 0.0
    try:
        _tot = Memory.total()
        if _tot and _tot > 0:
            total_mb = float(_tot) / BYTES_PER_MB
    except Exception:
        total_mb = 0.0

    def _memory_limit() -> float:
        avail_mb = latest_avail_mb
        if total_mb > 0.0:
            used_mb = max(0.0, total_mb - avail_mb)
            target_used_mb = total_mb * SPILL_TARGET_RATIO
            dyn = max(0.0, target_used_mb - used_mb)
        else:
            usable = max(0.0, avail_mb - RESERVE_MB)
            dyn = usable * SPILL_TARGET_RATIO
        return max(MIN_LIMIT_MB, min(MAX_LIMIT_MB, dyn))

    def _chunk_target_mb() -> float:
        return max(CHUNK_MIN_MB, min(CHUNK_MAX_MB, _memory_limit() * 0.25))

    pred_pool = Memory.Pool(capacity=4) if device.type in {"cuda", "xpu"} else None
    with contextlib.suppress(Exception):
        get_tlb().pin_thread()
        Memory.prefer_local_numa()

    run_model = to_ddp(model, device=device)
    run_model.eval()
    module_eval = run_model.module if hasattr(run_model, "module") else run_model
    distributed_sync(module_eval, device=device)
    total_batches = _num_batches(data_loader)
    status_bar = (
        get_tqdm(title="Prediction", total=total_batches, device=device)
        if local_rank == 0
        else None
    )
    chunk_idx = 0
    queue_depth: int = max(
        2, min(16, int(max(1.0, _memory_limit() / max(1.0, _chunk_target_mb()))))
    )

    if not chunk_dir:
        raise RuntimeError(
            "infer: chunk_dir must be specified to enable streaming saves."
        )
    os.makedirs(chunk_dir, exist_ok=True)
    try:
        import psutil
        avail_bytes = psutil.virtual_memory().available
    except ImportError:
        avail_bytes = None

    if avail_bytes is not None:
        budget_bytes = min(avail_bytes * 0.10, 4 * 1024**3)
    else:
        fallback_budget = 4 * 1024**3
        fallback_budget = min(fallback_budget, queue_depth * 1 * 1024**2)
        budget_bytes = fallback_budget
        import logging
        logging.warning(
            f"psutil not available — using fallback budget_bytes={budget_bytes} bytes"
        )

    ref_tail_shape: Optional[Tuple[int, ...]] = None
    ref_dtype: Optional[torch.dtype] = None
    bad_tail: Optional[Tuple[int, ...]] = None
    shape_inconsistent: bool = False

    dtype_size = (
        torch.tensor([], dtype=ref_dtype or torch.float32).element_size()
        if ref_dtype is not None
        else torch.tensor([], dtype=torch.float32).element_size()
    )
    tail_prod = 1
    for d in out_shape[1:]:
        tail_prod *= d

    batch_size = 1
    try:
        batch_size = int(out_shape[0])
    except Exception:
        import logging

        logging.debug(
            f"Could not infer batch_size from out_shape[0]={out_shape[0] if out_shape else None}"
        )
        batch_size = 1
    item_bytes = batch_size * tail_prod * dtype_size
    if item_bytes <= 0:
        import logging
        logging.error(
            f"Estimated item_bytes non-positive ({item_bytes}). Defaulting to dtype_size={dtype_size}"
        )
        item_bytes = dtype_size

    if item_bytes > 0:
        max_batches_by_mem = int(budget_bytes / item_bytes)
    else:
        max_batches_by_mem = int(queue_depth)

    max_batches_by_mem = max(1, max_batches_by_mem)
    HARD_MAX_BATCHES = 256
    buffer_max_batches = min(int(queue_depth), max_batches_by_mem, HARD_MAX_BATCHES)
    buffer_max_batches = max(2, buffer_max_batches)

    import logging

    logging.info(
        f"Buffer configured with max_batches={buffer_max_batches} "
        f"(queue_depth={queue_depth}, budget_bytes={budget_bytes}, item_bytes={item_bytes})"
    )

    buffer = Memory.Buffer(max_batches=buffer_max_batches)

    def _writer_loop() -> None:
        nonlocal chunk_idx
        while True:
            if buffer.is_stopped() and buffer.empty():
                break
            try:
                t = buffer.get(timeout=1)
            except Exception:
                continue
            path = os.path.join(chunk_dir, f"chunk_{chunk_idx:06d}.pt")
            with contextlib.suppress(Exception):
                torch.save(t, path)
            chunk_idx += 1

    import threading as _threading_for_writer

    _writer_thread = _threading_for_writer.Thread(target=_writer_loop, daemon=True)
    _writer_thread.start()

    def _is_shape_consistent(batch: List[Tuple[torch.Tensor, Optional[object], Optional[object]]]) -> bool:
        if not batch:
            return True
        tail = tuple(batch[0][0].shape[1:])
        dt = batch[0][0].dtype
        for t, _, _ in batch:
            if t.dtype != dt or tuple(t.shape[1:]) != tail:
                return False
        return True

    flop_counter = FlopCounter(run_model, mode="eval", device=device)
    use_timer = False
    io_bytes: float = 0.0
    io_time: float = 0.0
    comp_time: float = 0.0
    total_flops: float = 0.0
    t_fetch_start = time.perf_counter_ns()
    rank = torch.distributed.get_rank() if is_distributed() else 0

    try:
        with flop_counter, Gradient.inference(run_model), Autocast.float(device):
            for _idx, _raw in enumerate(data_loader):
                feat, _label, *_ = preprocess(_raw)
                X = to_torch_tensor(feat)
                X = torch.atleast_2d(X)
                if X.dim() != 2:
                    raise RuntimeError(
                        f"infer: feats.ndim={X.dim()} (expect 2), shape={tuple(X.shape)}"
                    )
                if X.shape[1] != int(ops.in_dim):
                    raise AssertionError(
                        "infer: feature dim mismatch — "
                        f"feats.shape[1]={X.shape[1]} != in_dim={ops.in_dim}."
                    )
                if X.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    X = X.to(dtype=torch.float32)
                moved_h2d = False
                if device.type in ("cuda", "xpu") and X.device.type == "cpu":
                    try:
                        if not (hasattr(X, "is_pinned") and X.is_pinned()):
                            X_pinned = torch.empty_like(X, device="cpu", pin_memory=True)
                            X_pinned.copy_(X, non_blocking=False)
                        else:
                            X_pinned = X
                        X = X_pinned.to(device, non_blocking=True)
                        X_pinned.record_stream(getattr(torch, device.type).current_stream(device))
                        moved_h2d = True
                    except Exception:
                        X = X.to(device, non_blocking=(device.type in ("cuda", "xpu")))
                        moved_h2d = True
                if use_timer:
                    ev_h2d_s, ev_h2d_e = (
                        torch.Event(device=device, enable_timing=True),
                        torch.Event(device=device, enable_timing=True),
                    )
                    ev_h2d_s.record()
                    if not moved_h2d:
                        X = X.to(device, non_blocking=True)
                    ev_h2d_e.record()
                    ev_h2d_e.synchronize()
                    h2d_s = float(ev_h2d_s.elapsed_time(ev_h2d_e)) / 1000.0
                else:
                    t_h2d_s = time.perf_counter_ns()
                    if not moved_h2d:
                        X = X.to(device, non_blocking=True)
                    t_h2d_e = time.perf_counter_ns()
                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                wait_s = (time.perf_counter_ns() - t_fetch_start) / 1_000_000_000.0
                io_time += wait_s + h2d_s
                with contextlib.suppress(Exception):
                    io_bytes += float(X.element_size() * X.nelement())
                t0 = time.perf_counter_ns()
                state = getattr(run_model, "_autobs_infer", None)
                if state is None:
                    state = {"mb": 0, "util_prev": None, "bps_est": None}
                    setattr(run_model, "_autobs_infer", state)
                B = int(X.shape[0])
                mb = int(state["mb"] or max(1, B // 2))
                mb = max(1, min(B, mb))
                chunks = max(1, math.ceil(B / mb))
                s_idx = 0
                pre_alloc = (
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                )
                for _micro in range(chunks):
                    sl = slice(s_idx, min(B, s_idx + mb))
                    s_idx += mb
                    Xi = X[sl]
                    with no_sync(run_model, enable=True):
                        with flop_counter.step(display=False) as step_counter:
                            with contextlib.suppress(Exception):
                                mark_step = getattr(
                                    getattr(torch, "compiler", None),
                                    "cudagraph_mark_step_begin",
                                    None,
                                )
                                if callable(mark_step):
                                    mark_step()
                            tdp = to_tensordict({"features": Xi})
                            pred_out = run_model(
                                tdp,
                                global_loss=None,
                                local_loss=None,
                                loss_weights=None,
                                calibrate_output=True,
                            )
                            if isinstance(pred_out, TensorDictBase):
                                tdp = pred_out
                                y_hat = tdp.get("pred")
                            else:
                                y_hat, _ = pred_out
                    try:
                        evt: Optional[object] = None
                        handle: Optional[object] = None
                        if pred_pool is not None:
                            y_hat_cpu, handle = pred_pool.get_like(
                                y_hat, return_handle=True
                            )
                        else:
                            y_hat_cpu = torch.empty_like(
                                y_hat, device="cpu", pin_memory=True
                            )
                        y_hat_cpu.copy_(y_hat.detach(), non_blocking=True)
                        try:
                            if y_hat.device.type in ("cuda", "xpu"):
                                backend = getattr(torch, y_hat.device.type)
                                stream = backend.current_stream(y_hat.device)
                                y_hat_cpu.record_stream(stream)
                                evt = backend.Event()
                                evt.record(stream)
                        except Exception:
                            evt = None
                        y_hat_cpu = y_hat_cpu.contiguous()
                    except Exception:
                        y_hat_cpu = y_hat.detach().cpu().contiguous()
                        evt = None
                        handle = None

                    try:
                        tail = tuple(y_hat_cpu.shape[1:])
                        if ref_tail_shape is None:
                            ref_tail_shape = tail
                            ref_dtype = y_hat_cpu.dtype
                        else:
                            if tail != ref_tail_shape or y_hat_cpu.dtype != ref_dtype:
                                if not shape_inconsistent:
                                    bad_tail = tail
                                shape_inconsistent = True
                    except Exception:
                        pass
                    if rank != 0:
                        del y_hat_cpu
                        if pred_pool is not None and handle is not None:
                            with contextlib.suppress(Exception):
                                pred_pool.release(handle)
                        if evt is not None:
                            del evt
                        continue
                    if evt is not None:
                        with contextlib.suppress(Exception):
                            evt.synchronize()
                    _host = torch.empty_like(y_hat_cpu, device="cpu", pin_memory=False)
                    _host.copy_(y_hat_cpu, non_blocking=False)
                    buffer.put(_host)
                    if pred_pool is not None and handle is not None:
                        with contextlib.suppress(Exception):
                            pred_pool.release(handle)
                    with contextlib.suppress(Exception):
                        step_flops = float(step_counter.get_total_flops())
                    total_flops += max(0.0, step_flops)

                comp_time += (time.perf_counter_ns() - t0) / 1_000_000_000.0
                util_cur = float((time.perf_counter_ns() - t0) / 1_000_000_000.0) / max(
                    float(wait_s + h2d_s) + float((time.perf_counter_ns() - t0) / 1_000_000_000.0), 1e-6
                )
                grew = (state.get("util_prev") is None) or (util_cur >= state["util_prev"] - 0.02)
                state["util_prev"] = util_cur
                if state.get("bps_est") is None and device.type == "cuda":
                    delta = max(0, torch.cuda.memory_allocated(device) - pre_alloc)
                    if delta > 0:
                        state["bps_est"] = delta / float(mb)
                mb_mem = B
                if device.type == "cuda" and state.get("bps_est"):
                    free, total = torch.cuda.mem_get_info(device)
                    alloc_now = torch.cuda.memory_allocated(device)
                    target = int(total * 0.90)
                    headroom = max(0, target - alloc_now)
                    mb_mem = max(1, min(B, int(headroom / max(1, int(state["bps_est"])))))
                mb_util = (min(B, mb + max(1, mb // 4)) if grew else max(1, int(math.ceil(mb * 0.9))))
                state["mb"] = max(1, min(mb_mem, mb_util))
                if local_rank == 0:
                    mbps = io_bytes / max(io_time, 1e-06) / 1_000_000.0
                    tflops = total_flops / max(comp_time, 1e-06) / 1_000_000_000_000.0
                    update_tqdm(status_bar, finish=1, mbps=mbps, tflops=tflops)
                t_fetch_start = time.perf_counter_ns()
                if pred_pool is not None and ((_idx + 1) & 255) == 0:
                    with contextlib.suppress(Exception):
                        pred_pool.collect()
    finally:
        with contextlib.suppress(Exception):
            buffer.stop()
        _writer_thread.join(timeout=10.0)
        if _writer_thread.is_alive():
            logging.error(
                "Writer thread did not finish within timeout; potential blocking on buffer drain"
            )
            logging.warning("Switching to non-blocking flush mode to avoid hang.")
            try:
                while not buffer.empty():
                    t_rem = buffer.get(block=False)
                    path_rem = os.path.join(chunk_dir, f"chunk_{chunk_idx:06d}.pt")
                    torch.save(t_rem, path_rem)
                    chunk_idx += 1
            except Exception as ex:
                logging.error(f"Fallback drain encountered exception: {ex!r}")
        else:
            logging.info("Writer thread terminated cleanly.")
        if local_rank == 0 and status_bar is not None:
            mbps = io_bytes / max(io_time, 1e-06) / 1_000_000.0
            tflops = total_flops / max(comp_time, 1e-06) / 1_000_000_000_000.0
            status_bar.set_postfix_str(
                f"{mbps:.2f} MB/s, {tflops:.2f} TFLOPS", refresh=True
            )
            status_bar.close()
    if rank == 0:

        if not chunk_dir:
            with contextlib.suppress(Exception):
                chunk_dir = new_dir("pred_chunks")
                os.makedirs(chunk_dir, exist_ok=True)
        manifest = {
            "dir": chunk_dir,
            "num_chunks": int(chunk_idx),
            "out_shape": list(ops.out_shape),
            "dtype": (str(ref_dtype).replace("torch.", "") if ref_dtype is not None else None),
        }
        if shape_inconsistent:
            manifest["variable_shape"] = True
            if ref_tail_shape is not None:
                manifest["example_tail_shape"] = list(ref_tail_shape)
        with contextlib.suppress(Exception):
            with open(
                os.path.join(chunk_dir or "", "manifest.json"), "w", encoding="utf-8"
            ) as manifest_file:
                json.dump(manifest, manifest_file)
        result = None
    return None


def main(*args: Any, **kwargs: Any) -> Optional[Instance]:
    from ..data.pipeline import fetch

    if not args:
        raise TypeError("main requires at least a RuntimeConfig argument")
    initialize_python_path()
    ret_sink: Optional[Dict[Any, Any]] = None
    if isinstance(args[0], RuntimeConfig):
        ops = args[0]
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if len(args) >= 2:
            ret_sink = args[1]
    elif len(args) >= 2 and isinstance(args[1], RuntimeConfig):
        local_rank = int(args[0])
        ops = args[1]
        if len(args) >= 3:
            ret_sink = args[2]
    else:
        raise TypeError(
            "main expects (RuntimeConfig,), (RuntimeConfig, ret_sink), (local_rank, RuntimeConfig), or (local_rank, RuntimeConfig, ret_sink) arguments"
        )

    if ops.mode == "train":
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
        device = get_device()
        _set_backend(device)
        backend = _backend_type(device)
        _initialize_group(backend, device, local_rank)
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        cfg = replace(cfg, device=device)
        model = Instance(ops.in_dim, ops.out_shape, config=cfg)
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            fallback_init = os.path.join(ops.init_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_init):
                cpu_state = torch.load(fallback_init, map_location="cpu")
                resize_scaler_buffer(model, cpu_state)
                model.load_state_dict(cpu_state, strict=False)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=ignored_pattern)
                    m_sd = get_model_state_dict(
                        model,
                        options=StateDictOptions(
                            full_state_dict=True, cpu_offload=False
                        ),
                    )
                    m_sd = _trim_dcp_keys(m_sd)
                    load(
                        state_dict={"model": m_sd},
                        storage_reader=FileSystemReader(ops.init_ckpt_dir),
                    )
                    resize_scaler_buffer(model, m_sd)
                    set_model_state_dict(
                        model, m_sd, options=StateDictOptions(strict=False)
                    )
        if ops.sources is None:
            raise RuntimeError("RuntimeConfig.sources is required but None")
        metadata = Metadata.for_device(device)
        expanded_sources = _expand(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        meta_info = _from_meta(_first_source_path(ops.sources))
        meta_feature_dim = int(meta_info.get("feature_dim", ops.in_dim))
        if meta_feature_dim != int(ops.in_dim):
            raise RuntimeError(
                "dataset feature_dim mismatch: "
                f"meta={meta_feature_dim}, expected in_dim={ops.in_dim}"
            )
        meta_label_shape = tuple(
            int(x) for x in meta_info.get("label_shape", list(ops.out_shape))
        )
        if tuple(meta_label_shape) != tuple(ops.out_shape):
            raise RuntimeError(
                "dataset label_shape mismatch: "
                f"meta={meta_label_shape}, expected out_shape={tuple(ops.out_shape)}"
            )
        fractions = meta_info.get("fractions", [1.0, 0.0])
        if isinstance(fractions, (list, tuple)) and len(fractions) >= 2:
            actual_val_frac = float(fractions[-1])
            if not math.isclose(
                actual_val_frac, float(ops.val_frac), rel_tol=0.001, abs_tol=0.001
            ):
                warnings.warn(
                    "val_frac=%s differs from memmap metadata (%s); using metadata value for loaders"
                    % (ops.val_frac, actual_val_frac)
                )
                ops = replace(ops, val_frac=actual_val_frac)
        model, _, _ = Fusion.use_nvidia_layers(model, device=device)
        Autocast.configure(model, metadata=metadata)
        param_dtype = _unify_param_dtype(
            model,
            prefer=(
                torch.bfloat16
                if getattr(device, "type", None) == "cuda"
                and torch.cuda.is_bf16_supported()
                else None
            ),
        )
        if param_dtype is None:
            param_dtype = torch.float32
        fp8_ok, fp8_reason = is_float8_supported(device)
        fp8_enabled = False
        fp8_backend: Optional[str] = None
        disable_note: Optional[str] = None
        if fp8_ok:
            model, fp8_enabled, fp8_backend = Fusion.enable_float8_training(
                model, metadata=metadata, logger=_float8_log
            )
            if not fp8_enabled:
                disable_note = fp8_backend
        else:
            disable_note = fp8_reason
        if not fp8_enabled:
            Autocast.configure(model, metadata=metadata)
            if disable_note:
                _float8_log(f"[FP8] disabled: {disable_note}")
        model.train()
        world = get_world_size(device)
        mesh = init_device_mesh(
            "cuda" if device.type == "cuda" else device.type, (world,)
        )
        amp_candidates = Autocast.float_amp_priority(device)
        amp_reduce_dtype = Autocast.negotiate(
            amp_candidates,
            fallback=torch.float32,
            context="fsdp.reduce",
            device=device,
            meta=metadata,
        )
        amp_buffers_dtype = Autocast.negotiate(
            amp_candidates,
            fallback=torch.float32,
            context="buffers.bn",
            device=device,
            meta=metadata,
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=amp_reduce_dtype,
            reduce_dtype=amp_reduce_dtype,
            output_dtype=amp_reduce_dtype,
            cast_forward_inputs=False,
        )
        ignored_params: List[torch.nn.Parameter] = []
        for module in model.modules():
            for name in ("alpha_t", "alpha_s", "gem_p", "cls_query", "cls"):
                if hasattr(module, name):
                    p = getattr(module, name)
                    if isinstance(p, torch.nn.Parameter):
                        ignored_params.append(p)
        ignored_param_registry = _IdentityParamSet(tuple(ignored_params))
        _m_pre = model.module if hasattr(model, "module") else model
        _preload_layers(_m_pre, device)
        _assert_unified_layer_dtype(_m_pre, device)
        _assert_no_meta_tensors(_m_pre)
        _assert_no_fake_dtensor(_m_pre)
        wrapped = set()
        try:
            for submodule in _get_layers(
                getattr(model, "processor", None)
            ) + _get_layers(getattr(model, "controller", None)):
                _wrap_fsdp(
                    submodule,
                    mesh,
                    mp_policy,
                    wrapped=wrapped,
                    ignored_param_registry=ignored_param_registry,
                )
            model = (
                _wrap_fsdp(
                    model,
                    mesh,
                    mp_policy,
                    wrapped=wrapped,
                    ignored_param_registry=ignored_param_registry,
                )
                or model
            )
        except (RuntimeError, ValueError, TypeError):
            model = to_fsdp(
                model,
                mesh=mesh,
                mp_policy=mp_policy,
                ignored_params=(
                    ignored_param_registry if len(ignored_param_registry) > 0 else None
                ),
                reshard_after_forward=False,
                sync_module_states=True,
            )
        for ignored_param in ignored_param_registry:
            _coerce_dtensor(ignored_param, mesh, placements=(Replicate(),))
        for parameter in model.parameters():
            _coerce_dtensor(parameter, mesh)
        _m_post = model.module if hasattr(model, "module") else model
        _assert_unified_layer_dtype(_m_post, device)
        _assert_no_meta_tensors(_m_post)
        _assert_no_fake_dtensor(_m_post, allow_dtensor=True)
        _enable_meta_monitor(_m_post)
        distributed_sync(_m_post, device=device)
        net_params = [p for p in model.parameters()]
        optimizer = AdamW.float(
            net_params,
            lr=ops.base_lr,
            weight_decay=ops.weight_decay,
            metadata=metadata,
            logger=None,
        )
        if ops.init_ckpt_dir is not None and os.path.isdir(ops.init_ckpt_dir):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                _initialize_adamw(optimizer)
                optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)
                try:
                    load(
                        state_dict={"optimizer": optim_sd},
                        storage_reader=FileSystemReader(ops.init_ckpt_dir),
                    )
                except (
                    FileNotFoundError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                    CheckpointException,
                ) as exc:
                    if "optimizer" not in str(exc).lower():
                        raise
                else:
                    set_optimizer_state_dict(
                        model,
                        optimizer,
                        optim_sd,
                        options=StateDictOptions(strict=False),
                    )
                    _initialize_adamw(optimizer)
        top_loss = DataFidelityLoss(
            out_shape=ops.out_shape,
            reduction="mean",
        )

        local_crps = CRPSLoss(
            dim=-1,
            reduction="none",
            detach_stats=True,
        )
        local_z = StandardNormalLoss(
            confidence=0.99,
            metric="z_value",
            two_tailed=True,
            penalty="softplus",
            tau=1.0,
            mu_mode="error",
            std_mode="pooled",
            ddof=1,
            clamp_max=8.0,
            detach_stats=True,
            dim=-1,
            reduction="none",
            skew=ops.loss_skew,
        )
        local_t = StudentsTLoss(
            confidence=0.99,
            metric="t_value",
            two_tailed=True,
            df=4,
            mu_mode="error",
            std_mode="pooled",
            ddof=1,
            clamp_max=8.0,
            detach_stats=True,
            dim=-1,
            reduction="none",
            skew=ops.loss_skew,
        )

        bottom_loss = TiledLoss(
            nn.Sequential(),
            mask_mode=ops.loss_mask_mode,
            mask_value=ops.loss_mask_value,
            tile_dim=ops.loss_tile_dim,
            tile_size=ops.loss_tile_size,
            reduction="mean",
        )
        bottom_loss.base = LinearCombinationLoss(
            coefficient=[0.8, 0.1, 0.1],
            loss=[local_crps, local_z, local_t],
            reduce_each=False,
            auto_schedule=True,
            schedule_momentum=0.9,
            min_coeff=0.05,
            max_coeff=0.90,
            eps=1e-6,
        )
        loss_controller = LossWeightController()
        ckpt_state_path = loader_state_path(ops.ckpt_dir or "")
        init_state_path = (
            loader_state_path(ops.init_ckpt_dir) if ops.init_ckpt_dir else None
        )
        state_train: Dict[str, Any] = {}
        state_val: Dict[str, Any] = {}
        _dlp = (
            ckpt_state_path
            if os.path.isfile(ckpt_state_path)
            else (
                init_state_path
                if init_state_path and os.path.isfile(init_state_path)
                else None
            )
        )
        restore_dl_state = False
        if _dlp:
            with contextlib.suppress(Exception):
                _dl_json = json.load(open(_dlp, "r", encoding="utf-8"))
                if isinstance(_dl_json, dict):
                    state_train = _dl_json.get("train", {}) or {}
                    state_val = _dl_json.get("val", {}) or {}
                    restore_dl_state = bool(state_train) or bool(state_val)
        train_loader: Any = None
        val_loader: Any = None
        keep: Any = None
        try:
            expanded_sources = _expand(ops.sources)
            if expanded_sources is not ops.sources:
                ops = replace(ops, sources=expanded_sources)
            accelerator_types = {"cuda", "xpu", "mps"}
            device_type = getattr(device, "type", None)
            if not device_type:
                device_str = str(device)
                device_type = device_str.split(":", 1)[0]
            non_blocking_copy = device_type in accelerator_types
            train_loader, val_loader, keep = fetch(
                sources=ops.sources,
                device=device,
                val_frac=float(ops.val_frac),
                non_blocking_copy=non_blocking_copy,
                flatten_features=True,
            )
            if restore_dl_state:
                with contextlib.suppress(Exception):
                    train_loader.load_state_dict(state_train)
                if val_loader is not None:
                    with contextlib.suppress(Exception):
                        val_loader.load_state_dict(state_val)
            train_steps = _num_batches(train_loader)
            val_steps = _num_batches(val_loader)
            steps_per_epoch = max(1, train_steps + val_steps)
            total_epochs = int(ops.epochs)
            total_steps = max(1, total_epochs * steps_per_epoch)
            if ops.warmup_ratio > 0.0:
                warmup_steps = max(1, int(total_steps * ops.warmup_ratio))
                main_steps = max(1, total_steps - warmup_steps)
            else:
                warmup_steps = 0
                main_steps = max(1, total_steps)
            base = float(ops.base_lr)
            emin = float(ops.eta_min)
            start_factor = 0.001
            lr_lambda = partial(
                _scheduler,
                warmup_steps=warmup_steps,
                start_factor=start_factor,
                base=base,
                main_steps=main_steps,
                emin=emin,
            )
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            scheduler_step_per_batch = True
            swa_helper: Optional[StochasticWeightAverage] = None
            swa_start_epoch = total_epochs
            enable_swa_cfg = bool(getattr(ops, "swa_enabled", False))
            start_epoch_cfg = getattr(ops, "swa_start_epoch", None)
            enable_swa = (
                enable_swa_cfg or start_epoch_cfg is not None
            ) and SWALR is not None
            if enable_swa:
                tracked_module = model.module if hasattr(model, "module") else model
                use_buffers = True
                try:
                    swa_helper = stochastic_weight_average(
                        tracked_module, use_buffers=use_buffers
                    )
                except Exception:
                    swa_helper = None
                if swa_helper is not None:
                    scheduler_step_per_batch = False
                    if start_epoch_cfg is not None:
                        try:
                            swa_start_epoch = max(0, int(start_epoch_cfg))
                        except (TypeError, ValueError):
                            swa_start_epoch = max(1, total_epochs // 2)
                    else:
                        swa_start_epoch = max(1, total_epochs // 2)
                    eta_min = float(getattr(ops, "eta_min", 0.0) or 0.0)
                    base_lr = float(ops.base_lr)
                    default_swa_lr = max(
                        1e-8, eta_min if eta_min > 0.0 else 0.1 * base_lr
                    )
                    swa_lr = default_swa_lr
                    anneal_epochs = max(1, max(1, total_epochs // 10))
                    try:
                        sched = SWALR(
                            optimizer,
                            swa_lr=swa_lr,
                            anneal_epochs=anneal_epochs,
                            anneal_strategy="cos",
                        )
                    except Exception:
                        scheduler_step_per_batch = True
                        swa_helper = None
                        swa_start_epoch = total_epochs
            scaler = torch.amp.GradScaler(
                enabled=(device.type == "cuda" and (not torch.cuda.is_bf16_supported()))
            )

                                                                                        
            try:
                get_tlb().pin_thread()
                with contextlib.suppress(Exception):
                    Memory.prefer_local_numa()
            except Exception:
                pass
            epochs(
                model=model,
                device=device,
                local_rank=local_rank,
                ops=ops,
                param_dtype=param_dtype,
                optimizer=optimizer,
                scaler=scaler,
                sched=sched,
                loss_controller=loss_controller,
                top_loss=top_loss,
                bottom_loss=bottom_loss,
                train_loader=train_loader,
                val_loader=val_loader,
                total_epochs=total_epochs,
                scheduler_step_per_batch=scheduler_step_per_batch,
                swa_helper=swa_helper,
                swa_start_epoch=swa_start_epoch,
                buffers_dtype=amp_buffers_dtype,
            )
        finally:
            if keep is not None:
                keep.cleanup()
        if local_rank == 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=ignored_pattern)
                model_sd = get_model_state_dict(
                    model,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                optim_sd = get_optimizer_state_dict(model, optimizers=optimizer)
                writer = FileSystemWriter(
                    ops.ckpt_dir or "", sync_files=True, overwrite=True
                )
                save(
                    state_dict={"model": model_sd, "optimizer": optim_sd},
                    storage_writer=writer,
                )
            if ops.ckpt_dir:
                fallback_path = os.path.join(ops.ckpt_dir, "model.pt")
                _state: Dict[str, Any] = {}
                for k, v in model.state_dict().items():
                    if isinstance(v, torch.Tensor):
                        t = v.detach().cpu()
                        if t.is_floating_point():
                            t = t.to(dtype=torch.float64)
                        _state[k] = t
                    else:
                        _state[k] = v
                torch.save(_state, fallback_path)
            with contextlib.suppress(Exception):
                _dl = {
                    "train": (
                        train_loader.state_dict() if train_loader is not None else {}
                    ),
                    "val": (val_loader.state_dict() if val_loader is not None else {}),
                }
                with open(
                    loader_state_path(ops.ckpt_dir or ""), "w", encoding="utf-8"
                ) as _f:
                    json.dump(_dl, _f)
        torch.distributed.barrier(
            device_ids=[local_rank] if device.type in ("cuda", "xpu") else None
        )
        torch.distributed.destroy_process_group()
        return None
    if ops.mode in ("predict", "infer"):
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
        device = get_device()
        _set_backend(device)
        backend = _backend_type(device)
        if not torch.distributed.is_initialized():
            _initialize_group(backend, device, local_rank)
        cfg = coerce_model_config(
            ops.cfg_dict if isinstance(ops.cfg_dict, dict) else ops.cfg_dict
        )
        model = Instance(ops.in_dim, ops.out_shape, config=cfg)
        if ops.model_ckpt_dir is not None and os.path.isdir(ops.model_ckpt_dir):
            fallback_model = os.path.join(ops.model_ckpt_dir, "model.pt")
            if os.path.isfile(fallback_model):
                cpu_state = torch.load(fallback_model, map_location="cpu")
                resize_scaler_buffer(model, cpu_state)
                model.load_state_dict(cpu_state, strict=False)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=ignored_pattern)
                    m_sd = get_model_state_dict(
                        model,
                        options=StateDictOptions(
                            full_state_dict=True, cpu_offload=True
                        ),
                    )
                    m_sd = _trim_dcp_keys(m_sd)
                    load(
                        state_dict={"model": m_sd},
                        storage_reader=FileSystemReader(ops.model_ckpt_dir),
                    )
                    resize_scaler_buffer(model, m_sd)
                    set_model_state_dict(
                        model, m_sd, options=StateDictOptions(strict=False)
                    )
        model.to(device, non_blocking=True).eval()
        metadata = Metadata.for_device(device)
        model, _, _ = Fusion.use_nvidia_layers(model, device=device)
        _m_eval = model.module if hasattr(model, "module") else model
        _preload_layers(_m_eval, device)
        _assert_unified_layer_dtype(_m_eval, device)
        _assert_no_meta_tensors(_m_eval)
        _assert_no_fake_dtensor(_m_eval)
        _enable_meta_monitor(_m_eval)
        _unify_param_dtype(
            model,
            prefer=(
                torch.bfloat16
                if (
                    getattr(device, "type", None) == "cuda"
                    and torch.cuda.is_bf16_supported()
                )
                else None
            ),
        )
        Autocast.configure(model, metadata=metadata)
        fp8_infer_ok, fp8_infer_reason = is_float8_supported(device)
        if fp8_infer_ok:
            model, _, _ = Fusion.enable_float8_prediction(
                model, metadata=metadata, logger=_float8_log
            )
        else:
            Autocast.configure(model, metadata=metadata)
            _float8_log(f"[FP8] disabled: {fp8_infer_reason}")
        if ops.sources is None:
            raise RuntimeError("RuntimeConfig.sources is required but None")
        model.eval()
        with contextlib.suppress(Exception):
            _calibrate_per_sample_mem(
                model=model,
                device=device,
                ops=ops,
            )

        expanded_sources = _expand(ops.sources)
        if expanded_sources is not ops.sources:
            ops = replace(ops, sources=expanded_sources)
        data_loader, _, keep = fetch(
            sources=ops.sources,
            device=device,
            val_frac=0.0,
            non_blocking_copy=True,
        )
        chunk_dir = (os.path.join(ops.ckpt_dir, "pred_chunks") if (ops.ckpt_dir or "") else None)
        if chunk_dir and torch.distributed.get_rank() == 0:
            with contextlib.suppress(Exception):
                os.makedirs(chunk_dir, exist_ok=True)
        if torch.distributed.is_initialized():
            pass
        if ops.mode in ("predict", "infer"):
            if not chunk_dir:
                raise RuntimeError("predict/infer requires chunk_dir (streaming enforced)")
        try:
                                                                                    
            result = infer(
                model=model,
                device=device,
                local_rank=local_rank,
                ops=ops,
                data_loader=data_loader,
                chunk_dir=chunk_dir,
            )
            if result is not None and ret_sink is not None:
                ret_sink.update(result)
        finally:
            if keep is not None:
                keep.cleanup()
        distributed_barrier(device)
        torch.distributed.destroy_process_group()
        return None
    raise ValueError(f"unsupported ops mode: {ops.mode}")


torch_compile_safe(runtime_module=sys.modules[__name__])
