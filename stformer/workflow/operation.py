# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import contextlib
import gc
import importlib
import json
import math
import multiprocessing
import os
import shutil
import socket
import sys
import time
import warnings

import torch
import torch.distributed
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
import torch.multiprocessing
from tqdm.auto import tqdm

from ..architecture.network import Config
from ..architecture.module import StandardNormalLoss
from ..architecture.module import StudentsTLoss
from ..architecture.module import TiledLoss
from ..architecture.network import Model
from ..pipeline.collate import stream
from ..pipeline.dataset import MemoryMappedTensorStream
from ..toolkit.capability import apply_threading_defaults
from ..toolkit.capability import get_device
from ..toolkit.capability import is_cpu_bf16_supported
from ..toolkit.capability import is_cuda_bf16_supported
from ..toolkit.capability import optimal_procs
from ..toolkit.optimization import AdamW
from ..toolkit.optimization import Architecture
from ..toolkit.optimization import Autocast
from ..toolkit.optimization import fsdp_no_sync
from ..toolkit.optimization import get_total_flops
from ..toolkit.optimization import register_flop_hooks

try:
    from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
except Exception:
    def precompute_float8_dynamic_scale_for_fsdp(*args: Any, **kwargs: Any) -> Any:
        return None

try:
    from torch.distributed.run import elastic_launch, LaunchConfig
except Exception:
    from torch.distributed.launcher.api import elastic_launch, LaunchConfig

try:
    from torch.distributed.algorithms.join import Join
except Exception:
    Join = None

try:
    from torch.utils.flop_counter import FlopCounterMode
except Exception:
    class _NoOpFlops:
        def __enter__(self) -> Any: return self
        def __exit__(self, *exc: Any) -> bool: return False
        def get_total_flops(self) -> int: return 0

    def FlopCounterMode(display: bool = False) -> Any:
        return _NoOpFlops()

sentences_to_ignore = [
    'torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.*',
    'torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to save in a single process.*',
    'TypedStorage is deprecated.*',
]
pattern_to_ignore = '|'.join((f'({sentence})' for sentence in sentences_to_ignore))


def _import_callable(spec: str) -> Callable:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('Empty spec for callable import')
    raw = spec.strip()
    root_pkg = __package__.split('.', 1)[0] if __package__ else 'stformer'
    default_module = f'{root_pkg}.toolkit.optimization'
    if ':' in raw:
        mod_part, fn_part = raw.split(':', 1)
    else:
        mod_part, fn_part = ('', raw)
    mod_part = mod_part.strip()
    fn_part = fn_part.strip()
    if not fn_part:
        raise ValueError(f'Missing function in spec: {spec}')
    if not mod_part:
        mod_name = default_module
    elif mod_part.startswith('.'):
        mod_name = f'{root_pkg}{mod_part}'
    elif not mod_part.startswith(root_pkg + '.') and (mod_part.split('.')[0] not in ('importlib', 'torch', 'math', 'sys')):
        mod_name = f'{root_pkg}.{mod_part}'
    else:
        mod_name = mod_part
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_part, None)
    if not callable(fn):
        raise TypeError(f'{mod_name}:{fn_part} is not callable or not found')
    return fn


def _prune_dcp_state_keys(state: Any) -> Any:
    try:
        keys = list(state.keys())
    except Exception:
        return state
    for k in list(keys):
        s = str(k)
        if s.endswith('._extra_state') or s.endswith('_extra_state'):
            try:
                state.pop(k, None)
            except Exception:
                pass
    return state


_NVTX_SOFT_COUNTER: float = 0.0
_NVTX_FLOPS_GETTER: Optional[Callable[[], float]] = None


def register_nvtx_flops_getter(fn: Callable[[], float]) -> None:
    global _NVTX_FLOPS_GETTER
    _NVTX_FLOPS_GETTER = fn


def _nvtx_soft_add(v: float) -> None:
    global _NVTX_SOFT_COUNTER
    try:
        _NVTX_SOFT_COUNTER += float(v) if float(v) > 0 else 0.0
    except Exception:
        pass


def _nvtx_soft_getter() -> float:
    return float(_NVTX_SOFT_COUNTER)


def _ensure_nvtx_flops_getter() -> None:
    global _NVTX_FLOPS_GETTER
    if _NVTX_FLOPS_GETTER is not None:
        return
    hook = os.environ.get('STF_NVTX_FLOPS_FN', '').strip()
    if hook:
        try:
            _NVTX_FLOPS_GETTER = _import_callable(hook)
            return
        except Exception:
            pass
    _NVTX_FLOPS_GETTER = _nvtx_soft_getter


class _NoOpNvtxCounter:
    def __enter__(self) -> '_NoOpNvtxCounter': return self
    def __exit__(self, *exc: Any) -> bool: return False
    def get_total_flops(self) -> float: return 0.0


class _NvtxCounter:
    def __init__(self, getter: Callable[[], float]) -> None:
        self._getter = getter
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> '_NvtxCounter':
        try:
            self._start = float(self._getter())
        except Exception:
            self._start = 0.0
        return self

    def __exit__(self, *exc: Any) -> bool:
        try:
            self._end = float(self._getter())
        except Exception:
            self._end = self._start
        return False

    def get_total_flops(self) -> float:
        end = float(self._end)
        if not end > self._start:
            try:
                end = float(self._getter())
            except Exception:
                end = self._start
        return max(0.0, float(end) - float(self._start))


def NVTXCounterMode(device: Optional[torch.device] = None) -> Any:
    try:
        dev = device if device is not None else get_device()
    except Exception:
        dev = None
    if dev is None or getattr(dev, 'type', None) != 'cuda':
        return _NoOpNvtxCounter()
    _ensure_nvtx_flops_getter()
    return _NvtxCounter(_NVTX_FLOPS_GETTER)


_SIZEOF = {
    'float64': 8,
    'float32': 4,
    'float16': 2,
    'bfloat16': 2,
    'int64': 8,
    'int32': 4,
    'int16': 2,
    'int8': 1,
    'uint8': 1,
    'bool': 1,
}


def _canonical_dtype(x: torch.dtype | str) -> str:
    if isinstance(x, torch.dtype):
        s = str(x).lower()
    else:
        s = str(x).strip().lower()
    if s.startswith('torch.'):
        s = s.split('.', 1)[1]
    s = s.lstrip('<>|=')
    aliases = {
        'float': 'float32',
        'double': 'float64',
        'half': 'float16',
        'halffloat': 'float16',
        'boolean': 'bool',
        'bool_': 'bool',
        'bf16': 'bfloat16',
        'f16': 'float16',
        'f32': 'float32',
        'f64': 'float64',
        'i8': 'int8',
        'i16': 'int16',
        'i32': 'int32',
        'i64': 'int64',
        'u8': 'uint8',
    }
    return aliases.get(s, s)


def _size(dtype: torch.dtype | str) -> int:
    n = _canonical_dtype(dtype)
    if n not in _SIZEOF:
        raise TypeError(f'unsupported dtype: {dtype}')
    return _SIZEOF[n]


def _mp_env() -> None:
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except Exception:
        pass
    start_method = 'spawn' if str(sys.platform).lower().startswith('win') else 'forkserver'
    for _mod in (multiprocessing, torch.multiprocessing):
        try:
            _mod.set_start_method(start_method, force=True)
        except RuntimeError:
            pass
        except Exception:
            pass


def _start_method() -> str:
    cur = torch.multiprocessing.get_start_method(allow_none=True)
    return 'spawn' if str(sys.platform).lower().startswith('win') else 'forkserver' if cur is None else str(cur)


def _status_bar(activity: str, total: int, dev: torch.device) -> tqdm:
    return tqdm(
        total=total,
        desc=f'{activity} ({dev.type.upper()})',
        unit='I/O < 0.01 MB/s, COM < 0.01 TFLOPS',
        bar_format='{desc}' + '{bar} {percentage:3.0f}% ' + '({unit}) Elapsed: {elapsed}, Remaining: {remaining}',
        colour='green',
        position=0,
        leave=False,
        file=sys.stdout,
    )


def _default_temp() -> str:
    return os.environ.get('TEMP', 'C:\\Windows\\Temp') if sys.platform.startswith('win') else '/tmp' if os.path.isdir('/tmp') else '/var/tmp'


def _new_dir(prefix: str) -> str:
    base = _default_temp()
    os.makedirs(base, exist_ok=True)
    d = os.path.join(base, f'{prefix}_{os.getpid()}_{os.urandom(4).hex()}')
    os.makedirs(d, exist_ok=True)
    return d


def _is_port_available(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _establish(ep: Optional[str]) -> str:
    default_host, default_port = ('127.0.0.1', 29500)
    if not ep:
        host, port = (default_host, default_port)
    elif ':' in ep:
        host, p = ep.split(':', 1)
        port = int(p)
    else:
        host, port = (ep, default_port)
    if port != 0 and (not _is_port_available(host, port)):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind((host, 0))
            _, free_port = s.getsockname()
        port = int(free_port)
    return f'{host}:{port}'


def _world_size(device: torch.device) -> int:
    if device.type == 'cuda':
        return torch.cuda.device_count()
    if device.type == 'xpu':
        return torch.xpu.device_count()
    return min(os.cpu_count() or 1, 4)


def _backend_type(device: torch.device) -> str:
    if device.type == 'cuda':
        return 'nccl'
    if device.type == 'xpu':
        return 'xccl'
    return 'gloo'


def _set_backend(device: torch.device) -> None:
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if device.type == 'cuda':
        torch.cuda.set_device(rank)
    elif device.type == 'xpu':
        torch.xpu.set_device(rank)
    else:
        try:
            import netifaces
            gws = netifaces.gateways()
            inet = gws['default'][netifaces.AF_INET][1]
            os.environ['GLOO_SOCKET_IFNAME'] = inet
            os.environ['TP_SOCKET_IFNAME'] = inet
        except Exception:
            pass


def _meta(memmap_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(memmap_dir, 'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _preprocess(
    data: Dict[Tuple, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple], Tuple[int, ...]]:

    def _to_tuple(x: Any) -> Any:
        if isinstance(x, tuple):
            return x
        if isinstance(x, list):
            return tuple(x)
        if isinstance(x, torch.Tensor):
            return tuple(x.flatten().detach().cpu().tolist())
        if hasattr(x, 'tolist'):
            v = x.tolist()
            return tuple(v if isinstance(v, (list, tuple)) else [v])
        return (x,)

    def _feat_row(x_tuple: Any) -> Any:
        try:
            vals = [float(v) for v in _to_tuple(x_tuple)]
        except Exception as e:
            raise TypeError(f'_preprocess: feature tuple에는 수치형 값만 와야 합니다. 문제 값={x_tuple!r}') from e
        return torch.as_tensor(vals, dtype=torch.float32)

    def _lbl(y: Any) -> Any:
        if isinstance(y, torch.Tensor):
            return y
        if hasattr(y, 'to_tensor'):
            return y.to_tensor()
        return torch.as_tensor(y)

    if isinstance(data, dict) and 'X' in data and ('Y' in data):
        x, y = (data['X'], data['Y'])
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and (x.dim() >= 2) and (y.dim() >= 2) and (x.shape[0] == y.shape[0]):
            xr, yt = (x, y)
            keys = [(int(i),) for i in range(int(x.shape[0]))]
            label_shape = tuple(yt.shape[1:])
            return (xr, yt, keys, label_shape)
        xr, yt = (_feat_row(x).unsqueeze(0), _lbl(y))
        if yt.dim() == 0 or yt.dim() == 1:
            yt = yt.unsqueeze(0)
        keys = [_to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    if isinstance(data, (tuple, list)) and len(data) >= 2:
        x, y = (data[0], data[1])
        xr = _feat_row(x).unsqueeze(0)
        yt = _lbl(y)
        if yt.dim() == 0:
            yt = yt.unsqueeze(0)
        elif yt.shape[0] != 1:
            yt = yt.unsqueeze(0)
        keys = [_to_tuple(x)]
        label_shape = tuple(yt.shape[1:])
        return (xr, yt, keys, label_shape)
    if isinstance(data, dict) and len(data) > 0:
        items = list(data.items())
        if any((isinstance(k, str) for k, _ in items)):
            raise TypeError("_preprocess: 다중 샘플 dict에서는 키가 튜플이어야 합니다. {'X':...,'Y':...}는 단일 샘플로 전달하세요.")
        keys: List[Tuple] = [_to_tuple(k) for k, _ in items]
        feats = torch.stack([_feat_row(k) for k in keys], dim=0)
        lbl_list = [_lbl(v) for _, v in items]
        if all((t.shape == lbl_list[0].shape for t in lbl_list)):
            labels = torch.stack(lbl_list, dim=0)
        else:
            labels = torch.cat([t.unsqueeze(0) for t in lbl_list], dim=0)
        label_shape = tuple(labels.shape[1:])
        return (feats, labels, keys, label_shape)
    raise ValueError('_preprocess: 지원하지 않는 입력 형식입니다. dict 또는 (X,Y)여야 합니다.')


def _postprocess(
    keys: List[Tuple],
    preds: torch.Tensor | Sequence[torch.Tensor],
) -> Dict[Tuple, torch.Tensor]:
    if isinstance(preds, torch.Tensor):
        if preds.dim() == 0:
            preds = preds.unsqueeze(0)
        if preds.shape[0] != len(keys):
            raise ValueError(f'preds batch={preds.shape[0]} != len(keys)={len(keys)}')
        rows = [preds[i].detach().cpu() for i in range(len(keys))]
    else:
        if len(preds) != len(keys):
            raise ValueError(f'len(preds)={len(preds)} != len(keys)={len(keys)}')
        rows = [p.detach().cpu() if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in preds]
    fixed_keys: List[Tuple] = []
    seen = set()
    for i, k in enumerate(keys):
        if not isinstance(k, tuple):
            try:
                k = tuple(k)
            except Exception:
                k = (k,)
        k_out = k
        if k in seen:
            k_out = k + (i,)
        seen.add(k_out)
        fixed_keys.append(k_out)
    return {k: v for k, v in zip(fixed_keys, rows)}


@torch.no_grad()
def recompute_y_stats(model: Any, loader: Any) -> None:
    dev = next(model.parameters()).device
    model.y_min.fill_(float('inf'))
    model.y_max.fill_(float('-inf'))
    model.y_sum.zero_()
    model.y_sum2.zero_()
    model.y_count.zero_()
    model.y_stats_ready.fill_(False)
    model.eval()
    for X, Y in loader:
        if Y is None:
            continue
        model.update_y_stats(Y.to(dev))
    model.finalize_y_stats()


def _ensure_uniform_param_dtype(
    model: Any,
    prefer: Optional[torch.dtype] = None,
) -> Optional[torch.dtype]:
    dtypes = set(p.dtype for p in model.parameters() if p is not None)
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
        params = getattr(mod, '_parameters', None)
        if not params:
            continue
        for name, p in list(params.items()):
            if p is None or p.dtype == tgt:
                continue
            new_p = torch.nn.Parameter(p.detach().to(tgt), requires_grad=p.requires_grad)
            setattr(mod, name, new_p)
    return tgt


def train(
    model: Model,
    data: Dict[Tuple, torch.Tensor],
    *args: Any,
    epochs: int = 5,
    batch_size: int = 128,
    val_frac: float = 0.1,
    base_lr: float = 0.001,
    weight_decay: float = 0.0001,
    warmup_ratio: float = 0.0,
    eta_min: float = 0.0,
    run_id: str = 'torch',
    seed: int = 42,
    max_nodes: int = 1,
    rdzv_backend: Optional[str] = 'c10d',
    rdzv_endpoint: Optional[str] = '127.0.0.1:29500',
    io_backend: str = 'auto',
    prefetch_factor: Optional[int] = 1,
    grad_accum_steps: int = 1,
    overlap_h2d: bool = True,
    loss_tile_dim: Optional[int] = None,
    loss_tile_size: Optional[int] = None,
    loss_mask_mode: str = 'none',
    loss_mask_value: Optional[float] = None,
    ckpt_interval_steps: Optional[int] = None,
    **kwargs: Any
) -> Model:
    feats, labels, _, label_shape = _preprocess(data)
    torch.multiprocessing.allow_connection_pickling()
    _mp_env()
    memmap_dir = _new_dir('memmap_ds')
    MemoryMappedTensorStream.materialize(
        {'features': feats, 'labels': labels},
        memmap_dir=memmap_dir,
        train_frac=1.0 - float(val_frac),
        val_frac=float(val_frac),
        shuffle=False,
    )
    ckpt_dir = _new_dir('ckpt_dcp')
    init_dir = _new_dir('init_dcp')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=pattern_to_ignore)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        save(state_dict={'model': m_sd}, storage_writer=FileSystemWriter(init_dir, sync_files=True, overwrite=True))
    _ = _establish(rdzv_endpoint)
    device = get_device()
    apply_threading_defaults()
    caps = optimal_procs()
    nprocs = caps['nproc_per_node']
    try:
        cfg_obj = getattr(model, '_Model__config', None)
    except Exception:
        cfg_obj = None
    cfg_dict: Dict[str, Any] = asdict(cfg_obj) if isinstance(cfg_obj, Config) else asdict(Config())
    lc = LaunchConfig(
        min_nodes=1,
        max_nodes=max_nodes,
        nproc_per_node=nprocs,
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=_establish(rdzv_endpoint),
        run_id=run_id,
        max_restarts=0,
        monitor_interval=5,
        start_method=_start_method(),
    )
    parameters = (
        memmap_dir,
        ckpt_dir,
        init_dir,
        int(feats.shape[1]),
        tuple(label_shape),
        cfg_dict,
        int(epochs),
        int(batch_size),
        float(val_frac),
        float(base_lr),
        float(weight_decay),
        float(warmup_ratio),
        float(eta_min),
        int(seed),
        str(io_backend),
        prefetch_factor,
        grad_accum_steps,
        overlap_h2d,
        loss_tile_dim,
        loss_tile_size,
        loss_mask_mode,
        loss_mask_value,
        ckpt_interval_steps,
    )
    elastic_launch(lc, _epochs)(*parameters)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=pattern_to_ignore)
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        m_sd = get_model_state_dict(model, options=opts)
        m_sd = _prune_dcp_state_keys(m_sd)
        load(state_dict={'model': m_sd}, storage_reader=FileSystemReader(ckpt_dir))
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
    try:
        shutil.rmtree(memmap_dir, ignore_errors=True)
    except Exception:
        pass
    try:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    except Exception:
        pass
    try:
        shutil.rmtree(init_dir, ignore_errors=True)
    except Exception:
        pass
    return model


def _epochs(
    memmap_dir: str,
    ckpt_dir: str,
    init_ckpt_dir: Optional[str],
    in_dim: int,
    out_shape: Sequence[int],
    cfg_dict: Dict[str, Any],
    epochs: int,
    batch_size: int,
    val_frac: float,
    base_lr: float,
    weight_decay: float,
    warmup_ratio: float,
    eta_min: float,
    seed: int,
    io_backend: str = 'auto',
    prefetch_factor: Optional[int] = 1,
    grad_accum_steps: int = 1,
    overlap_h2d: bool = True,
    loss_tile_dim: Optional[int] = None,
    loss_tile_size: Optional[int] = None,
    loss_mask_mode: str = 'none',
    loss_mask_value: Optional[float] = None,
    ckpt_interval_steps: Optional[int] = None,
) -> None:
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = get_device()
    _set_backend(device)
    torch.distributed.init_process_group(backend=_backend_type(device))
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    register_nvtx_flops_getter(get_total_flops)
    cfg = Config(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict or Config()
    cfg = Config(**{**asdict(cfg), 'device': device})
    model = Model(in_dim, out_shape, config=cfg)
    if init_ckpt_dir is not None and os.path.isdir(init_ckpt_dir):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=pattern_to_ignore)
            opts = StateDictOptions(full_state_dict=True, cpu_offload=False)
            m_sd = get_model_state_dict(model, options=opts)
            m_sd = _prune_dcp_state_keys(m_sd)
            load(state_dict={'model': m_sd}, storage_reader=FileSystemReader(init_ckpt_dir))
            set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))

    def _float8_log(msg: str) -> None:
        try:
            import torch.distributed as dist
            is_main = dist.is_initialized() and dist.get_rank() == 0
        except Exception:
            is_main = True
        if is_main:
            warnings.warn(msg)

    model, _, _ = Architecture.use_te_module(model, device=device)
    _ensure_uniform_param_dtype(model, prefer=torch.bfloat16 if getattr(device, 'type', None) == 'cuda' and torch.cuda.is_bf16_supported() else None)
    model, _fp8_training_enabled, _ = Architecture.enable_float8_training(model, device=device, prefer='te', logger=_float8_log)
    model.train()
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    mesh = init_device_mesh('cuda' if device.type == 'cuda' else device.type, (world,))
    match device.type:
        case 'cuda':
            param_dtype = torch.bfloat16 if is_cuda_bf16_supported() else torch.float16
            reduce_dtype = torch.float32
            cast_forward_inputs = True
        case 'xpu':
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            cast_forward_inputs = False
        case 'mps':
            param_dtype = torch.float16
            reduce_dtype = param_dtype
            cast_forward_inputs = False
        case 'cpu':
            param_dtype = torch.bfloat16 if is_cpu_bf16_supported() else torch.float32
            reduce_dtype = torch.float32
            cast_forward_inputs = False if is_cpu_bf16_supported() else True
        case _:
            param_dtype = torch.float32
            reduce_dtype = torch.float32
            cast_forward_inputs = True
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=None,
        cast_forward_inputs=cast_forward_inputs,
    )
    ignored_params: list[torch.nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            for p in module.parameters(recurse=False):
                ignored_params.append(p)
        for name in ('alpha_t', 'alpha_s', 'gem_p', 'cls_query', 'cls'):
            if hasattr(module, name):
                p = getattr(module, name)
                if isinstance(p, torch.nn.Parameter):
                    ignored_params.append(p)
    ignored_params = set(ignored_params)
    try:
        if hasattr(model, 'local') and hasattr(model.local, 'blocks'):
            for m in model.local.blocks:
                fully_shard(
                    m,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                    ignored_params=[param for param in m.parameters(recurse=True) if param in ignored_params],
                ).set_requires_gradient_sync(True)
        global_net = None
        if hasattr(model, 'global'):
            maybe_global = getattr(model, 'global')
            if hasattr(maybe_global, 'blocks'):
                global_net = maybe_global
        elif hasattr(model, '_global') and hasattr(model._global, 'blocks'):
            global_net = model._global
        if global_net is not None:
            for m in global_net.blocks:
                fully_shard(
                    m,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                    ignored_params=[param for param in m.parameters(recurse=True) if param in ignored_params],
                ).set_requires_gradient_sync(True)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False, ignored_params=ignored_params).set_requires_gradient_sync(True)
    except Exception:
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, ignored_params=ignored_params, reshard_after_forward=False).set_requires_gradient_sync(True)
    net_params = [p for p in model.parameters()]
    optimizer = AdamW.float(net_params, lr=base_lr, weight_decay=weight_decay, use_fp8=device.type == 'cuda', use_foreach=False, use_fused=False, logger=None)

    def _dl_ckpt(p: Any) -> Any:
        return os.path.join(p, 'dataloader.json')

    meta = _meta(memmap_dir)
    train_loader0, val_loader0, keep0 = stream(
        memmap_dir=memmap_dir,
        device=device,
        batch_size=batch_size,
        val_frac=val_frac,
        prefetch_factor=prefetch_factor,
        non_blocking_copy=True,
        seed=seed,
        shuffle=True,
    )
    train_steps = len(train_loader0)
    val_steps = len(val_loader0) if val_loader0 is not None else 0
    steps_per_epoch = max(1, train_steps + val_steps)

    def _as_tensor(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, 'to_tensor'):
            return obj.to_tensor()
        return torch.as_tensor(obj)

    # bootstrap running stats from first pass
    for _step_idx, _raw in enumerate(train_loader0):
        _feat0, _label0, *_ = _preprocess(_raw)
        if hasattr(model, 'update_x_stats'):
            try:
                model.update_x_stats(_feat0)
            except Exception:
                model.update_x_stats(_as_tensor(_feat0).detach().cpu())
        _label0 = _as_tensor(_label0)
        _Y0_flat = _label0.view(_label0.shape[0], -1)
        model.update_y_stats(_Y0_flat)
    model.finalize_y_stats()
    if hasattr(model, 'finalize_x_stats'):
        model.finalize_x_stats()
    _t = StudentsTLoss(confidence=0.99, metric='t_value', two_tailed=True, df=4, mu_mode='error', std_mode='pooled', ddof=1, clamp_max=8.0, detach_stats=True, dim=-1, reduction='none')
    _z = StandardNormalLoss(confidence=0.99, metric='z_value', two_tailed=True, penalty='softplus', tau=1.0, mu_mode='error', std_mode='pooled', ddof=1, clamp_max=8.0, detach_stats=True, dim=-1, reduction='none')
    top_loss = TiledLoss(_t, mask_mode=loss_mask_mode, mask_value=loss_mask_value, tile_dim=loss_tile_dim, tile_size=loss_tile_size, reduction='mean')
    bottom_loss = TiledLoss(_z, mask_mode=loss_mask_mode, mask_value=loss_mask_value, tile_dim=loss_tile_dim, tile_size=loss_tile_size, reduction='mean')
    loss_weights = (0.5, 0.5)
    if keep0 is not None:
        keep0.cleanup()
    total_steps = epochs * steps_per_epoch
    if warmup_ratio > 0.0:
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        main_steps = max(1, total_steps - warmup_steps)
    else:
        warmup_steps = 0
        main_steps = max(1, total_steps)
    base = float(base_lr)
    emin = float(eta_min)
    start_factor = 0.001

    def _scheduler(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return start_factor + (1.0 - start_factor) * (step / max(1, warmup_steps))
        t = step - warmup_steps
        frac_min = emin / base if base > 0.0 else 0.0
        return frac_min + (1.0 - frac_min) * 0.5 * (1.0 + math.cos(math.pi * t / max(1, main_steps)))

    def _join_context(m: torch.nn.Module) -> contextlib.AbstractContextManager:
        try:
            is_joinable = hasattr(m, 'join_hook')
        except Exception:
            is_joinable = False
        return Join([m], throw_on_early_termination=True) if Join is not None and is_joinable else contextlib.nullcontext()

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_scheduler)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda' and (not torch.cuda.is_bf16_supported()))
    status_bar = _status_bar('Training', total_steps, device) if local_rank == 0 else None
    use_timer = device.type in ('cuda', 'xpu', 'mps') and hasattr(torch, 'Event')

    for epoch in range(epochs):
        io_time = torch.tensor(0.0, device=device, dtype=torch.float64)
        comp_time = torch.tensor(0.0, device=device, dtype=torch.float64)
        io_bytes = torch.tensor(0.0, device=device, dtype=torch.float64)
        flops = torch.tensor(0.0, device=device, dtype=torch.float64)
        train_loader, val_loader, keep = stream(
            memmap_dir=memmap_dir,
            device=device,
            batch_size=batch_size,
            val_frac=val_frac,
            prefetch_factor=prefetch_factor,
            non_blocking_copy=True,
            seed=seed,
            shuffle=True,
        )
        try:
            dl_state_path = os.path.join(ckpt_dir, 'dataloader.json') if os.path.isfile(os.path.join(ckpt_dir, 'dataloader.json')) else (
                os.path.join(init_ckpt_dir, 'dataloader.json') if (init_ckpt_dir and os.path.isfile(os.path.join(init_ckpt_dir, 'dataloader.json'))) else None
            )
            if dl_state_path:
                with open(dl_state_path, 'r', encoding='utf-8') as _f:
                    _dl = json.load(_f)
                state_train = _dl.get('train', {})
                state_val = _dl.get('val', {})
                with contextlib.suppress(Exception):
                    train_loader.load_state_dict(state_train)
                if val_loader is not None:
                    with contextlib.suppress(Exception):
                        val_loader.load_state_dict(state_val)
        except Exception:
            pass
        with register_flop_hooks(model, mode='train'):
            model.train()
            running = torch.zeros((), device=device, dtype=torch.float32)
            n_batches = 0
            train_steps_t = torch.tensor(train_steps, device=device, dtype=torch.int64)
            val_steps_t = torch.tensor(val_steps, device=device, dtype=torch.int64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(train_steps_t, op=torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(val_steps_t, op=torch.distributed.ReduceOp.MIN)
            train_steps = int(train_steps_t.item())
            val_steps = int(val_steps_t.item())
            steps_per_epoch = max(1, train_steps + val_steps)
            total_steps = epochs * steps_per_epoch
            if local_rank == 0 and status_bar is not None:
                status_bar.total = total_steps
            t_fetch_start = time.perf_counter_ns()
            with _join_context(model):
                for step_idx, _raw in enumerate(train_loader):
                    feat, label, *_ = _preprocess(_raw)
                    X = feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
                    X = torch.atleast_2d(X)
                    if X.dim() != 2:
                        raise RuntimeError(f'features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}')
                    if X.shape[1] != in_dim:
                        raise RuntimeError(f'feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}')
                    Y = label if isinstance(label, torch.Tensor) else torch.as_tensor(label)
                    t_ready = time.perf_counter_ns()
                    if use_timer and getattr(device, 'type', None) == 'cuda':
                        h2d_start = torch.Event(device=device, enable_timing=True)
                        h2d_end = torch.Event(device=device, enable_timing=True)
                        h2d_start.record()
                        if getattr(X, 'device', None).type == 'cpu':
                            X = X.to(device, non_blocking=True)
                        if getattr(Y, 'device', None).type == 'cpu':
                            Y = Y.to(device, non_blocking=True)
                        h2d_end.record()
                        h2d_end.synchronize()
                        h2d_s = float(h2d_start.elapsed_time(h2d_end)) / 1000.0
                    else:
                        t_h2d_start = time.perf_counter_ns()
                        X = X.to(device, non_blocking=True)
                        Y = Y.to(device, non_blocking=True)
                        t_h2d_end = time.perf_counter_ns()
                        h2d_s = (t_h2d_end - t_h2d_start) / 1_000_000_000.0
                    wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                    io_time += torch.tensor(wait_s + h2d_s, device=device, dtype=torch.float64)
                    io_bytes += torch.tensor(X.element_size() * X.nelement() + Y.element_size() * Y.nelement(), device=device, dtype=torch.float64)
                    if use_timer:
                        ev_start = torch.Event(device=device, enable_timing=True)
                        ev_end = torch.Event(device=device, enable_timing=True)
                    else:
                        t_comp_start = time.perf_counter_ns()
                    if step_idx % max(1, grad_accum_steps) == 0:
                        optimizer.zero_grad(set_to_none=True)
                    with FlopCounterMode(display=False) as fcm_step, NVTXCounterMode(device=device) as nvtx_step:
                        if use_timer:
                            ev_start.record()
                        with Autocast.float(device):
                            Y_flat = Y.reshape(Y.shape[0], -1).to(device, dtype=next(model.parameters()).dtype)
                            _, loss_val = model(
                                X,
                                labels_flat=Y_flat,
                                global_loss=top_loss,
                                local_loss=bottom_loss,
                                loss_weights=loss_weights,
                            )
                            if loss_val is None:
                                raise RuntimeError('loss is None')
                            loss = loss_val
                            running += loss.detach().to(running.dtype)
                            n_batches += 1
                        _acc_last = (step_idx + 1) % max(1, grad_accum_steps) == 0
                        with fsdp_no_sync(model, enable=grad_accum_steps > 1 and (not _acc_last)):
                            if scaler.is_enabled():
                                scaler.scale(loss / max(1, grad_accum_steps)).backward()
                            else:
                                (loss / max(1, grad_accum_steps)).backward()
                    if use_timer:
                        ev_end.record()
                        ev_end.synchronize()
                        comp_ms = ev_start.elapsed_time(ev_end)
                        comp_time += torch.tensor(float(comp_ms) / 1000.0, dtype=torch.float64, device=device)
                    else:
                        comp_time += torch.tensor((time.perf_counter_ns() - t_comp_start) / 1_000_000_000.0, device=device, dtype=torch.float64)
                    fcm_flops = float(fcm_step.get_total_flops())
                    try:
                        _nvtx_soft_add(fcm_flops)
                    except Exception:
                        pass
                    nvtx_flops = float(nvtx_step.get_total_flops()) if getattr(device, 'type', None) == 'cuda' else 0.0
                    step_flops = max(fcm_flops, nvtx_flops, get_total_flops(reset=True))
                    flops += torch.tensor(step_flops, device=device, dtype=torch.float64)
                    do_step = (step_idx + 1) % max(1, grad_accum_steps) == 0
                    if do_step:
                        max_grad_norm = getattr(cfg, 'max_grad_norm', None)
                        if max_grad_norm is not None:
                            if scaler.is_enabled():
                                with contextlib.suppress(Exception):
                                    scaler.unscale_(optimizer)
                            with contextlib.suppress(Exception):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        sched.step()
                    if _fp8_training_enabled:
                        with contextlib.suppress(Exception):
                            precompute_float8_dynamic_scale_for_fsdp(model)
                    if local_rank == 0 and status_bar is not None:
                        status_bar.update(1)
                    t_fetch_start = time.perf_counter_ns()
                    if ckpt_interval_steps is not None and (step_idx + 1) % ckpt_interval_steps == 0 and (local_rank == 0):
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message=pattern_to_ignore)
                            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
                            m_sd = get_model_state_dict(model, options=opts)
                            o_sd = get_optimizer_state_dict(optimizer)
                            if torch.distributed.is_initialized():
                                torch.distributed.barrier(device_ids=[local_rank] if device.type in ('cuda', 'xpu') else None)
                            save(state_dict={'model': m_sd, 'optimizer': o_sd}, storage_writer=FileSystemWriter(ckpt_dir, sync_files=True, overwrite=True))
                            if torch.distributed.is_initialized():
                                torch.distributed.barrier(device_ids=[local_rank] if device.type in ('cuda', 'xpu') else None)
                            with contextlib.suppress(Exception):
                                _dl = {'train': train_loader.state_dict(), 'val': val_loader.state_dict() if val_loader is not None else {}}
                                with open(_dl_ckpt(ckpt_dir), 'w', encoding='utf-8') as _f:
                                    json.dump(_dl, _f)
                                if torch.distributed.is_initialized():
                                    torch.distributed.barrier(device_ids=[local_rank] if device.type in ('cuda', 'xpu') else None)
        if val_loader is not None:
            with register_flop_hooks(model, mode='eval'):
                model.eval()
                s = torch.zeros((), device=device, dtype=torch.float32)
                m = 0
                with torch.no_grad(), Autocast.float(device):
                    t_fetch_start = time.perf_counter_ns()
                    with _join_context(model):
                        v_step_count = 0
                        for step_idx, _raw in enumerate(val_loader):
                            feat, label, *_ = _preprocess(_raw)
                            X = feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
                            X = torch.atleast_2d(X)
                            if X.dim() != 2:
                                raise RuntimeError(f'features.ndim={X.dim()} (expect 2). got shape={tuple(X.shape)}')
                            if X.shape[1] != in_dim:
                                raise RuntimeError(f'feature dim mismatch: X.shape[1]={X.shape[1]} != in_dim={in_dim}')
                            Y = label if isinstance(label, torch.Tensor) else torch.as_tensor(label)
                            t_ready = time.perf_counter_ns()
                            if use_timer and getattr(device, 'type', None) == 'cuda':
                                h2d_start = torch.Event(device=device, enable_timing=True)
                                h2d_end = torch.Event(device=device, enable_timing=True)
                                h2d_start.record()
                                X = X.to(device, non_blocking=True)
                                Y = Y.to(device, non_blocking=True)
                                h2d_end.record()
                                h2d_end.synchronize()
                                h2d_s = float(h2d_start.elapsed_time(h2d_end)) / 1000.0
                            else:
                                t_h2d_start = time.perf_counter_ns()
                                X = X.to(device, non_blocking=True)
                                Y = Y.to(device, non_blocking=True)
                                t_h2d_end = time.perf_counter_ns()
                                h2d_s = (t_h2d_end - t_h2d_start) / 1_000_000_000.0
                            v_step_count = step_idx + 1
                            wait_s = (t_ready - t_fetch_start) / 1_000_000_000.0
                            io_time += torch.tensor(wait_s + h2d_s, device=device, dtype=torch.float64)
                            io_bytes += torch.tensor(X.element_size() * X.nelement() + Y.element_size() * Y.nelement(), device=device, dtype=torch.float64)
                            if use_timer:
                                ev_start = torch.Event(device=device, enable_timing=True)
                                ev_end = torch.Event(device=device, enable_timing=True)
                                ev_start.record()
                            else:
                                t_comp_start = time.perf_counter_ns()
                            with FlopCounterMode(display=False) as fcm_val, NVTXCounterMode(device=device) as nvtx_val:
                                Yv_flat = Y.reshape(Y.shape[0], -1).to(device, dtype=next(model.parameters()).dtype)
                                _, loss_val = model(X, labels_flat=Yv_flat, global_loss=top_loss, local_loss=bottom_loss, loss_weights=loss_weights)
                            if use_timer:
                                ev_end.record()
                                ev_end.synchronize()
                                comp_time += torch.tensor(float(ev_start.elapsed_time(ev_end)) / 1000.0, device=device, dtype=torch.float64)
                            else:
                                comp_time += torch.tensor((time.perf_counter_ns() - t_comp_start) / 1_000_000_000.0, device=device, dtype=torch.float64)
                            try:
                                fcm_vflops = float(fcm_val.get_total_flops())
                            except Exception:
                                fcm_vflops = 0.0
                            with contextlib.suppress(Exception):
                                _nvtx_soft_add(fcm_vflops)
                            nvtx_vflops = float(nvtx_val.get_total_flops()) if getattr(device, 'type', None) == 'cuda' else 0.0
                            v_step_flops = max(fcm_vflops, nvtx_vflops, get_total_flops(reset=True))
                            flops += torch.tensor(v_step_flops, device=device, dtype=torch.float64)
                            s += loss_val.detach().to(s.dtype)
                            m += 1
                            t_fetch_start = time.perf_counter_ns()
                with contextlib.suppress(Exception):
                    val_steps = int(v_step_count)
        if keep is not None:
            keep.cleanup()
        gc.collect()
        torch.distributed.barrier(device_ids=[local_rank] if device.type in ('cuda', 'xpu') else None)
        torch.distributed.all_reduce(comp_time, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(io_time, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(flops, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(io_bytes, op=torch.distributed.ReduceOp.SUM)
        comp_time = comp_time / max(1, world)
        io_time = io_time / max(1, world)
        flops = flops / max(1, world)
        io_bytes = io_bytes / max(1, world)
        mbps_t = io_bytes / io_time.clamp_min(1e-06) / 1_000_000.0
        tflops_t = flops / comp_time.clamp_min(1e-06) / 1_000_000_000_000.0
        if local_rank == 0 and status_bar is not None:
            mbps = float(mbps_t)
            tflops = float(tflops_t)
            substr = [
                f'I/O = {mbps:.2f} MB/s' if mbps >= 0.01 else 'I/O < 0.01 MB/s',
                f'COM = {tflops:.2f} TFLOPS' if tflops >= 0.01 else 'COM < 0.01 TFLOPS',
            ]
            unit_str = ', '.join(substr)
            status_bar.unit = unit_str
            status_bar.refresh()
    if local_rank == 0:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=pattern_to_ignore)
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_sd = get_model_state_dict(model, options=opts)
            optim_sd = get_optimizer_state_dict(optimizer)
            writer = FileSystemWriter(ckpt_dir, sync_files=True, overwrite=True)
            save(state_dict={'model': model_sd, 'optimizer': optim_sd}, storage_writer=writer)
        with contextlib.suppress(Exception):
            _dl = {'train': train_loader.state_dict(), 'val': val_loader.state_dict() if val_loader is not None else {}}
            with open(os.path.join(ckpt_dir, 'dataloader.json'), 'w', encoding='utf-8') as _f:
                json.dump(_dl, _f)
    torch.distributed.barrier(device_ids=[local_rank] if device.type in ('cuda', 'xpu') else None)
    with contextlib.suppress(Exception):
        if local_rank == 0 and status_bar is not None:
            status_bar.close()
    torch.distributed.destroy_process_group()


def predict(
    model: Model,
    data: Dict[Tuple, torch.Tensor],
    *args: Any,
    batch_size: int = 512,
    seed: int = 7,
    io_backend: str = 'auto',
    prefetch_factor: Optional[int] = 1,
    **kwargs: Any
) -> Dict[Tuple, torch.Tensor]:
    _mp_env()
    tmp_dir = _new_dir('infer')
    dcp_dir = os.path.join(tmp_dir, 'dcp')
    memmap_dir = os.path.join(tmp_dir, 'memmap')
    device = get_device()
    torch.multiprocessing.allow_connection_pickling()
    nprocs = _world_size(device) if device.type in ('cuda', 'xpu') else 1
    rank_hint = int(os.environ.get('LOCAL_RANK', 0))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=pattern_to_ignore)
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            m_sd = get_model_state_dict(model, options=opts)
            if torch.distributed.is_initialized():
                torch.distributed.barrier(device_ids=[rank_hint] if device.type in ('cuda', 'xpu') else None)
            save(state_dict={'model': m_sd}, storage_writer=FileSystemWriter(dcp_dir, sync_files=True, overwrite=True))
            if torch.distributed.is_initialized():
                torch.distributed.barrier(device_ids=[rank_hint] if device.type in ('cuda', 'xpu') else None)
        try:
            cfg_obj = getattr(model, '_Model__config', None)
        except Exception:
            cfg_obj = None
        cfg_dict = asdict(cfg_obj) if isinstance(cfg_obj, Config) else asdict(Config())
        # fill missing labels with zeros of the correct shape for pipeline consistency
        if any(v is None for v in data.values()):
            dummy_shape = tuple(model.out_shape)
            data = {k: torch.zeros(dummy_shape) if v is None else torch.as_tensor(v).view(*dummy_shape) for k, v in data.items()}
        feats, labels, keys, label_shape = _preprocess(data)
        MemoryMappedTensorStream.materialize({'features': feats, 'labels': labels}, memmap_dir=memmap_dir, train_frac=1.0, val_frac=0.0, shuffle=False)
        manager = multiprocessing.Manager()
        ret_dict = manager.dict()
        torch.multiprocessing.start_processes(
            _infer,
            args=(ret_dict, dcp_dir, memmap_dir, int(feats.shape[1]), tuple(label_shape), cfg_dict, keys, int(batch_size), seed, io_backend, prefetch_factor),
            nprocs=nprocs,
            join=True,
            daemon=False,
            start_method=_start_method(),
        )
        out: Dict[Tuple, torch.Tensor] = dict(ret_dict)
        return out
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _infer(
    local_rank: int,
    ret_dict: Dict[Any, Any],
    model_ckpt_dir: Optional[str],
    memmap_dir: str,
    in_dim: int,
    out_shape: Sequence[int],
    cfg_dict: Dict[str, Any],
    keys: List[Tuple],
    batch_size: int = 512,
    seed: int = 7,
    io_backend: str = 'auto',
    prefetch_factor: Optional[int] = 1,
) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % max(1, torch.cuda.device_count()))
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.set_device(local_rank % max(1, torch.xpu.device_count()))
    except Exception:
        pass
    device = get_device()
    cfg = Config(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict or Config()
    model = Model(in_dim, out_shape, config=cfg)
    if model_ckpt_dir is not None and os.path.isdir(model_ckpt_dir):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=pattern_to_ignore)
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            m_sd = get_model_state_dict(model, options=opts)
            m_sd = _prune_dcp_state_keys(m_sd)
            load(state_dict={'model': m_sd}, storage_reader=FileSystemReader(model_ckpt_dir))
            set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
    model.to(device, non_blocking=True).eval()

    def _float8_log(msg: str) -> None:
        warnings.warn(msg)

    model, _, _ = Architecture.use_te_module(model, device=device)
    _ensure_uniform_param_dtype(model, prefer=torch.bfloat16 if getattr(device, 'type', None) == 'cuda' and torch.cuda.is_bf16_supported() else None)
    model, _, _ = Architecture.enable_float8_prediction(model, device=device, prefer='te', logger=_float8_log, dynamic_activations=True)
    model.eval()
    train_loader, _, keep = stream(
        memmap_dir=memmap_dir,
        device=device,
        batch_size=batch_size,
        val_frac=0.0,
        prefetch_factor=prefetch_factor,
        non_blocking_copy=True,
        seed=seed,
        shuffle=False,
    )
    status_bar = _status_bar('Prediction', len(train_loader), device)
    with register_flop_hooks(model, mode='eval'):
        io_bytes: float = 0.0
        io_time: float = 0.0
        comp_time: float = 0.0
        total_flops: float = 0.0
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        register_nvtx_flops_getter(get_total_flops)
        use_timer = device.type in ('cuda', 'xpu', 'mps') and hasattr(torch, 'Event')
        t_fetch_start = time.perf_counter_ns()
        preds: List[torch.Tensor] = []
        with torch.no_grad(), Autocast.float(device):
            for _step_idx, _raw in enumerate(train_loader):
                feat, _label, *_ = _preprocess(_raw)
                X = feat if isinstance(feat, torch.Tensor) else torch.as_tensor(feat)
                X = torch.atleast_2d(X)
                if X.dim() != 2:
                    raise RuntimeError(f'_infer: feats.ndim={X.dim()} (expect 2), shape={tuple(X.shape)}')
                if X.shape[1] != int(in_dim):
                    raise AssertionError(f'_infer: feature dim mismatch — feats.shape[1]={X.shape[1]} != in_dim={in_dim}.')
                if X.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    X = X.to(dtype=torch.float32)
                if use_timer and getattr(device, 'type', None) == 'cuda':
                    ev_h2d_s = torch.Event(device=device, enable_timing=True)
                    ev_h2d_e = torch.Event(device=device, enable_timing=True)
                    ev_h2d_s.record()
                    X = X.to(device, non_blocking=True)
                    ev_h2d_e.record()
                    ev_h2d_e.synchronize()
                    h2d_s = float(ev_h2d_s.elapsed_time(ev_h2d_e)) / 1000.0
                else:
                    t_h2d_s = time.perf_counter_ns()
                    X = X.to(device, non_blocking=True)
                    t_h2d_e = time.perf_counter_ns()
                    h2d_s = (t_h2d_e - t_h2d_s) / 1_000_000_000.0
                wait_s = (time.perf_counter_ns() - t_fetch_start) / 1_000_000_000.0
                io_time += wait_s + h2d_s
                with contextlib.suppress(Exception):
                    io_bytes += float(X.element_size() * X.nelement())
                if use_timer and getattr(device, 'type', None) == 'cuda':
                    ev_start = torch.Event(device=device, enable_timing=True)
                    ev_end = torch.Event(device=device, enable_timing=True)
                    ev_start.record()
                else:
                    t0 = time.perf_counter_ns()
                with FlopCounterMode(display=False) as fcm_val, NVTXCounterMode(device=device) as nvtx_val:
                    with contextlib.suppress(Exception):
                        mark_step = getattr(getattr(torch, 'compiler', None), 'cudagraph_mark_step_begin', None)
                        if callable(mark_step):
                            mark_step()
                    y_hat_out, _ = model(X, labels_flat=None, global_loss=None, local_loss=None, loss_weights=None)
                preds.append(y_hat_out.detach().cpu())
                if use_timer and getattr(device, 'type', None) == 'cuda':
                    ev_end.record()
                    ev_end.synchronize()
                    comp_time += float(ev_start.elapsed_time(ev_end)) / 1000.0
                else:
                    t1 = time.perf_counter_ns()
                    comp_time += (t1 - t0) / 1_000_000_000.0
                try:
                    fcm_flops = float(fcm_val.get_total_flops())
                except Exception:
                    fcm_flops = 0.0
                try:
                    nvtx_flops = float(nvtx_val.get_total_flops()) if getattr(device, 'type', None) == 'cuda' else 0.0
                except Exception:
                    nvtx_flops = 0.0
                try:
                    step_flops = max(fcm_flops, nvtx_flops, get_total_flops(reset=True))
                except Exception:
                    step_flops = max(fcm_flops, nvtx_flops)
                total_flops += max(0.0, step_flops)
                mbps = io_bytes / max(io_time, 1e-06) / 1_000_000.0
                tflops = total_flops / max(comp_time, 1e-06) / 1_000_000_000_000.0
                status_bar.unit = (f'I/O = {mbps:.2f} MB/s' if mbps >= 0.01 else 'I/O < 0.01 MB/s') + ', ' + (f'COM = {tflops:.2f} TFLOPS' if tflops >= 0.01 else 'COM < 0.01 TFLOPS')
                status_bar.update(1)
                t_fetch_start = time.perf_counter_ns()
    with contextlib.suppress(Exception):
        status_bar.close()
    flat = torch.cat(preds, dim=0)
    pred_struct = Model.unflatten_labels(flat, out_shape)
    ret = _postprocess(keys, pred_struct)
    ret_dict.update(ret)
    if keep is not None:
        keep.cleanup()