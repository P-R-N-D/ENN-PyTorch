# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from tensordict import TensorDict, TensorDictBase, stack

from ..backend.system import get_tlb

try:
    from torchdata.nodes import BaseNode, Loader as _Loader
except Exception:
    from torchdata.nodes import BaseNode, Loader as _Loader

from .nodes import (
    Dataset,
    Disposable,
    Connector,
    Loader,
    Multiplexer,
    Sampler,
    SourceSpec,
)
from typing import Mapping as _Mapping


def _is_source_spec(obj: Any) -> bool:
    if not isinstance(obj, _Mapping):
        return False
    if "kind" not in obj or "path" not in obj:
        return False
    p = obj.get("path")
    try:
        os.fspath(p)
    except Exception:
        return False
    return True


def dataset(
    source: SourceSpec,
    *args: Any,
    split: str = "train",
    val_frac: float = 0.0,
    **kwargs: Any,
) -> "Dataset":
    kind = str(source.get("kind"))
    if kind != "memmap":
        raise ValueError(f"Unsupported source kind: {kind!r}")
    path = os.fspath(source.get("path", ""))
    if not path:
        raise ValueError("SourceSpec['path'] must be provided")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"memmap directory not found: {path!r}")
    sp = str(split or "train")
    if sp not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got: {sp!r}")
    vf = float(val_frac)
    if not (0.0 <= vf <= 1.0):
        raise ValueError(f"val_frac must be in [0,1], got: {vf}")
    return Dataset(path, split=sp, val_frac=vf)

def _process(
    batch: Mapping[str, Any],
    *args: Any,
    flatten_features: bool,
    labels_dtype: Optional[torch.dtype],
    sanitize: bool,
    **kwargs: Any,
) -> Dict[str, Any]:
    features = batch["X"]
    labels = batch["Y"]
    if (
        flatten_features
        and isinstance(features, torch.Tensor)
        and (features.dim() >= 2)
    ):
        features = features.flatten(start_dim=1)
    if labels_dtype is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(dtype=labels_dtype, non_blocking=True)
    if sanitize and torch.is_floating_point(labels):
        labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
    return {"X": features, "Y": labels}


def collate(
    sample: Any,
    *args: Any,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Any:
    converter = partial(
        _process,
        flatten_features=flatten_features,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
    )
    if isinstance(sample, TensorDictBase):
        return sample
    if isinstance(sample, (list, tuple)):
        batch_list = []
        for item in sample:
            if isinstance(item, TensorDictBase):
                batch_list.append(item); continue
            if isinstance(item, Mapping):
                conv = converter(item)
                td = TensorDict(
                    {"X": conv["X"], "Y": conv["Y"], "features": conv["X"], "labels": conv["Y"]},
                    batch_size=[],
                )
                batch_list.append(td)
            else:
                batch_list.append(item)
        if all(isinstance(elem, TensorDictBase) for elem in batch_list):
            return stack(batch_list, dim=0)
        return batch_list
    if isinstance(sample, Mapping):
        conv = converter(sample)
        return TensorDict({"X": conv["X"], "Y": conv["Y"], "features": conv["X"], "labels": conv["Y"]}, batch_size=[])
    return sample

def _env_optimize_threads() -> Dict[str, int]:
    try:
        workers = max(1, int(os.environ.get("STNET_WORKERS", "4")))
    except Exception:
        workers = 4
    try:
        pfetch = max(1, int(os.environ.get("STNET_PREFETCH", "8")))
    except Exception:
        pfetch = 8
    return {"dataloader_workers": workers, "prefetch_factor": pfetch}


def compose(
    node_or_nodes: Union[BaseNode, Sequence[BaseNode], Mapping[str, BaseNode]],
    *args: Any,
    device: Union[str, torch.device],
    map_fn: Callable[[Any], Any],
    prefetch_factor: int,
    non_blocking_copy: bool,
    io_workers: int,
    prebatch: int,
    **kwargs: Any,
) -> Tuple[BaseNode, BaseNode, BaseNode]:
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    try:
        get_tlb(io_workers=io_workers)
    except Exception:
        pass

    mux = Multiplexer(
        stop_criteria=os.environ.get("STNET_MULTINODE_STOP", "ALL_DATASETS_EXHAUSTED"),
        seed=int(os.environ.get("STNET_BLOCK_SEED", "0") or "0"),
    )
    source = mux.compose(node_or_nodes)

    mapper = Connector(
        map_fn=map_fn,
        io_workers=io_workers,
        prebatch=prebatch,
        prefetch_factor=prefetch_factor,
        device=device_obj,
        non_blocking=bool(non_blocking_copy),
    )
    mapped = mapper.compose(source)
    return source, mapped, mapped


def fetch(
    sources: Union[
        SourceSpec,
        Sequence[SourceSpec],
        Mapping[str, SourceSpec],
    ],
    device: Union[str, torch.device],
    batch_size: int,
    val_frac: float,
    *args: Any,
    prefetch_factor: int = 2,
    non_blocking_copy: bool = True,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = False,
    flatten_features: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any], Disposable]:
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    threads = _env_optimize_threads()
    io_workers = max(1, int(threads.get("dataloader_workers", 2)))
    prebatch = max(1, int(threads.get("prefetch_factor", 2)))

    map_fn = partial(
        collate,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )

    allocated = Disposable()

    def _node_for(spec: SourceSpec, split: str, shuffle: bool) -> Tuple[BaseNode, int]:
        ds = dataset(spec, split=split, val_frac=float(val_frac))
        allocated.add(ds)
        samp = Sampler(start=ds.start, end=ds.end, batch_size=int(batch_size), shuffle=shuffle, seed=int(os.environ.get("STNET_SAMPLER_SEED","0") or "0"))
        node = samp.compose(ds)
        return node, len(samp)

    if isinstance(sources, Mapping) and not _is_source_spec(sources):
        nodes_map: Dict[str, BaseNode] = {}
        lengths: Dict[str, int] = {}
        for key, spec in sources.items():
            node, length = _node_for(spec, split="train", shuffle=True)
            nodes_map[str(key)] = node
            lengths[str(key)] = length
        _, mapped, _ = compose(nodes_map, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
        train_length = sum(lengths.values()) if lengths else None
    elif isinstance(sources, (list, tuple)):
        nodes_list: list[BaseNode] = []
        lengths: list[int] = []
        for spec in sources:
            node, length = _node_for(spec, split="train", shuffle=True)
            nodes_list.append(node); lengths.append(length)
        _, mapped, _ = compose(nodes_list, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
        train_length = sum(lengths) if lengths else None
    else:
        node, length = _node_for(sources, split="train", shuffle=True)
        _, mapped, _ = compose(node, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
        train_length = length

    train_loader = Loader(device=device_obj, node=mapped, prefetch_factor=int(prefetch_factor), non_blocking=bool(non_blocking_copy), length=train_length)

    val_loader = None
    if float(val_frac) > 0.0:
        if isinstance(sources, Mapping) and not _is_source_spec(sources):
            nodes_map = {}
            lengths = {}
            for key, spec in sources.items():
                node, length = _node_for(spec, split="val", shuffle=False)
                nodes_map[str(key)] = node; lengths[str(key)] = length
            _, vmapped, _ = compose(nodes_map, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
            val_len = sum(lengths.values()) if lengths else None
        elif isinstance(sources, (list, tuple)):
            nodes_list = []
            lengths = []
            for spec in sources:
                node, length = _node_for(spec, split="val", shuffle=False)
                nodes_list.append(node); lengths.append(length)
            _, vmapped, _ = compose(nodes_list, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
            val_len = sum(lengths) if lengths else None
        else:
            node, length = _node_for(sources, split="val", shuffle=False)
            _, vmapped, _ = compose(node, device=device_obj, map_fn=map_fn, prefetch_factor=int(prefetch_factor), non_blocking_copy=bool(non_blocking_copy), io_workers=io_workers, prebatch=prebatch)
            val_len = length
        val_loader = Loader(device=device_obj, node=vmapped, prefetch_factor=int(prefetch_factor), non_blocking=bool(non_blocking_copy), length=val_len)

    return (train_loader, val_loader, allocated)