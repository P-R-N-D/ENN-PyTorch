# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from tensordict import TensorDict, TensorDictBase, stack

from ..backend.system import get_tlb, optimize_threads

try:
    from torchdata.nodes import BaseNode, Loader as _Loader, ParallelMapper
except Exception:
    from torchdata.nodes import BaseNode, Loader as _Loader
    ParallelMapper = None

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

def compose(
    node_or_nodes: Union[BaseNode, Sequence[BaseNode], Mapping[str, BaseNode]],
    *args: Any,
    device: Union[str, torch.device],
    map_fn: Callable[[Any], Any],
    prefetch_factor: int,
    non_blocking_copy: bool,
    io_workers: int,
    prebatch: int,
    weights: Optional[Mapping[str, float]] = None,
    **kwargs: Any,
) -> Tuple[BaseNode, BaseNode, BaseNode]:
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    try:
        get_tlb(io_workers=io_workers)
    except Exception:
        pass

    mx_weights = None
    if isinstance(node_or_nodes, Mapping) and isinstance(weights, Mapping):
        mx_weights = weights

    mux = Multiplexer(stop_criteria="CYCLE_FOREVER", seed=0, weights=mx_weights)
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
    non_blocking_copy: bool = True,
    labels_dtype: Optional[torch.dtype] = None,
    sanitize: bool = True,
    flatten_features: bool = True,
    train_weights: Optional[Mapping[str, float]] = None,
    val_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Any, Optional[Any], Disposable]:
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device

    # Use system-level optimizer hints (sets torch threads as well)
    hints = optimize_threads()
    io_workers = max(1, int(hints.get("num_workers", 1)))
    prebatch = max(1, int(hints.get("prebatch", max(1, io_workers * 2))))
    pf_depth = max(1, int(hints.get("prefetch_factor", 1)))

    map_fn = partial(
        collate,
        labels_dtype=labels_dtype,
        sanitize=sanitize,
        flatten_features=flatten_features,
    )

    allocated = Disposable()

    def _node_for(spec: SourceSpec, split: str, shuffle: bool) -> Tuple[BaseNode, int, Any]:
        """
        각 source에 대해 여기서는 Dataset + SamplerWrapper까지만 만든다.
        이후 멀티 소스 믹싱과 배치 구성은 compose(...)에서 일괄 처리.

        반환:
          - sampler_node: Sampler.compose(ds) → SamplerWrapper(self)
          - length: sampler의 길이
          - ds: 해당 torch.utils.data.Dataset (나중에 __getitem__에 사용)
        """
        ds = dataset(spec, split=split, val_frac=float(val_frac))
        allocated.add(ds)

        samp = Sampler(
            start=ds.start,
            end=ds.end,
            batch_size=int(batch_size),
            shuffle=shuffle,
            seed=0,
        )

        sampler_node = samp.compose(ds)  # SamplerWrapper(self)
        return sampler_node, len(samp), ds

    # --- train loader 구성 ---
    if isinstance(sources, Mapping) and not _is_source_spec(sources):
        sampler_nodes: Dict[str, BaseNode] = {}
        lengths: Dict[str, int] = {}
        datasets: Dict[str, Any] = {}
        for key, spec in sources.items():
            sampler_node, length, ds = _node_for(spec, split="train", shuffle=True)
            sampler_nodes[str(key)] = sampler_node
            lengths[str(key)] = length
            datasets[str(key)] = ds

        if not sampler_nodes:
            raise RuntimeError("No training sources provided")
        if ParallelMapper is None:
            raise RuntimeError("torchdata.nodes.ParallelMapper is required")

        sample_nodes: Dict[str, BaseNode] = {}
        for key, sampler_node in sampler_nodes.items():
            ds = datasets[key]
            getitem = getattr(ds, "__getitem__", None)
            if not callable(getitem):
                raise TypeError(f"Dataset for key {key!r} has no __getitem__")
            sample_nodes[key] = ParallelMapper(
                sampler_node,
                getitem,
                num_workers=1,
                in_order=True,
                method="thread",
                multiprocessing_context=None,
                max_concurrent=None,
                snapshot_frequency=1,
                prebatch=None,
            )

        _, mapped, _ = compose(
            sample_nodes,
            device=device_obj,
            map_fn=map_fn,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )
        train_length = sum(lengths.values()) if lengths else None

    elif isinstance(sources, (list, tuple)):
        sampler_list: list[BaseNode] = []
        lengths: list[int] = []
        datasets: list[Any] = []
        for spec in sources:
            sampler_node, length, ds = _node_for(spec, split="train", shuffle=True)
            sampler_list.append(sampler_node)
            lengths.append(length)
            datasets.append(ds)

        if not sampler_list:
            raise RuntimeError("No training sources provided")
        if ParallelMapper is None:
            raise RuntimeError("torchdata.nodes.ParallelMapper is required")

        sample_nodes: list[BaseNode] = []
        for sampler_node, ds in zip(sampler_list, datasets):
            getitem = getattr(ds, "__getitem__", None)
            if not callable(getitem):
                raise TypeError("Dataset has no __getitem__")
            sample_nodes.append(
                ParallelMapper(
                    sampler_node,
                    getitem,
                    num_workers=1,
                    in_order=True,
                    method="thread",
                    multiprocessing_context=None,
                    max_concurrent=None,
                    snapshot_frequency=1,
                    prebatch=None,
                )
            )

        _, mapped, _ = compose(
            sample_nodes,
            device=device_obj,
            map_fn=map_fn,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )
        train_length = sum(lengths) if lengths else None

    else:
        sampler_node, train_length, ds = _node_for(sources, split="train", shuffle=True)
        if ParallelMapper is None:
            raise RuntimeError("torchdata.nodes.ParallelMapper is required")
        getitem = getattr(ds, "__getitem__", None)
        if not callable(getitem):
            raise TypeError("Dataset has no __getitem__")
        sample_node = ParallelMapper(
            sampler_node,
            getitem,
            num_workers=1,
            in_order=True,
            method="thread",
            multiprocessing_context=None,
            max_concurrent=None,
            snapshot_frequency=1,
            prebatch=None,
        )
        _, mapped, _ = compose(
            sample_node,
            device=device_obj,
            map_fn=map_fn,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )

    train_loader = Loader(
        device=device_obj,
        node=mapped,
        prefetch_factor=int(pf_depth),
        non_blocking=bool(non_blocking_copy),
        length=train_length,
    )

    # --- val loader 구성 ---
    val_loader = None
    if float(val_frac) > 0.0:
        if isinstance(sources, Mapping) and not _is_source_spec(sources):
            sampler_nodes: Dict[str, BaseNode] = {}
            lengths: Dict[str, int] = {}
            datasets: Dict[str, Any] = {}
            for key, spec in sources.items():
                sampler_node, length, ds = _node_for(spec, split="val", shuffle=False)
                sampler_nodes[str(key)] = sampler_node
                lengths[str(key)] = length
                datasets[str(key)] = ds
            if not sampler_nodes:
                raise RuntimeError("No validation sources provided")
            if ParallelMapper is None:
                raise RuntimeError("torchdata.nodes.ParallelMapper is required")

            sample_nodes: Dict[str, BaseNode] = {}
            for key, sampler_node in sampler_nodes.items():
                ds = datasets[key]
                getitem = getattr(ds, "__getitem__", None)
                if not callable(getitem):
                    raise TypeError(f"Dataset for key {key!r} has no __getitem__")
                sample_nodes[key] = ParallelMapper(
                    sampler_node,
                    getitem,
                    num_workers=1,
                    in_order=True,
                    method="thread",
                    multiprocessing_context=None,
                    max_concurrent=None,
                    snapshot_frequency=1,
                    prebatch=None,
                )

            _, vmapped, _ = compose(
                sample_nodes,
                device=device_obj,
                map_fn=map_fn,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader(
                device=device_obj,
                node=vmapped,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=sum(lengths.values()) if lengths else None,
            )

        elif isinstance(sources, (list, tuple)):
            sampler_list: list[BaseNode] = []
            lengths: list[int] = []
            datasets: list[Any] = []
            for spec in sources:
                sampler_node, length, ds = _node_for(spec, split="val", shuffle=False)
                sampler_list.append(sampler_node)
                lengths.append(length)
                datasets.append(ds)
            if not sampler_list:
                raise RuntimeError("No validation sources provided")
            if ParallelMapper is None:
                raise RuntimeError("torchdata.nodes.ParallelMapper is required")

            sample_nodes: list[BaseNode] = []
            for sampler_node, ds in zip(sampler_list, datasets):
                getitem = getattr(ds, "__getitem__", None)
                if not callable(getitem):
                    raise TypeError("Dataset has no __getitem__")
                sample_nodes.append(
                    ParallelMapper(
                        sampler_node,
                        getitem,
                        num_workers=1,
                        in_order=True,
                        method="thread",
                        multiprocessing_context=None,
                        max_concurrent=None,
                        snapshot_frequency=1,
                        prebatch=None,
                    )
                )

            _, vmapped, _ = compose(
                sample_nodes,
                device=device_obj,
                map_fn=map_fn,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader(
                device=device_obj,
                node=vmapped,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=sum(lengths) if lengths else None,
            )

        else:
            sampler_node, val_len, ds = _node_for(sources, split="val", shuffle=False)
            if ParallelMapper is None:
                raise RuntimeError("torchdata.nodes.ParallelMapper is required")
            getitem = getattr(ds, "__getitem__", None)
            if not callable(getitem):
                raise TypeError("Dataset has no __getitem__")
            sample_node = ParallelMapper(
                sampler_node,
                getitem,
                num_workers=1,
                in_order=True,
                method="thread",
                multiprocessing_context=None,
                max_concurrent=None,
                snapshot_frequency=1,
                prebatch=None,
            )
            _, vmapped, _ = compose(
                sample_node,
                device=device_obj,
                map_fn=map_fn,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader(
                device=device_obj,
                node=vmapped,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=val_len,
            )

    return (train_loader, val_loader, allocated)
