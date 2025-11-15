# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from tensordict import TensorDict, TensorDictBase, stack

from ..backend.system import get_tlb, optimize_threads

try:
    from torchdata.nodes import BaseNode
except Exception:
    from torchdata.nodes import BaseNode

from .nodes import (
    Dataset,
    Disposable,
    Connector,
    Loader,
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


def collate(batch: Any, *args: Any, labels_dtype: Optional[torch.dtype] = None, **kwargs: Any) -> Any:
    """
    Keep it simple: stack per-key when possible. Pre/post-processing is done in map_fn.
    """
    if isinstance(batch, (list, tuple)):
        if not batch:
            return batch
        if all(isinstance(elem, TensorDictBase) for elem in batch):
            return stack(list(batch), dim=0)
        if all(isinstance(elem, Mapping) for elem in batch):
            Xs = [elem.get("X") for elem in batch]
            Ys = [elem.get("Y") for elem in batch]
            try:
                if all(isinstance(x, torch.Tensor) for x in Xs):
                    Xs = torch.stack(Xs, dim=0)
            except Exception:
                pass
            try:
                if all(isinstance(y, torch.Tensor) for y in Ys):
                    Ys = torch.stack(Ys, dim=0)
            except Exception:
                pass
            if labels_dtype is not None and isinstance(Ys, torch.Tensor):
                Ys = Ys.to(dtype=labels_dtype, non_blocking=True)
            return TensorDict({"X": Xs, "Y": Ys, "features": Xs, "labels": Ys}, batch_size=[])
        return batch
    if isinstance(batch, Mapping):
        X = batch.get("X")
        Y = batch.get("Y")
        if labels_dtype is not None and isinstance(Y, torch.Tensor):
            Y = Y.to(dtype=labels_dtype, non_blocking=True)
        return TensorDict({"X": X, "Y": Y, "features": X, "labels": Y}, batch_size=[])
    return batch

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
    device_obj = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )
    try:
        get_tlb(io_workers=io_workers)
    except Exception:
        pass

    mx_weights = None
    if isinstance(node_or_nodes, Mapping) and isinstance(weights, Mapping):
        mx_weights = weights

    # 유한 스트림: 한 번 소진되면 종료
    sampler = Sampler(stop_criteria="ALL_DATASETS_EXHAUSTED", seed=0, weights=mx_weights)
    source = sampler.compose(node_or_nodes)

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
    device_obj = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )

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

    def _node_for(
        spec: SourceSpec, key: str, split: str, shuffle: bool
    ) -> Tuple[BaseNode, int, Any]:
        ds = dataset(spec, split=split, val_frac=float(val_frac))
        allocated.add(ds)

        sampler_node = ds.compose(
            batch_size=int(batch_size),
            shuffle=shuffle,
            seed=0,
            key=str(key),
        )
        return sampler_node, len(ds), ds

    if isinstance(sources, Mapping) and not _is_source_spec(sources):
        sampler_nodes: Dict[str, BaseNode] = {}
        lengths: Dict[str, int] = {}
        datasets: Dict[str, Any] = {}
        for key, spec in sources.items():
            sampler_node, length, ds = _node_for(spec, key=str(key), split="train", shuffle=True)
            if length > 0:
                sampler_nodes[str(key)] = sampler_node
                lengths[str(key)] = length
                datasets[str(key)] = ds

        # 길이 0 소스 제거 후 가중치 정리
        if isinstance(train_weights, Mapping):
            train_weights = {k: v for k, v in dict(train_weights).items() if k in sampler_nodes}

        if not sampler_nodes:
            raise RuntimeError("No non-empty training sources provided")

        def iterate(sample):
            def _one(smpl):
                if (
                    isinstance(smpl, (list, tuple))
                    and len(smpl) == 2
                    and isinstance(smpl[0], str)
                ):
                    k, rng = smpl
                    s, e = int(rng[0]), int(rng[1])
                    ds = datasets.get(k)
                    if ds is None:
                        raise KeyError(f"Unknown dataset key: {k}")
                    batch = ds.get(s, e)
                    return map_fn(batch)
                return map_fn(smpl)

            if isinstance(sample, list):
                return [_one(smpl) for smpl in sample]
            return _one(sample)

        _, mapped, _ = compose(
            sampler_nodes,
            device=device_obj,
            map_fn=iterate,
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
        datasets: Dict[str, Any] = {}
        for i, spec in enumerate(sources):
            key = str(i)
            sampler_node, length, ds = _node_for(spec, key=key, split="train", shuffle=True)
            if length > 0:
                sampler_list.append(sampler_node)
                lengths.append(length)
                datasets[key] = ds

        if not sampler_list:
            raise RuntimeError("No non-empty training sources provided")

        def iterate(sample):
            def _one(smpl):
                if (
                    isinstance(smpl, (list, tuple))
                    and len(smpl) == 2
                    and isinstance(smpl[0], str)
                ):
                    k, rng = smpl
                    s, e = int(rng[0]), int(rng[1])
                    ds = datasets.get(k)
                    if ds is None:
                        raise KeyError(f"Unknown dataset key: {k}")
                    batch = ds.get(s, e)
                    return map_fn(batch)
                return map_fn(smpl)

            if isinstance(sample, list):
                return [_one(smpl) for smpl in sample]
            return _one(sample)

        _, mapped, _ = compose(
            sampler_list,
            device=device_obj,
            map_fn=iterate,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )
        train_length = sum(lengths) if lengths else None

    else:
        sampler_node, train_length, ds = _node_for(sources, key="0", split="train", shuffle=True)
        datasets: Dict[str, Any] = {"0": ds}

        def iterate(sample):
            def _one(smpl):
                if (
                    isinstance(smpl, (list, tuple))
                    and len(smpl) == 2
                    and isinstance(smpl[0], str)
                ):
                    k, rng = smpl
                    s, e = int(rng[0]), int(rng[1])
                    ds_ = datasets.get(k)
                    if ds_ is None:
                        raise KeyError(f"Unknown dataset key: {k}")
                    batch = ds_.get(s, e)
                    return map_fn(batch)
                return map_fn(smpl)

            if isinstance(sample, list):
                return [_one(smpl) for smpl in sample]
            return _one(sample)

        _, mapped, _ = compose(
            sampler_node,
            device=device_obj,
            map_fn=iterate,
            prefetch_factor=int(pf_depth),
            non_blocking_copy=bool(non_blocking_copy),
            io_workers=io_workers,
            prebatch=prebatch,
            weights=train_weights,
        )

    train_loader = Loader.compose(
        mapped,
        device=device_obj,
        prefetch_factor=int(pf_depth),
        non_blocking=bool(non_blocking_copy),
        length=train_length,
    )

    val_loader = None
    if float(val_frac) > 0.0:
        if isinstance(sources, Mapping) and not _is_source_spec(sources):
            sampler_nodes: Dict[str, BaseNode] = {}
            lengths: Dict[str, int] = {}
            datasets: Dict[str, Any] = {}
            for key, spec in sources.items():
                sampler_node, length, ds = _node_for(spec, key=str(key), split="val", shuffle=False)
                if length > 0:
                    sampler_nodes[str(key)] = sampler_node
                    lengths[str(key)] = length
                    datasets[str(key)] = ds

            if isinstance(val_weights, Mapping):
                val_weights = {k: v for k, v in dict(val_weights).items() if k in sampler_nodes}
            if not sampler_nodes:
                raise RuntimeError("No non-empty validation sources provided")

            def iterate(sample):
                def _one(smpl):
                    if (
                        isinstance(smpl, (list, tuple))
                        and len(smpl) == 2
                        and isinstance(smpl[0], str)
                    ):
                        k, rng = smpl
                        s, e = int(rng[0]), int(rng[1])
                        ds = datasets.get(k)
                        if ds is None:
                            raise KeyError(f"Unknown dataset key: {k}")
                        batch = ds.get(s, e)
                        return map_fn(batch)
                    return map_fn(smpl)

                if isinstance(sample, list):
                    return [_one(smpl) for smpl in sample]
                return _one(sample)

            _, vmapped, _ = compose(
                sampler_nodes,
                device=device_obj,
                map_fn=iterate,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader.compose(
                vmapped,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=sum(lengths.values()) if lengths else None,
            )

        elif isinstance(sources, (list, tuple)):
            sampler_list: list[BaseNode] = []
            lengths: list[int] = []
            datasets: Dict[str, Any] = {}
            for i, spec in enumerate(sources):
                k = str(i)
                sampler_node, length, ds = _node_for(spec, key=k, split="val", shuffle=False)
                if length > 0:
                    sampler_list.append(sampler_node)
                    lengths.append(length)
                    datasets[k] = ds
            if not sampler_list:
                raise RuntimeError("No non-empty validation sources provided")

            def iterate(sample):
                def _one(smpl):
                    if (
                        isinstance(smpl, (list, tuple))
                        and len(smpl) == 2
                        and isinstance(smpl[0], str)
                    ):
                        k, rng = smpl
                        s, e = int(rng[0]), int(rng[1])
                        ds = datasets.get(k)
                        if ds is None:
                            raise KeyError(f"Unknown dataset key: {k}")
                        batch = ds.get(s, e)
                        return map_fn(batch)
                    return map_fn(smpl)

                if isinstance(sample, list):
                    return [_one(smpl) for smpl in sample]
                return _one(sample)

            _, vmapped, _ = compose(
                sampler_list,
                device=device_obj,
                map_fn=iterate,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader.compose(
                vmapped,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=sum(lengths) if lengths else None,
            )

        else:
            sampler_node, val_len, ds = _node_for(sources, key="0", split="val", shuffle=False)
            datasets: Dict[str, Any] = {"0": ds}

            def iterate(sample):
                def _one(smpl):
                    if (
                        isinstance(smpl, (list, tuple))
                        and len(smpl) == 2
                        and isinstance(smpl[0], str)
                    ):
                        k, rng = smpl
                        s, e = int(rng[0]), int(rng[1])
                        ds_ = datasets.get(k)
                        if ds_ is None:
                            raise KeyError(f"Unknown dataset key: {k}")
                        batch = ds_.get(s, e)
                        return map_fn(batch)
                    return map_fn(smpl)

                if isinstance(sample, list):
                    return [_one(smpl) for smpl in sample]
                return _one(sample)

            _, vmapped, _ = compose(
                sampler_node,
                device=device_obj,
                map_fn=iterate,
                prefetch_factor=int(pf_depth),
                non_blocking_copy=bool(non_blocking_copy),
                io_workers=io_workers,
                prebatch=prebatch,
                weights=val_weights,
            )
            val_loader = Loader.compose(
                vmapped,
                device=device_obj,
                prefetch_factor=int(pf_depth),
                non_blocking=bool(non_blocking_copy),
                length=val_len,
            )

    return (train_loader, val_loader, allocated)
