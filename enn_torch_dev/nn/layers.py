from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext

import torch
from torch import Tensor, nn


class Reducer(nn.Module):
    """
    Pure tensor reducer.

    Reducer is a stateless module that reduces multiple tensors into one tensor.
    It does not know about store keys, plans, executors, or KV stores, and it has
    no trainable parameters.

    Supported ops:
      - sum
      - mean
      - min
      - max

    Weights are accepted only for sum/mean. They are rejected for min/max because
    the semantics are ambiguous.

    cast:
      - False: no forced safe cast
      - True : always cast to the selected compute dtype
      - None : auto, cast only when the dtype/op combination needs it

    cast_dtype:
      - If None, a baseline compute dtype is selected from dtype/op/weights.
    """

    SUPPORTED_OPS = {"sum", "mean", "min", "max"}
    WEIGHTED_OPS = {"sum", "mean"}
    ORDERED_OPS = {"min", "max"}

    _AUTOCAST_DEVICE_TYPES = {
        "cuda",
        "cpu",
        "xpu",
        "hpu",
        "mtia",
        "maia",
    }

    def __init__(
        self,
        *,
        strict_shape: bool = True,
        strict_dtype: bool = True,
        validate_device: bool = True,
        eps: float = 1e-12,
        cast: bool | None = None,
        cast_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.strict_shape = strict_shape
        self.strict_dtype = strict_dtype
        self.validate_device = validate_device
        self.eps = float(eps)
        self.cast = cast
        self.cast_dtype = cast_dtype

    def forward(
        self,
        tensors: Sequence[Tensor],
        *,
        op: str = "mean",
        weights: Sequence[float] | Tensor | None = None,
    ) -> Tensor:
        op = self._normalize_op(op)
        xs = self._validate_tensors(tensors)

        if op in self.ORDERED_OPS and weights is not None:
            raise ValueError(f"{op!r} does not support weights.")

        dtype = self._resolve_compute_dtype(xs, op=op, weights=weights)
        ws = self._make_weights(
            weights,
            source_count=len(xs),
            ref=xs[0],
            dtype=dtype,
        )

        if op in self.ORDERED_OPS:
            self._require_ordered_compatible(xs, op)

        with self._disabled_autocast_if_needed(xs[0], dtype):
            match op:
                case "sum":
                    return self._reduce_sum(xs, weights=ws, dtype=dtype)

                case "mean":
                    return self._reduce_mean(xs, weights=ws, dtype=dtype)

                case "min":
                    return self._reduce_min(xs, dtype=dtype)

                case "max":
                    return self._reduce_max(xs, dtype=dtype)

                case _:
                    raise AssertionError(f"Unreachable reducer op: {op}")

    def _reduce_sum(
        self,
        xs: list[Tensor],
        *,
        weights: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        out = self._prepare_value(
            xs[0],
            weight=self._weight_at(weights, 0),
            dtype=dtype,
        ).clone()

        for i, x in enumerate(xs[1:], start=1):
            out.add_(
                self._prepare_value(
                    x,
                    weight=self._weight_at(weights, i),
                    dtype=dtype,
                )
            )

        return out

    def _reduce_mean(
        self,
        xs: list[Tensor],
        *,
        weights: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        out = self._reduce_sum(xs, weights=weights, dtype=dtype)

        if weights is None:
            return out / len(xs)

        denom = weights.sum()
        if torch.abs(denom).item() < self.eps:
            raise ValueError(
                "Cannot compute weighted mean because weight sum is too close to zero."
            )

        return out / denom

    def _reduce_min(
        self,
        xs: list[Tensor],
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        out = xs[0].to(dtype=dtype).clone()

        for x in xs[1:]:
            out = torch.minimum(out, x.to(dtype=dtype))

        return out

    def _reduce_max(
        self,
        xs: list[Tensor],
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        out = xs[0].to(dtype=dtype).clone()

        for x in xs[1:]:
            out = torch.maximum(out, x.to(dtype=dtype))

        return out

    def _prepare_value(
        self,
        tensor: Tensor,
        *,
        weight: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        value = tensor.to(dtype=dtype)

        if weight is None:
            return value

        return value * weight.to(device=value.device, dtype=value.dtype)

    @staticmethod
    def _weight_at(weights: Tensor | None, index: int) -> Tensor | None:
        if weights is None:
            return None
        return weights[index]

    def _normalize_op(self, op: str) -> str:
        if not isinstance(op, str):
            raise TypeError(f"op must be str, got {type(op)!r}")

        normalized = op.lower().strip()
        if normalized not in self.SUPPORTED_OPS:
            supported = ", ".join(sorted(self.SUPPORTED_OPS))
            raise ValueError(
                f"Unsupported reducer op: {normalized!r}. Supported ops: {supported}"
            )

        return normalized

    def _validate_tensors(self, tensors: Sequence[Tensor]) -> list[Tensor]:
        xs = list(tensors)

        if not xs:
            raise ValueError("Reducer requires at least one tensor.")

        first = xs[0]
        if not isinstance(first, Tensor):
            raise TypeError(f"Expected Tensor at index 0, got {type(first)!r}")

        for i, x in enumerate(xs):
            if not isinstance(x, Tensor):
                raise TypeError(f"Expected Tensor at index {i}, got {type(x)!r}")

            if self.strict_shape and x.shape != first.shape:
                raise ValueError(
                    "All tensors must have the same shape. "
                    f"tensors[0].shape={tuple(first.shape)}, "
                    f"tensors[{i}].shape={tuple(x.shape)}"
                )

            if self.strict_dtype and x.dtype != first.dtype:
                raise ValueError(
                    "All tensors must have the same dtype. "
                    f"tensors[0].dtype={first.dtype}, tensors[{i}].dtype={x.dtype}"
                )

            if self.validate_device and x.device != first.device:
                raise ValueError(
                    "All tensors must be on the same device when validate_device=True. "
                    f"tensors[0].device={first.device}, tensors[{i}].device={x.device}"
                )

        return xs

    def _resolve_compute_dtype(
        self,
        xs: list[Tensor],
        *,
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if self.cast not in {True, False, None}:
            raise TypeError(f"cast must be bool | None, got {type(self.cast)!r}")

        if self.cast_dtype is not None and not isinstance(self.cast_dtype, torch.dtype):
            raise TypeError(
                f"cast_dtype must be torch.dtype | None, got {type(self.cast_dtype)!r}"
            )

        dtype = xs[0].dtype

        if not self.strict_dtype:
            for x in xs[1:]:
                dtype = torch.promote_types(dtype, x.dtype)

        if isinstance(weights, Tensor):
            if torch.is_complex(weights):
                raise ValueError("Reducer does not support complex weights.")
            dtype = torch.promote_types(dtype, weights.dtype)

        if self.cast is False:
            return self._required_dtype_without_auto_cast(dtype, op=op, weights=weights)

        if self.cast is True:
            return self.cast_dtype or self._baseline_dtype(dtype, op=op, weights=weights)

        if self._should_auto_cast(dtype, op=op, weights=weights):
            return self.cast_dtype or self._baseline_dtype(dtype, op=op, weights=weights)

        return self._required_dtype_without_auto_cast(dtype, op=op, weights=weights)

    def _required_dtype_without_auto_cast(
        self,
        dtype: torch.dtype,
        *,
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if dtype == torch.bool:
            raise TypeError("Reducer does not support bool tensors.")

        if torch.empty((), dtype=dtype).is_complex():
            if op in self.ORDERED_OPS:
                raise TypeError(f"{op!r} does not support complex tensors.")
            return dtype

        if op == "mean" and not self._is_float_dtype(dtype):
            return torch.get_default_dtype()

        if weights is not None and not self._is_float_dtype(dtype):
            return torch.get_default_dtype()

        return dtype

    def _should_auto_cast(
        self,
        dtype: torch.dtype,
        *,
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> bool:
        if dtype == torch.bool:
            raise TypeError("Reducer does not support bool tensors.")

        if dtype in {torch.float16, torch.bfloat16}:
            return op in {"sum", "mean"} or weights is not None

        if dtype == self._complex32_dtype():
            return True

        if not self._is_float_or_complex_dtype(dtype):
            return op == "sum" or op == "mean" or weights is not None

        return False

    def _baseline_dtype(
        self,
        dtype: torch.dtype,
        *,
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if dtype == torch.bool:
            raise TypeError("Reducer does not support bool tensors.")

        if dtype in {torch.float16, torch.bfloat16}:
            if op in {"sum", "mean"} or weights is not None:
                return torch.float32
            return dtype

        if dtype in {torch.float32, torch.float64}:
            return dtype

        if dtype == self._complex32_dtype():
            if op in self.ORDERED_OPS:
                raise TypeError(f"{op!r} does not support complex tensors.")
            return torch.complex64

        if dtype in {torch.complex64, torch.complex128}:
            if op in self.ORDERED_OPS:
                raise TypeError(f"{op!r} does not support complex tensors.")
            return dtype

        if op == "mean" or weights is not None:
            return torch.float32

        if op == "sum":
            return torch.int64

        return dtype

    def _make_weights(
        self,
        weights: Sequence[float] | Tensor | None,
        *,
        source_count: int,
        ref: Tensor,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if weights is None:
            return None

        if ref.is_complex():
            if isinstance(weights, Tensor) and torch.is_complex(weights):
                raise ValueError("Reducer does not support complex weights.")

        if isinstance(weights, Tensor):
            if weights.requires_grad:
                raise ValueError("Reducer weights must not require gradients.")
            w = weights.to(device=ref.device, dtype=dtype)
        else:
            if any(isinstance(weight, complex) for weight in weights):
                raise ValueError("Reducer does not support complex weights.")
            w = torch.tensor(weights, device=ref.device, dtype=dtype)

        if w.ndim != 1:
            raise ValueError(f"weights must be 1-D. Got shape={tuple(w.shape)}.")

        if w.numel() != source_count:
            raise ValueError(
                f"weights length must match tensor count. "
                f"len(weights)={w.numel()}, tensor_count={source_count}"
            )

        return w

    def _disabled_autocast_if_needed(
        self, ref: Tensor, dtype: torch.dtype
    ) -> AbstractContextManager[None]:
        if dtype == ref.dtype:
            return nullcontext()

        device_type = ref.device.type
        if device_type not in self._AUTOCAST_DEVICE_TYPES:
            return nullcontext()

        return torch.autocast(device_type=device_type, enabled=False)

    @staticmethod
    def _complex32_dtype() -> torch.dtype | None:
        value = getattr(torch, "complex32", None)
        return value if isinstance(value, torch.dtype) else None

    @staticmethod
    def _is_float_dtype(dtype: torch.dtype) -> bool:
        return torch.empty((), dtype=dtype).is_floating_point()

    @staticmethod
    def _is_float_or_complex_dtype(dtype: torch.dtype) -> bool:
        probe = torch.empty((), dtype=dtype)
        return probe.is_floating_point() or probe.is_complex()

    def _require_ordered_compatible(self, xs: Sequence[Tensor], op: str) -> None:
        if any(x.is_complex() for x in xs):
            raise TypeError(f"{op!r} does not support complex tensors.")

    def extra_repr(self) -> str:
        return (
            f"strict_shape={self.strict_shape}, "
            f"strict_dtype={self.strict_dtype}, "
            f"validate_device={self.validate_device}, "
            f"eps={self.eps}, "
            f"cast={self.cast}, "
            f"cast_dtype={self.cast_dtype}"
        )
