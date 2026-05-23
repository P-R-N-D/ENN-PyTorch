from __future__ import annotations

import math
from typing import Any
from collections.abc import Sequence

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

    Weights are accepted only for sum/mean. They are rejected for min/max.
    Weighted mean normalizes weights before accumulation to avoid avoidable
    numerator overflow.

    ``master_dtype`` controls real-valued accumulation precision for numerically
    sensitive reductions. It is also used to derive the complex accumulation
    dtype. Pure integer sums are accumulated as int64.
    """

    SUPPORTED_OPS = {"sum", "mean", "min", "max"}
    ORDERED_OPS = {"min", "max"}
    MASTER_DTYPES = {torch.float32, torch.float64}

    def __init__(
        self,
        strict_shape: bool = True,
        strict_dtype: bool = True,
        strict_device: bool = True,
        eps: float = 1e-12,
        master_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.strict_shape = strict_shape
        self.strict_dtype = strict_dtype
        self.strict_device = strict_device
        self.eps = float(eps)
        self.master_dtype = master_dtype
        self._verify_config()

    def forward(
        self,
        tensors: Sequence[Tensor],
        op: str = "mean",
        weights: Sequence[float] | Tensor | None = None,
        chunk_size: int | None = None,
    ) -> Tensor:
        op = self._norm_op_str(op)
        xs = self._norm_tensor(tensors)
        chunk_size = self._norm_chunk_size(chunk_size)
        if chunk_size is not None and chunk_size >= len(xs):
            chunk_size = None

        if op in self.ORDERED_OPS and weights is not None:
            raise ValueError(f"{op!r} does not support weights.")

        if op in self.ORDERED_OPS:
            self._verify_no_complex(xs, op)

        dtype = self._infer_reduction_dtype(xs, op=op, weights=weights)
        weight_dtype = self._weight_dtype(dtype=dtype, weights=weights)
        ws = self._norm_weight(
            weights,
            source_count=len(xs),
            ref=xs[0],
            dtype=weight_dtype,
        )

        match op:
            case "sum":
                if chunk_size is None:
                    return self._sum(xs, weights=ws, dtype=dtype)
                return self._sum_chunked(
                    xs, weights=ws, dtype=dtype, chunk_size=chunk_size
                )

            case "mean":
                if chunk_size is None:
                    return self._mean(xs, weights=ws, dtype=dtype)
                return self._mean_chunked(
                    xs, weights=ws, dtype=dtype, chunk_size=chunk_size
                )

            case "min":
                if chunk_size is None:
                    return self._min(xs, dtype=dtype)
                return self._min_chunked(xs, dtype=dtype, chunk_size=chunk_size)

            case "max":
                if chunk_size is None:
                    return self._max(xs, dtype=dtype)
                return self._max_chunked(xs, dtype=dtype, chunk_size=chunk_size)

            case _:
                raise AssertionError(f"Unreachable reducer op: {op}")

    def _sum(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        return self._sum_range(
            xs,
            weights=weights,
            dtype=dtype,
            start=0,
            end=len(xs),
        )

    def _sum_range(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
        start: int,
        end: int,
    ) -> Tensor:
        out = self._scale(
            xs[start],
            weight=self._get_coeff(weights, start),
            dtype=dtype,
        ).clone()

        for i in range(start + 1, end):
            out.add_(
                self._scale(
                    xs[i],
                    weight=self._get_coeff(weights, i),
                    dtype=dtype,
                )
            )

        return out

    def _sum_chunked(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
        chunk_size: int,
    ) -> Tensor:
        out: Tensor | None = None
        n = len(xs)

        for base in range(0, n, chunk_size):
            end = min(base + chunk_size, n)

            if self._can_stack(xs, base, end):
                chunk = torch.stack(
                    [x.to(dtype=dtype) for x in xs[base:end]],
                    dim=0,
                )

                if weights is not None:
                    view = (end - base,) + (1,) * xs[base].ndim
                    coeff = (
                        weights[base:end]
                        .to(device=chunk.device)
                        .reshape(view)
                    )
                    chunk.mul_(coeff)

                reduced = chunk.sum(dim=0)
            else:
                reduced = self._sum_range(
                    xs,
                    weights=weights,
                    dtype=dtype,
                    start=base,
                    end=end,
                )

            if out is None:
                out = reduced
            else:
                out.add_(reduced)

        if out is None:
            raise AssertionError("Reducer requires at least one tensor.")
        return out

    def _mean(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        if weights is None:
            coeff = 1.0 / len(xs)
            out = xs[0].to(dtype=dtype).clone()
            out.mul_(coeff)
            for x in xs[1:]:
                out.add_(x.to(dtype=dtype), alpha=coeff)
            return out

        coeffs = self._mean_coeffs(weights)
        return self._sum(xs, weights=coeffs, dtype=dtype)

    def _mean_chunked(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
        chunk_size: int,
    ) -> Tensor:
        if weights is None:
            coeffs = torch.full(
                (len(xs),),
                1.0 / len(xs),
                device=xs[0].device,
                dtype=self._weight_dtype(dtype=dtype, weights=None),
            )
        else:
            coeffs = self._mean_coeffs(weights)

        return self._sum_chunked(
            xs,
            weights=coeffs,
            dtype=dtype,
            chunk_size=chunk_size,
        )

    def _mean_coeffs(self, weights: Tensor) -> Tensor:
        scale_floor = torch.finfo(weights.dtype).tiny
        scale = weights.abs().amax().clamp_min(scale_floor)
        scaled = weights / scale
        denom = self._safe_signed_denom(scaled)
        return scaled / denom

    def _safe_signed_denom(self, values: Tensor) -> Tensor:
        denom = values.sum()
        safe_eps = max(self.eps, torch.finfo(values.dtype).tiny)
        eps = torch.full_like(denom, safe_eps)
        denom_eps = torch.where(denom < 0, -eps, eps)
        return torch.where(torch.abs(denom) < safe_eps, denom_eps, denom)

    def _min(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
    ) -> Tensor:
        return self._min_range(xs, dtype=dtype, start=0, end=len(xs))

    def _min_range(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
        start: int,
        end: int,
    ) -> Tensor:
        out = xs[start].to(dtype=dtype).clone()

        for i in range(start + 1, end):
            out = torch.minimum(out, xs[i].to(dtype=dtype))

        return out

    def _min_chunked(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
        chunk_size: int,
    ) -> Tensor:
        out: Tensor | None = None
        n = len(xs)

        for base in range(0, n, chunk_size):
            end = min(base + chunk_size, n)

            if self._can_stack(xs, base, end):
                chunk = torch.stack(
                    [x.to(dtype=dtype) for x in xs[base:end]],
                    dim=0,
                )
                reduced = chunk.amin(dim=0)
            else:
                reduced = self._min_range(xs, dtype=dtype, start=base, end=end)

            out = reduced if out is None else torch.minimum(out, reduced)

        if out is None:
            raise AssertionError("Reducer requires at least one tensor.")
        return out

    def _max(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
    ) -> Tensor:
        return self._max_range(xs, dtype=dtype, start=0, end=len(xs))

    def _max_range(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
        start: int,
        end: int,
    ) -> Tensor:
        out = xs[start].to(dtype=dtype).clone()

        for i in range(start + 1, end):
            out = torch.maximum(out, xs[i].to(dtype=dtype))

        return out

    def _max_chunked(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
        chunk_size: int,
    ) -> Tensor:
        out: Tensor | None = None
        n = len(xs)

        for base in range(0, n, chunk_size):
            end = min(base + chunk_size, n)

            if self._can_stack(xs, base, end):
                chunk = torch.stack(
                    [x.to(dtype=dtype) for x in xs[base:end]],
                    dim=0,
                )
                reduced = chunk.amax(dim=0)
            else:
                reduced = self._max_range(xs, dtype=dtype, start=base, end=end)

            out = reduced if out is None else torch.maximum(out, reduced)

        if out is None:
            raise AssertionError("Reducer requires at least one tensor.")
        return out

    def _scale(
        self,
        tensor: Tensor,
        weight: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        value = tensor.to(dtype=dtype)

        if weight is None:
            return value

        return value * weight.to(device=value.device)

    @staticmethod
    def _get_coeff(weights: Tensor | None, index: int) -> Tensor | None:
        if weights is None:
            return None
        return weights[index]

    @staticmethod
    def _can_stack(xs: Sequence[Tensor], start: int, end: int) -> bool:
        shape = xs[start].shape
        return all(x.shape == shape for x in xs[start + 1 : end])

    def _norm_op_str(self, op: str) -> str:
        if not isinstance(op, str):
            raise TypeError(f"op must be str, got {type(op)!r}")

        normalized = op.lower().strip()
        if normalized not in self.SUPPORTED_OPS:
            supported = ", ".join(sorted(self.SUPPORTED_OPS))
            raise ValueError(
                f"Unsupported reducer op: {normalized!r}. Supported ops: {supported}"
            )

        return normalized

    @staticmethod
    def _norm_chunk_size(chunk_size: int | None) -> int | None:
        if chunk_size is None:
            return None
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be int | None, got {type(chunk_size)!r}"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        return chunk_size

    def _norm_tensor(self, tensors: Sequence[Tensor]) -> list[Tensor]:
        xs = list(tensors)

        if not xs:
            raise ValueError("Reducer requires at least one tensor.")

        first = xs[0]
        if not isinstance(first, Tensor):
            raise TypeError(f"Expected Tensor at index 0, got {type(first)!r}")

        for i, x in enumerate(xs):
            if not isinstance(x, Tensor):
                raise TypeError(f"Expected Tensor at index {i}, got {type(x)!r}")

            if x.dtype == torch.bool:
                raise TypeError("Reducer does not support bool tensors.")

            if x.is_quantized:
                raise TypeError("Reducer does not support quantized tensors.")

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

            if self.strict_device and x.device != first.device:
                raise ValueError(
                    "All tensors must be on the same device when strict_device=True. "
                    f"tensors[0].device={first.device}, tensors[{i}].device={x.device}"
                )

        return xs

    def _infer_reduction_dtype(
        self,
        xs: list[Tensor],
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        dtype = self._promote_input_dtype(xs)
        has_weights = weights is not None

        if op in self.ORDERED_OPS:
            if self._is_complex(dtype):
                raise TypeError(f"{op!r} does not support complex tensors.")
            return dtype

        if self._is_complex(dtype):
            return self._complex_dtype_for(dtype, weights=weights)

        if self._is_float(dtype):
            return self._real_dtype_for(dtype, weights=weights)

        if op == "sum" and not has_weights:
            return torch.int64

        return self._real_dtype_for(dtype, weights=weights)

    def _promote_input_dtype(self, xs: list[Tensor]) -> torch.dtype:
        dtype = xs[0].dtype
        if self.strict_dtype:
            return dtype

        for x in xs[1:]:
            dtype = torch.promote_types(dtype, x.dtype)

        return dtype

    def _real_dtype_for(
        self,
        dtype: torch.dtype,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if dtype == torch.float64 or self._weight_is_float64(weights):
            return torch.float64
        if self.master_dtype == torch.float64:
            return torch.float64
        return torch.float32

    def _complex_dtype_for(
        self,
        dtype: torch.dtype,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if dtype == torch.complex128:
            return torch.complex128
        if self.master_dtype == torch.float64 or self._weight_is_float64(weights):
            return torch.complex128
        return torch.complex64

    def _weight_dtype(
        self,
        dtype: torch.dtype,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if dtype in {torch.float64, torch.complex128}:
            return torch.float64
        if self._weight_is_float64(weights):
            return torch.float64
        return self.master_dtype

    @staticmethod
    def _weight_is_float64(weights: Sequence[float] | Tensor | None) -> bool:
        return isinstance(weights, Tensor) and weights.dtype == torch.float64

    def _norm_weight(
        self,
        weights: Sequence[float] | Tensor | None,
        source_count: int,
        ref: Tensor,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if weights is None:
            return None

        if isinstance(weights, Tensor):
            if weights.requires_grad:
                raise ValueError("Reducer weights must not require gradients.")
            if torch.is_complex(weights):
                raise ValueError("Reducer does not support complex weights.")
            w = weights.to(device=ref.device, dtype=dtype)
        else:
            values: list[float] = []
            for weight in weights:
                probe = torch.as_tensor(weight)
                if probe.ndim != 0:
                    raise ValueError("Reducer weights must be scalar values.")
                if torch.is_complex(probe):
                    raise ValueError("Reducer does not support complex weights.")
                value = float(probe)
                if not math.isfinite(value):
                    raise ValueError("Reducer weights must be finite.")
                values.append(value)
            w = torch.tensor(values, device=ref.device, dtype=dtype)

        if w.ndim != 1:
            raise ValueError(f"weights must be 1-D. Got shape={tuple(w.shape)}.")

        if w.numel() != source_count:
            raise ValueError(
                f"weights length must match tensor count. "
                f"len(weights)={w.numel()}, tensor_count={source_count}"
            )

        return w

    def _verify_config(self) -> None:
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be finite and positive, got {self.eps}")

        if self.master_dtype not in self.MASTER_DTYPES:
            raise TypeError(
                "master_dtype must be torch.float32 or torch.float64, "
                f"got {self.master_dtype!r}"
            )

    @staticmethod
    def _is_float(dtype: torch.dtype) -> bool:
        return torch.empty((), dtype=dtype).is_floating_point()

    @staticmethod
    def _is_complex(dtype: torch.dtype) -> bool:
        return torch.empty((), dtype=dtype).is_complex()

    def _verify_no_complex(self, xs: Sequence[Tensor], op: str) -> None:
        if any(x.is_complex() for x in xs):
            raise TypeError(f"{op!r} does not support complex tensors.")

    def extra_repr(self) -> str:
        return (
            f"strict_shape={self.strict_shape}, "
            f"strict_dtype={self.strict_dtype}, "
            f"strict_device={self.strict_device}, "
            f"eps={self.eps}, "
            f"master_dtype={self.master_dtype}"
        )


class ConvND(nn.Module):
    """
    Channel-last Conv1d/Conv2d/Conv3d adapter for region-local tensors.

    Expected input shape:
        (B, R, *local_shape, D)

    The layer inspects ``local_shape``:
      - rank 1 -> depthwise-separable Conv1d
      - rank 2 -> depthwise-separable Conv2d
      - rank 3 -> depthwise-separable Conv3d
      - rank 0 or rank > 3 -> identity fallback

    This is intentionally an adapter over PyTorch's optimized 1D/2D/3D
    convolution layers, not a custom arbitrary-rank convolution kernel.
    """

    SUPPORTED_LOCAL_DIMS = {1, 2, 3}

    def __init__(
        self,
        dim: int,
        *args: Any,
        kernel_size: int = 3,
        enabled: bool = True,
        bias: bool = True,
        activation: str = "gelu",
        residual: bool = True,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        _ = args
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.enabled = bool(enabled)
        self.residual = bool(residual)

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if self.kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}"
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "ConvND requires an odd kernel_size so the local shape "
                f"can be preserved. Got kernel_size={kernel_size}."
            )

        self.conv1 = self._get_conv(1, bias=bias, activation=activation)
        self.conv2 = self._get_conv(2, bias=bias, activation=activation)
        self.conv3 = self._get_conv(3, bias=bias, activation=activation)

        if self.residual:
            self.residual_scale = nn.Parameter(
                torch.tensor(float(residual_scale_init))
            )
        else:
            self.register_parameter("residual_scale", None)

    def forward(self, x: Tensor, *args: Any) -> Tensor:
        _ = args
        if not isinstance(x, Tensor):
            raise TypeError(f"ConvND expects Tensor, got {type(x)!r}")

        if not self.enabled:
            return x

        if x.ndim < 3:
            return x

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim={self.dim}, got shape={tuple(x.shape)}"
            )

        if not x.is_floating_point():
            return x

        local_ndim = x.ndim - 3
        match local_ndim:
            case 1:
                return self._forward_1d(x)
            case 2:
                return self._forward_2d(x)
            case 3:
                return self._forward_3d(x)
            case _:
                return x

    def _forward_1d(self, x: Tensor) -> Tensor:
        B, R, L, D = x.shape
        h = x.reshape(B * R, L, D).transpose(1, 2).contiguous()
        y = self.conv1(h)
        y = y.transpose(1, 2).reshape(B, R, L, D)
        return self._postprocess(x, y)

    def _forward_2d(self, x: Tensor) -> Tensor:
        B, R, H, W, D = x.shape
        h = x.reshape(B * R, H, W, D).permute(0, 3, 1, 2).contiguous()
        y = self.conv2(h)
        y = y.permute(0, 2, 3, 1).reshape(B, R, H, W, D)
        return self._postprocess(x, y)

    def _forward_3d(self, x: Tensor) -> Tensor:
        B, R, T, H, W, D = x.shape
        h = (
            x.reshape(B * R, T, H, W, D)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        y = self.conv3(h)
        y = y.permute(0, 2, 3, 4, 1).reshape(B, R, T, H, W, D)
        return self._postprocess(x, y)

    def _postprocess(self, x: Tensor, y: Tensor) -> Tensor:
        if not self.residual:
            return y
        scale = self.residual_scale.to(device=y.device, dtype=y.dtype)
        return x + scale * y

    def _get_conv(
        self,
        ndim: int,
        *args: Any,
        bias: bool,
        activation: str,
    ) -> nn.Sequential:
        _ = args
        padding = self.kernel_size // 2
        conv_cls: type[nn.Conv1d] | type[nn.Conv2d] | type[nn.Conv3d]
        match ndim:
            case 1:
                conv_cls = nn.Conv1d
            case 2:
                conv_cls = nn.Conv2d
            case 3:
                conv_cls = nn.Conv3d
            case _:
                raise ValueError(f"Unsupported conv ndim: {ndim}")

        return nn.Sequential(
            conv_cls(
                self.dim,
                self.dim,
                kernel_size=self.kernel_size,
                padding=padding,
                groups=self.dim,
                bias=bias,
            ),
            self._get_activation(activation),
            conv_cls(
                self.dim,
                self.dim,
                kernel_size=1,
                padding=0,
                groups=1,
                bias=bias,
            ),
        )

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        normalized = str(name).lower().strip()
        match normalized:
            case "gelu":
                return nn.GELU()
            case "silu" | "swish":
                return nn.SiLU()
            case "relu":
                return nn.ReLU()
            case "identity" | "none" | "linear":
                return nn.Identity()
            case _:
                raise ValueError(f"Unsupported activation: {name!r}")
