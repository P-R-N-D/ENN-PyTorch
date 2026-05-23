from __future__ import annotations

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
    Weighted mean uses an epsilon-safe denominator to avoid division by zero.

    cast:
      - False: no forced safe cast
      - True : always cast to the selected compute dtype
      - None : auto, cast only when the dtype/op combination needs it

    output_dtype:
      - If None, a baseline compute dtype is selected from dtype/op/weights.
      - If provided, used when cast policy selects forced/default promotion.

    master_*_dtype:
      - Canonical master dtypes used by internal promotion rules.
      - Defaults are int64 / float32 / complex64.
    """

    SUPPORTED_OPS = {"sum", "mean", "min", "max"}
    ORDERED_OPS = {"min", "max"}
    MASTER_INT_DTYPE = torch.int64
    MASTER_FLOAT_DTYPE = torch.float32
    MASTER_COMPLEX_DTYPE = torch.complex64

    def __init__(
        self,
        strict_shape: bool = True,
        strict_dtype: bool = True,
        strict_device: bool = True,
        eps: float = 1e-12,
        cast: bool | None = None,
        output_dtype: torch.dtype | None = None,
        master_int_dtype: torch.dtype | None = None,
        master_float_dtype: torch.dtype | None = None,
        master_complex_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.strict_shape = strict_shape
        self.strict_dtype = strict_dtype
        self.strict_device = strict_device
        self.eps = float(eps)
        self.cast = cast
        self.output_dtype = output_dtype
        self.master_int_dtype = master_int_dtype or self.MASTER_INT_DTYPE
        self.master_float_dtype = master_float_dtype or self.MASTER_FLOAT_DTYPE
        self.master_complex_dtype = master_complex_dtype or self.MASTER_COMPLEX_DTYPE
        self._verify_master_dtypes()

    def forward(
        self,
        tensors: Sequence[Tensor],
        op: str = "mean",
        weights: Sequence[float] | Tensor | None = None,
        chunk_size: int | None = None,
    ) -> Tensor:
        op = self._norm_op_str(op)
        xs = self._norm_tensor(tensors)
        self._verify_no_bool(xs)

        if op in self.ORDERED_OPS and weights is not None:
            raise ValueError(f"{op!r} does not support weights.")

        dtype = self._infer_reduction_dtype(xs, op=op, weights=weights)
        ws = self._norm_weight(
            weights,
            source_count=len(xs),
            ref=xs[0],
            dtype=dtype,
        )

        if op in self.ORDERED_OPS:
            self._verify_no_complex(xs, op)

        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {chunk_size}")
            if chunk_size >= len(xs):
                chunk_size = None

        match op:
            case "sum":
                if chunk_size is None:
                    return self._sum(xs, weights=ws, dtype=dtype)
                return self._sum_chunked(xs, weights=ws, dtype=dtype, chunk_size=chunk_size)

            case "mean":
                if chunk_size is None:
                    return self._mean(xs, weights=ws, dtype=dtype)
                return self._mean_chunked(xs, weights=ws, dtype=dtype, chunk_size=chunk_size)

            case "min":
                return self._min(xs, dtype=dtype)

            case "max":
                return self._max(xs, dtype=dtype)

            case _:
                raise AssertionError(f"Unreachable reducer op: {op}")

    def _sum(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        out = self._scale(
            xs[0],
            weight=self._get_coeff(weights, 0),
            dtype=dtype,
        ).clone()

        for i, x in enumerate(xs[1:], start=1):
            out.add_(
                self._scale(
                    x,
                    weight=self._get_coeff(weights, i),
                    dtype=dtype,
                )
            )

        return out

    def _mean(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        out = self._sum(xs, weights=weights, dtype=dtype)

        if weights is None:
            return out / len(xs)

        return out / self._safe_weight_denom(weights)

    def _sum_chunked(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
        chunk_size: int,
    ) -> Tensor:
        out = self._scale(
            xs[0],
            weight=self._get_coeff(weights, 0),
            dtype=dtype,
        ).clone()
        n = len(xs)
        for base in range(1, n, chunk_size):
            end = min(base + chunk_size, n)
            for i in range(base, end):
                out.add_(
                    self._scale(
                        xs[i],
                        weight=self._get_coeff(weights, i),
                        dtype=dtype,
                    )
                )
        return out

    def _mean_chunked(
        self,
        xs: list[Tensor],
        weights: Tensor | None,
        dtype: torch.dtype,
        chunk_size: int,
    ) -> Tensor:
        out = self._sum_chunked(xs, weights=weights, dtype=dtype, chunk_size=chunk_size)
        if weights is None:
            return out / len(xs)
        return out / self._safe_weight_denom(weights)

    def _safe_weight_denom(self, weights: Tensor) -> Tensor:
        denom = weights.sum()
        real_dtype = denom.real.dtype if torch.is_complex(denom) else denom.dtype
        eps_floor = torch.finfo(real_dtype).tiny
        safe_eps = max(self.eps, eps_floor)
        eps_val = torch.full_like(denom, safe_eps)
        denom_sign_ref = denom.real if torch.is_complex(denom) else denom
        denom_eps = torch.where(denom_sign_ref < 0, -eps_val, eps_val)
        return torch.where(torch.abs(denom) < safe_eps, denom_eps, denom)

    def _min(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
    ) -> Tensor:
        out = xs[0].to(dtype=dtype).clone()

        for x in xs[1:]:
            out = torch.minimum(out, x.to(dtype=dtype))

        return out

    def _max(
        self,
        xs: list[Tensor],
        dtype: torch.dtype,
    ) -> Tensor:
        out = xs[0].to(dtype=dtype).clone()

        for x in xs[1:]:
            out = torch.maximum(out, x.to(dtype=dtype))

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

        return value * weight.to(device=value.device, dtype=value.dtype)

    @staticmethod
    def _get_coeff(weights: Tensor | None, index: int) -> Tensor | None:
        if weights is None:
            return None
        return weights[index]

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
        if self.cast not in {True, False, None}:
            raise TypeError(f"cast must be bool | None, got {type(self.cast)!r}")

        if self.output_dtype is not None and not isinstance(self.output_dtype, torch.dtype):
            raise TypeError(
                f"output_dtype must be torch.dtype | None, got {type(self.output_dtype)!r}"
            )

        dtype = xs[0].dtype

        if not self.strict_dtype:
            for x in xs[1:]:
                dtype = torch.promote_types(dtype, x.dtype)

        if isinstance(weights, Tensor):
            if torch.is_complex(weights):
                raise ValueError("Reducer does not support complex weights.")
            dtype = torch.promote_types(dtype, weights.dtype)

        match self.cast:
            case False:
                return self._infer_safe_dtype(dtype, op=op, weights=weights)
            case True:
                return self.output_dtype or self._get_default_dtype(
                    dtype, op=op, weights=weights
                )
            case None:
                if self._is_cast_needed(dtype, op=op, weights=weights):
                    return self.output_dtype or self._get_default_dtype(
                        dtype, op=op, weights=weights
                    )
                return self._infer_safe_dtype(dtype, op=op, weights=weights)
            case _:
                raise AssertionError(f"Unreachable cast policy: {self.cast!r}")

    def _infer_safe_dtype(
        self,
        dtype: torch.dtype,
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if dtype == torch.bool:
            raise TypeError("Reducer does not support bool tensors.")

        if torch.empty((), dtype=dtype).is_complex():
            if op in self.ORDERED_OPS:
                raise TypeError(f"{op!r} does not support complex tensors.")
            return dtype

        if op == "mean" and not self._is_float(dtype):
            return self.master_float_dtype

        if weights is not None and not self._is_float(dtype):
            return self.master_float_dtype

        return dtype

    def _is_cast_needed(
        self,
        dtype: torch.dtype,
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> bool:
        if dtype == torch.bool:
            raise TypeError("Reducer does not support bool tensors.")

        if dtype in {torch.float16, torch.bfloat16}:
            return op in {"sum", "mean"} or weights is not None

        if dtype == self._has_torch_complex32():
            return True

        if not self._is_float_or_complex(dtype):
            return op == "sum" or op == "mean" or weights is not None

        return False

    def _get_default_dtype(
        self,
        dtype: torch.dtype,
        op: str,
        weights: Sequence[float] | Tensor | None,
    ) -> torch.dtype:
        if dtype == torch.bool:
            raise TypeError("Reducer does not support bool tensors.")

        if dtype in {torch.float16, torch.bfloat16}:
            if op in {"sum", "mean"} or weights is not None:
                return self.master_float_dtype
            return dtype
        elif dtype in {torch.float32, torch.float64}:
            return dtype
        elif dtype == self._has_torch_complex32():
            if op in self.ORDERED_OPS:
                raise TypeError(f"{op!r} does not support complex tensors.")
            return self.master_complex_dtype
        elif dtype in {torch.complex64, torch.complex128}:
            if op in self.ORDERED_OPS:
                raise TypeError(f"{op!r} does not support complex tensors.")
            return dtype
        elif op == "mean" or weights is not None:
            return self.master_float_dtype
        elif op == "sum":
            return self.master_int_dtype
        else:
            return dtype

    def _verify_master_dtypes(self) -> None:
        if self.master_int_dtype == torch.bool:
            raise TypeError("master_int_dtype must be an integer dtype, not bool.")
        int_probe = torch.empty((), dtype=self.master_int_dtype)
        if int_probe.is_quantized:
            raise TypeError("master_int_dtype must not be a quantized dtype.")
        if int_probe.is_floating_point():
            raise TypeError("master_int_dtype must be an integer dtype.")
        elif int_probe.is_complex():
            raise TypeError("master_int_dtype must be an integer dtype.")

        float_probe = torch.empty((), dtype=self.master_float_dtype)
        if not float_probe.is_floating_point():
            raise TypeError("master_float_dtype must be a floating dtype.")

        complex_probe = torch.empty((), dtype=self.master_complex_dtype)
        if not complex_probe.is_complex():
            raise TypeError("master_complex_dtype must be a complex dtype.")


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
            if any(torch.is_complex(torch.as_tensor(weight)) for weight in weights):
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

    @staticmethod
    def _has_torch_complex32() -> torch.dtype | None:
        value = getattr(torch, "complex32", None)
        return value if isinstance(value, torch.dtype) else None

    @staticmethod
    def _is_float(dtype: torch.dtype) -> bool:
        return torch.empty((), dtype=dtype).is_floating_point()

    @staticmethod
    def _is_float_or_complex(dtype: torch.dtype) -> bool:
        probe = torch.empty((), dtype=dtype)
        return probe.is_floating_point() or probe.is_complex()

    def _verify_no_complex(self, xs: Sequence[Tensor], op: str) -> None:
        if any(x.is_complex() for x in xs):
            raise TypeError(f"{op!r} does not support complex tensors.")

    def _verify_no_bool(self, xs: Sequence[Tensor]) -> None:
        if any(x.dtype == torch.bool for x in xs):
            raise TypeError("Reducer does not support bool tensors.")

    def extra_repr(self) -> str:
        return (
            f"strict_shape={self.strict_shape}, "
            f"strict_dtype={self.strict_dtype}, "
            f"strict_device={self.strict_device}, "
            f"eps={self.eps}, "
            f"cast={self.cast}, "
            f"output_dtype={self.output_dtype}, "
            f"master_int_dtype={self.master_int_dtype}, "
            f"master_float_dtype={self.master_float_dtype}, "
            f"master_complex_dtype={self.master_complex_dtype}"
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
