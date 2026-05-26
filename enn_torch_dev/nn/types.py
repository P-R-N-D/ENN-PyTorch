from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ContextSummary:
    """
    Packed coarse context produced by ``Composer``.

    ``tokens`` is the dense sequence passed to global attention.
    ``attn_bias`` is an optional key-side salience bias with shape
    ``(B, 1, 1, T)``. Attention implementations may add it to attention
    logits before softmax.

    Fields:
      - tokens: dense global context tokens, shaped ``(B, T, D)``.
      - token_mask: optional valid-token mask, shaped ``(B, T)``.
      - attn_bias: optional key-side attention bias, shaped ``(B, 1, 1, T)``.
      - salience: optional salience value per token, shaped ``(B, T)``.
      - score: optional raw salience score per token, shaped ``(B, T)``.
      - original_shape: compressed context shape ``(B, R, K, D)`` before
        flattening.
    """

    tokens: Tensor
    token_mask: Tensor | None
    attn_bias: Tensor | None
    salience: Tensor | None
    score: Tensor | None
    original_shape: tuple[int, int, int, int]
