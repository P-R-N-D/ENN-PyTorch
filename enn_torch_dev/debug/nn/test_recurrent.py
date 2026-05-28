from __future__ import annotations

import pytest
import torch

from enn_torch_dev.nn import RecurrentContextHead


def test_recurrent_context_head_preserves_batch_first_shape() -> None:
    head = RecurrentContextHead(input_dim=8)
    x = torch.randn(2, 5, 8)

    out = head(x)

    assert out.shape == x.shape


def test_recurrent_context_head_returns_state() -> None:
    head = RecurrentContextHead(input_dim=8, hidden_dim=6, num_layers=2)
    x = torch.randn(2, 5, 8)

    out, state = head(x, return_state=True)

    assert out.shape == x.shape
    assert state.shape == (2, 2, 6)


def test_recurrent_context_head_accepts_initial_state() -> None:
    head = RecurrentContextHead(input_dim=8, hidden_dim=6, num_layers=2)
    x = torch.randn(2, 5, 8)
    state = torch.zeros(2, 2, 6)

    out, next_state = head(x, state, return_state=True)

    assert out.shape == x.shape
    assert next_state.shape == state.shape


def test_recurrent_context_head_supports_time_first_layout() -> None:
    head = RecurrentContextHead(input_dim=8, batch_first=False)
    x = torch.randn(5, 2, 8)

    out, state = head(x, return_state=True)

    assert out.shape == x.shape
    assert state.shape == (1, 2, 8)


def test_recurrent_context_head_supports_no_residual() -> None:
    head = RecurrentContextHead(input_dim=8, residual=False)
    x = torch.randn(2, 5, 8)

    out = head(x)

    assert out.shape == x.shape


def test_recurrent_context_head_supports_no_norm() -> None:
    head = RecurrentContextHead(input_dim=8, norm=False)
    x = torch.randn(2, 5, 8)

    out = head(x)

    assert out.shape == x.shape


def test_recurrent_context_head_projects_hidden_dim_to_input_dim() -> None:
    head = RecurrentContextHead(input_dim=8, hidden_dim=5)
    x = torch.randn(2, 5, 8)

    out = head(x)

    assert out.shape == x.shape


def test_recurrent_context_head_backpropagates() -> None:
    head = RecurrentContextHead(input_dim=8)
    x = torch.randn(2, 5, 8, requires_grad=True)

    loss = head(x).sum()
    loss.backward()

    assert x.grad is not None
    assert any(param.grad is not None for param in head.parameters())


def test_recurrent_context_head_rejects_bad_input() -> None:
    head = RecurrentContextHead(input_dim=8)

    with pytest.raises(TypeError, match="Tensor"):
        head(object())

    with pytest.raises(ValueError, match="3D"):
        head(torch.randn(2, 8))

    with pytest.raises(ValueError, match="input_dim"):
        head(torch.randn(2, 5, 7))

    with pytest.raises(TypeError, match="floating"):
        head(torch.ones(2, 5, 8, dtype=torch.int64))


def test_recurrent_context_head_rejects_bad_state() -> None:
    head = RecurrentContextHead(input_dim=8, hidden_dim=6, num_layers=2)
    x = torch.randn(2, 5, 8)

    with pytest.raises(TypeError, match="state"):
        head(x, object())

    with pytest.raises(ValueError, match="state"):
        head(x, torch.zeros(1, 2, 6))

    with pytest.raises(ValueError, match="same dtype"):
        head(x, torch.zeros(2, 2, 6, dtype=torch.float64))


def test_recurrent_context_head_validates_constructor_arguments() -> None:
    with pytest.raises(TypeError, match="input_dim"):
        RecurrentContextHead(input_dim=True)

    with pytest.raises(ValueError, match="input_dim"):
        RecurrentContextHead(input_dim=0)

    with pytest.raises(TypeError, match="hidden_dim"):
        RecurrentContextHead(input_dim=8, hidden_dim=False)

    with pytest.raises(ValueError, match="hidden_dim"):
        RecurrentContextHead(input_dim=8, hidden_dim=0)

    with pytest.raises(TypeError, match="num_layers"):
        RecurrentContextHead(input_dim=8, num_layers=True)

    with pytest.raises(ValueError, match="num_layers"):
        RecurrentContextHead(input_dim=8, num_layers=0)

    with pytest.raises(ValueError, match="dropout"):
        RecurrentContextHead(input_dim=8, dropout=float("nan"))

    with pytest.raises(ValueError, match="dropout"):
        RecurrentContextHead(input_dim=8, dropout=1.5)

    with pytest.raises(TypeError, match="batch_first"):
        RecurrentContextHead(input_dim=8, batch_first=1)

    with pytest.raises(TypeError, match="residual"):
        RecurrentContextHead(input_dim=8, residual=1)

    with pytest.raises(TypeError, match="norm"):
        RecurrentContextHead(input_dim=8, norm=1)


def test_recurrent_context_head_rejects_bad_return_state() -> None:
    head = RecurrentContextHead(input_dim=8)
    x = torch.randn(2, 5, 8)

    with pytest.raises(TypeError, match="return_state"):
        head(x, return_state=1)
