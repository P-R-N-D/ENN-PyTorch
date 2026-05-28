from __future__ import annotations

import math

import pytest
import torch

from enn_torch_dev.nn import LocalGlobalFusion


def test_local_global_fusion_init_logit_zero_averages_outputs() -> None:
    module = LocalGlobalFusion(init_logit=0.0)
    global_out = torch.zeros(2, 3)
    local_out = torch.ones(2, 3)

    out = module(global_out, local_out)

    assert torch.allclose(out, torch.full((2, 3), 0.5))


def test_local_global_fusion_uses_sigmoid_gate() -> None:
    module = LocalGlobalFusion(init_logit=math.log(3.0))
    global_out = torch.zeros(2, 3)
    local_out = torch.full((2, 3), 4.0)

    out = module(global_out, local_out)

    assert torch.allclose(out, torch.full((2, 3), 3.0))
    assert torch.allclose(module.gate, torch.tensor(0.75))


def test_local_global_fusion_registers_learnable_parameter() -> None:
    module = LocalGlobalFusion(learnable=True)

    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    assert set(params) == {"logit"}
    assert params["logit"].requires_grad
    assert buffers == {}


def test_local_global_fusion_can_use_fixed_buffer() -> None:
    module = LocalGlobalFusion(learnable=False)

    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())

    assert params == {}
    assert set(buffers) == {"logit"}
    assert not buffers["logit"].requires_grad


def test_local_global_fusion_backpropagates_to_inputs_and_gate() -> None:
    module = LocalGlobalFusion(learnable=True)
    global_out = torch.zeros(2, 3, requires_grad=True)
    local_out = torch.ones(2, 3, requires_grad=True)

    loss = module(global_out, local_out).sum()
    loss.backward()

    assert global_out.grad is not None
    assert local_out.grad is not None
    assert module.logit.grad is not None


def test_local_global_fusion_rejects_shape_mismatch() -> None:
    module = LocalGlobalFusion()

    with pytest.raises(ValueError, match="same shape"):
        module(torch.zeros(2, 3), torch.zeros(2, 4))


def test_local_global_fusion_rejects_dtype_mismatch() -> None:
    module = LocalGlobalFusion()

    with pytest.raises(ValueError, match="same dtype"):
        module(
            torch.zeros(2, 3, dtype=torch.float32),
            torch.zeros(2, 3, dtype=torch.float64),
        )


def test_local_global_fusion_rejects_non_floating_dtype() -> None:
    module = LocalGlobalFusion()

    with pytest.raises(TypeError, match="floating dtype"):
        module(torch.zeros(2, 3, dtype=torch.int64), torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(TypeError, match="floating dtype"):
        module(torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool))


def test_local_global_fusion_rejects_non_tensor_inputs() -> None:
    module = LocalGlobalFusion()

    with pytest.raises(TypeError, match="global_out"):
        module(object(), torch.zeros(2, 3))
    with pytest.raises(TypeError, match="local_out"):
        module(torch.zeros(2, 3), object())


def test_local_global_fusion_validates_constructor_arguments() -> None:
    with pytest.raises(TypeError, match="init_logit"):
        LocalGlobalFusion(init_logit="0.0")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="learnable"):
        LocalGlobalFusion(learnable=1)  # type: ignore[arg-type]
