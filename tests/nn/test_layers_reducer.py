import pytest
import torch

from enn_torch_dev.nn.layers import Reducer


def _sample_tensors(dtype=torch.float32, device=None):
    device = device or torch.device("cpu")
    return [
        torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=dtype, device=device),
        torch.tensor([[0.5, 5.0], [-1.0, 2.0]], dtype=dtype, device=device),
        torch.tensor([[2.0, -3.0], [0.0, 1.0]], dtype=dtype, device=device),
    ]


def test_basic_ops_match_torch():
    xs = _sample_tensors()
    reducer = Reducer()

    assert torch.allclose(reducer(xs, op="sum"), torch.stack(xs, dim=0).sum(dim=0))
    assert torch.allclose(reducer(xs, op="mean"), torch.stack(xs, dim=0).mean(dim=0), atol=1e-7, rtol=1e-6)
    assert torch.allclose(reducer(xs, op="min"), torch.stack(xs, dim=0).amin(dim=0))
    assert torch.allclose(reducer(xs, op="max"), torch.stack(xs, dim=0).amax(dim=0))


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max"])
def test_chunked_matches_non_chunked(op):
    xs = _sample_tensors()
    reducer = Reducer()
    non_chunked = reducer(xs, op=op)
    chunked = reducer(xs, op=op, chunk_size=2)
    assert torch.allclose(chunked, non_chunked, atol=1e-7, rtol=1e-6)


def test_weighted_sum_matches_manual():
    xs = _sample_tensors()
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    reducer = Reducer()

    got = reducer(xs, op="sum", weights=weights)
    expected = sum(x * w for x, w in zip(xs, weights))
    assert torch.allclose(got, expected)


def test_weighted_mean_matches_manual_formula():
    xs = _sample_tensors()
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    reducer = Reducer()

    got = reducer(xs, op="mean", weights=weights)
    expected = sum(x * w for x, w in zip(xs, weights)) / weights.sum()
    assert torch.allclose(got, expected)


def test_single_tensor_input_rejected():
    x = torch.randn(2, 2)
    reducer = Reducer()
    with pytest.raises(TypeError, match="not a single Tensor"):
        reducer(x)


@pytest.mark.parametrize("bad", [torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("inf")])])
def test_tensor_weights_nan_or_inf_rejected(bad):
    xs = _sample_tensors()[:2]
    reducer = Reducer()
    with pytest.raises(ValueError, match="finite"):
        reducer(xs, op="sum", weights=bad)


@pytest.mark.parametrize("bad", [[1.0, float("nan")], [1.0, float("inf")]])
def test_sequence_weights_nan_or_inf_rejected(bad):
    xs = _sample_tensors()[:2]
    reducer = Reducer()
    with pytest.raises(ValueError, match="finite"):
        reducer(xs, op="sum", weights=bad)


@pytest.mark.parametrize("weights", [torch.tensor([1.0, -1.0]), torch.tensor([1e-20, -1e-20])])
def test_weighted_mean_zero_or_near_zero_sum_rejected(weights):
    xs = _sample_tensors()[:2]
    reducer = Reducer(eps=1e-12)
    with pytest.raises(ValueError, match="non-zero sum"):
        reducer(xs, op="mean", weights=weights)


@pytest.mark.parametrize("op", ["min", "max"])
def test_min_max_reject_weights(op):
    xs = _sample_tensors()
    reducer = Reducer()
    with pytest.raises(ValueError, match="does not support weights"):
        reducer(xs, op=op, weights=[1.0, 1.0, 1.0])


@pytest.mark.parametrize("op", ["min", "max"])
def test_min_max_reject_complex(op):
    xs = [
        torch.tensor([1 + 2j, 2 + 3j], dtype=torch.complex64),
        torch.tensor([0 + 1j, 3 + 1j], dtype=torch.complex64),
    ]
    reducer = Reducer()
    with pytest.raises(TypeError, match="does not support complex"):
        reducer(xs, op=op)


def test_strict_shape_false_allows_broadcastable_shapes():
    xs = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([10.0, 20.0]),
        torch.tensor([[0.5], [1.5]]),
    ]
    reducer = Reducer(strict_shape=False)
    got = reducer(xs, op="sum")
    expected = xs[0] + xs[1] + xs[2]
    assert torch.allclose(got, expected)


def test_strict_dtype_false_promotes_dtype():
    xs = [
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([0.5, 1.5], dtype=torch.float32),
    ]
    reducer = Reducer(strict_dtype=False)
    out = reducer(xs, op="sum")
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.tensor([1.5, 3.5], dtype=torch.float32))


def test_sum_integer_input_no_weights_outputs_int64():
    xs = [
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([3, 4], dtype=torch.int32),
    ]
    reducer = Reducer()
    out = reducer(xs, op="sum")
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([4, 6], dtype=torch.int64))


def _available_non_cpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_strict_device_false_moves_to_first_tensor_device():
    other = _available_non_cpu_device()
    if other is None:
        pytest.skip("CUDA/MPS unavailable; skipping mixed-device test")

    cpu = torch.device("cpu")
    first = torch.tensor([1.0, 2.0], device=other)
    second = torch.tensor([3.0, 4.0], device=cpu)

    reducer = Reducer(strict_device=False)
    out = reducer([first, second], op="sum")

    assert out.device == first.device
    expected = first + second.to(other)
    assert torch.allclose(out, expected)
