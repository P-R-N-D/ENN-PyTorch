from __future__ import annotations

import pytest
import torch

from enn_torch_dev.executor import GraphValue, KVStore, KeyRef


def test_kv_store_set_get_and_metadata() -> None:
    x = torch.randn(2, 3)
    store = KVStore()

    store.set(
        "input.x",
        x,
        layout="B,D",
        mask_key="input.mask",
        origin="test",
        meta={"kind": "feature"},
    )

    assert store.has("input.x")
    assert torch.equal(store.get("input.x"), x)

    value = store.get_value("input.x")
    assert isinstance(value, GraphValue)
    assert value.layout == "B,D"
    assert value.mask_key == "input.mask"
    assert value.origin == "test"
    assert value.meta == {"kind": "feature"}


def test_kv_store_set_value_preserves_graph_value() -> None:
    x = torch.randn(2, 3)
    value = GraphValue(data=x, layout="B,D", origin="prepared")
    store = KVStore()

    store.set_value("prepared.x", value)

    assert store.get_value("prepared.x") is value
    assert torch.equal(store.get("prepared.x"), x)


def test_kv_store_set_value_rejects_non_graph_value() -> None:
    store = KVStore()

    with pytest.raises(TypeError, match="GraphValue"):
        store.set_value("x", torch.tensor(1.0))


def test_kv_store_resolve_keyref() -> None:
    x = torch.randn(2, 3)
    store = KVStore({"x": x})

    assert torch.equal(store.resolve(KeyRef("x")), x)


def test_keyref_rejects_non_string_key() -> None:
    with pytest.raises(TypeError, match="string"):
        KeyRef(123)


def test_keyref_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        KeyRef("")


def test_keyref_rejects_surrounding_whitespace_key() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        KeyRef(" x ")


def test_kv_store_has_returns_false_for_invalid_keys() -> None:
    store = KVStore({"x": torch.randn(2, 3)})

    assert not store.has("")
    assert not store.has(" x ")
    assert "" not in store
    assert " x " not in store


def test_kv_store_optional_keyref_returns_default() -> None:
    store = KVStore()

    assert store.resolve(KeyRef("missing", optional=True, default=123)) == 123


def test_kv_store_missing_key_raises() -> None:
    store = KVStore()

    with pytest.raises(KeyError, match="missing"):
        store.get("missing")


def test_kv_store_rejects_empty_key() -> None:
    store = KVStore()

    with pytest.raises(ValueError, match="non-empty"):
        store.set("", torch.tensor(1.0))


def test_kv_store_rejects_non_string_key() -> None:
    store = KVStore()

    with pytest.raises(TypeError, match="string"):
        store.get(123)


@pytest.mark.parametrize("bad_key", [" x", "x ", "\tx"])
def test_kv_store_rejects_surrounding_whitespace_key(bad_key: str) -> None:
    store = KVStore({"x": torch.randn(2, 3)})

    with pytest.raises(ValueError, match="whitespace"):
        store.get(bad_key)

    with pytest.raises(ValueError, match="whitespace"):
        store.set(bad_key, torch.tensor(1.0))

    with pytest.raises(ValueError, match="whitespace"):
        store.delete(bad_key)


def test_kv_store_to_moves_tensor_values_only() -> None:
    x = torch.randn(2, 3)
    store = KVStore({"x": x, "name": "raw"})

    store.to("cpu")

    assert store.get("x").device.type == "cpu"
    assert store.get("name") == "raw"


def test_kv_store_shallow_copy_preserves_values() -> None:
    store = KVStore({"x": torch.randn(2, 3)})
    copied = store.shallow_copy()

    assert copied is not store
    assert torch.equal(copied.get("x"), store.get("x"))
