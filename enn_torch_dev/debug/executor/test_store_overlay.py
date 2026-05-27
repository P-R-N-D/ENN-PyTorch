from __future__ import annotations

import pytest

from enn_torch_dev.executor.store import KVStore


def test_fork_reads_parent_keys() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork()

    assert child.has("x")
    assert child.get("x") == 1


def test_fork_set_does_not_mutate_parent() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork()
    child.set("y", 2)

    assert not parent.has("y")
    assert child.get("y") == 2


def test_local_override_behavior() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork()
    child.set("x", 100)

    assert parent.get("x") == 1
    assert child.get("x") == 100


def test_keys_include_parent() -> None:
    parent = KVStore({"a": 1, "b": 2})
    child = parent.fork({"b": 20, "c": 3})

    assert set(child.local_keys()) == {"b", "c"}
    assert set(child.keys(include_parent=True)) == {"a", "b", "c"}


def test_parent_metadata_fallback() -> None:
    parent = KVStore()
    parent.set("x", 1, meta={"role": "parent"})
    child = parent.fork()

    gv = child.get_value("x")
    assert gv.meta["role"] == "parent"


def test_commit_to_default_selected_and_no_overwrite() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork({"x": 10, "y": 20})

    # default: commit only local keys
    target = KVStore({"x": 999})
    child.commit_to(target)
    assert target.get("x") == 10
    assert target.get("y") == 20

    # selected keys + overwrite=False
    target2 = KVStore({"x": 999})
    child.commit_to(target2, keys=("x", "y"), overwrite=False)
    assert target2.get("x") == 999
    assert target2.get("y") == 20


def test_commit_to_overwrite_false_respects_target_parent_visible_key() -> None:
    source = KVStore({"x": 10})
    target_parent = KVStore({"x": 999})
    target = target_parent.fork()

    source.commit_to(target, overwrite=False)

    assert target.local_keys() == ()
    assert target.get("x") == 999


def test_commit_rejects_parent_only_key() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork()
    target = KVStore()

    with pytest.raises(KeyError):
        child.commit_to(target, keys=("x",))


def test_delete_is_local_only() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork({"y": 2})

    child.delete("y")
    assert not child.has("y")
    assert parent.has("x")

    with pytest.raises(KeyError):
        child.delete("x")


def test_optional_default_fallback_from_parent() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork()

    assert child.get("x", optional=True, default=-1) == 1
    assert child.get("missing", optional=True, default=-1) == -1


def test_shallow_copy_preserves_parent() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork({"y": 2})
    copied = child.shallow_copy()

    assert copied.parent is parent
    assert copied.get("x") == 1
    assert copied.get("y") == 2


def test_clear_is_local_only() -> None:
    parent = KVStore({"x": 1})
    child = parent.fork({"y": 2})

    child.clear()
    assert child.get("x") == 1
    with pytest.raises(KeyError):
        child.get("y")


def test_to_moves_local_only_by_default() -> None:
    import torch

    parent = KVStore({"x": torch.tensor([1.0])})
    child = parent.fork({"y": torch.tensor([2.0])})

    before_parent = id(parent.get("x"))
    child.to("cpu")
    after_parent = id(parent.get("x"))

    assert before_parent == after_parent
