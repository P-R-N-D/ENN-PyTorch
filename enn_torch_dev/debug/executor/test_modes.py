from __future__ import annotations

import pytest

from enn_torch_dev.executor import ExecutorModeSpec


def test_executor_mode_spec_defaults_to_plain_mode() -> None:
    spec = ExecutorModeSpec()

    assert spec.is_plain
    assert spec.mode_names == ("plain",)


def test_executor_mode_spec_represents_tile_mode() -> None:
    spec = ExecutorModeSpec(tile=True)

    assert not spec.is_plain
    assert spec.mode_names == ("tile",)


def test_executor_mode_spec_represents_stream_mode() -> None:
    spec = ExecutorModeSpec(stream=True)

    assert not spec.is_plain
    assert spec.mode_names == ("stream",)


def test_executor_mode_spec_allows_tile_and_stream_composition() -> None:
    spec = ExecutorModeSpec(tile=True, stream=True)

    assert not spec.is_plain
    assert spec.mode_names == ("tile", "stream")


def test_executor_mode_spec_requires_tile_for_global_local() -> None:
    with pytest.raises(ValueError, match="requires tile=True"):
        ExecutorModeSpec(global_local=True)


def test_executor_mode_spec_represents_global_local_tile_mode() -> None:
    spec = ExecutorModeSpec(tile=True, global_local=True)

    assert not spec.is_plain
    assert spec.mode_names == ("tile", "global_local")


def test_executor_mode_spec_allows_stream_with_global_local_tile_mode() -> None:
    spec = ExecutorModeSpec(tile=True, stream=True, global_local=True)

    assert not spec.is_plain
    assert spec.mode_names == ("tile", "stream", "global_local")


def test_executor_mode_spec_rejects_non_bool_flags() -> None:
    with pytest.raises(TypeError, match="tile"):
        ExecutorModeSpec(tile=1)

    with pytest.raises(TypeError, match="stream"):
        ExecutorModeSpec(stream="yes")

    with pytest.raises(TypeError, match="global_local"):
        ExecutorModeSpec(global_local=None)
