from __future__ import annotations

import pytest

from enn_torch_dev.executor import ExecutorModeSpec, ModelExecutionSpec


def test_model_execution_spec_defaults_to_plain_local_mode() -> None:
    spec = ModelExecutionSpec()

    assert spec.context == "local"
    assert not spec.tile
    assert not spec.stateful
    assert spec.tile_shape is None
    assert spec.executor_mode == ExecutorModeSpec()


def test_model_execution_spec_maps_local_tiled_mode() -> None:
    spec = ModelExecutionSpec(
        context="local",
        tile=True,
        tile_shape=[2, 3],
        tile_stride=[1, 2],
        tile_dims=[-2, -1],
    )

    assert spec.tile_shape == (2, 3)
    assert spec.tile_stride == (1, 2)
    assert spec.tile_dims == (-2, -1)
    assert spec.to_executor_mode_spec() == ExecutorModeSpec(tile=True)


def test_model_execution_spec_maps_global_local_mode() -> None:
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        tile_shape=(4, 4),
    )

    assert spec.uses_global_local
    assert spec.executor_mode == ExecutorModeSpec(tile=True, global_local=True)


def test_model_execution_spec_maps_stateful_local_mode() -> None:
    spec = ModelExecutionSpec(stateful=True)

    assert spec.executor_mode == ExecutorModeSpec(stream=True)


def test_model_execution_spec_maps_stateful_tiled_mode() -> None:
    spec = ModelExecutionSpec(
        tile=True,
        stateful=True,
        tile_shape=(8,),
    )

    assert spec.executor_mode == ExecutorModeSpec(tile=True, stream=True)


def test_model_execution_spec_maps_stateful_global_local_mode() -> None:
    spec = ModelExecutionSpec(
        context="global_local",
        tile=True,
        stateful=True,
        tile_shape=(8, 8),
    )

    assert spec.executor_mode == ExecutorModeSpec(
        tile=True,
        stream=True,
        global_local=True,
    )


def test_model_execution_spec_requires_tile_shape_when_tiling() -> None:
    with pytest.raises(ValueError, match="tile=True requires tile_shape"):
        ModelExecutionSpec(tile=True)


def test_model_execution_spec_rejects_tile_options_when_tile_is_disabled() -> None:
    with pytest.raises(ValueError, match="require tile=True"):
        ModelExecutionSpec(tile_shape=(4,))

    with pytest.raises(ValueError, match="require tile=True"):
        ModelExecutionSpec(tile_stride=(4,))

    with pytest.raises(ValueError, match="require tile=True"):
        ModelExecutionSpec(tile_dims=(-1,))


def test_model_execution_spec_requires_tile_for_global_local_context() -> None:
    with pytest.raises(ValueError, match="global_local.*requires tile=True"):
        ModelExecutionSpec(context="global_local")


def test_model_execution_spec_validates_context() -> None:
    with pytest.raises(TypeError, match="context"):
        ModelExecutionSpec(context=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="local.*global_local"):
        ModelExecutionSpec(context="global")


def test_model_execution_spec_validates_bool_flags() -> None:
    with pytest.raises(TypeError, match="tile"):
        ModelExecutionSpec(tile=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="stateful"):
        ModelExecutionSpec(stateful="yes")  # type: ignore[arg-type]


def test_model_execution_spec_validates_tile_shape() -> None:
    with pytest.raises(TypeError, match="tile_shape"):
        ModelExecutionSpec(tile=True, tile_shape=4)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tile_shape"):
        ModelExecutionSpec(tile=True, tile_shape=())

    with pytest.raises(TypeError, match="tile_shape"):
        ModelExecutionSpec(tile=True, tile_shape=(True,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive"):
        ModelExecutionSpec(tile=True, tile_shape=(0,))


def test_model_execution_spec_validates_tile_stride() -> None:
    with pytest.raises(TypeError, match="tile_stride"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_stride=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tile_stride length"):
        ModelExecutionSpec(tile=True, tile_shape=(4, 4), tile_stride=(2,))

    with pytest.raises(TypeError, match="tile_stride"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_stride=(False,))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_stride=(-1,))


def test_model_execution_spec_validates_tile_dims() -> None:
    with pytest.raises(TypeError, match="tile_dims"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_dims=-1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tile_dims length"):
        ModelExecutionSpec(tile=True, tile_shape=(4, 4), tile_dims=(-1,))

    with pytest.raises(TypeError, match="tile_dims"):
        ModelExecutionSpec(tile=True, tile_shape=(4,), tile_dims=(True,))  # type: ignore[arg-type]
