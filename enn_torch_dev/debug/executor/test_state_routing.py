from __future__ import annotations

import pytest
import torch
from torch import nn

from enn_torch_dev.executor import (
    GraphExecutor,
    GraphValue,
    KVStore,
    KeyRef,
    NodeSpec,
    StateRoute,
)
from enn_torch_dev.nn import RecurrentContextHead


class _StateProducer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, x.shape[0], 3, dtype=x.dtype, device=x.device)


def test_state_route_builds_input_kwargs_and_output_keys() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")

    kwargs = route.input_kwargs()

    assert set(kwargs) == {"state", "return_state"}
    assert kwargs["state"] == KeyRef("ctx.state.in", optional=True, default=None)
    assert kwargs["return_state"] == KeyRef(
        "__state.return_state__",
        optional=True,
        default=True,
    )
    assert route.output_keys("ctx.out") == ("ctx.out", "ctx.state.out")


def test_state_route_merges_existing_input_kwargs() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")

    kwargs = route.input_kwargs({"x_scale": KeyRef("scale")})

    assert kwargs["x_scale"] == KeyRef("scale")
    assert kwargs["state"] == KeyRef("ctx.state.in", optional=True, default=None)
    assert kwargs["return_state"].default is True


def test_state_route_rejects_conflicting_input_kwargs() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")

    with pytest.raises(ValueError, match="conflict"):
        route.input_kwargs({"state": KeyRef("other.state")})

    with pytest.raises(ValueError, match="conflict"):
        route.input_kwargs({"return_state": KeyRef("flag")})


def test_state_route_enable_return_state_writes_flag() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out", return_state_key="flag")
    store = KVStore()

    returned = route.enable_return_state(store)

    assert returned is store
    assert store.get("flag") is True




def test_state_route_can_make_state_input_required() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")

    kwargs = route.input_kwargs(state_optional=False)

    assert kwargs["state"] == KeyRef("ctx.state.in", optional=False, default=None)


def test_state_route_rejects_bad_state_optional_flag() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")

    with pytest.raises(TypeError, match="state_optional"):
        route.input_kwargs(state_optional=1)

def test_state_route_validates_constructor_arguments() -> None:
    with pytest.raises(TypeError, match="state_input_key"):
        StateRoute(1, "state.out")

    with pytest.raises(ValueError, match="different"):
        StateRoute("state", "state")

    with pytest.raises(ValueError, match="differ"):
        StateRoute("state.in", "state.out", state_arg="state", return_state_arg="state")

    with pytest.raises(ValueError, match="return_state_key"):
        StateRoute("state.in", "state.out", return_state_key="state.in")

    with pytest.raises(ValueError, match="return_state_key"):
        StateRoute("state.in", "state.out", return_state_key="state.out")

    with pytest.raises(ValueError, match="primary_output_key"):
        StateRoute("state.in", "state.out").output_keys("state.out")

    with pytest.raises(ValueError, match="primary_output_key"):
        StateRoute("state.in", "state.out").output_keys("state.in")

    with pytest.raises(ValueError, match="primary_output_key"):
        StateRoute("state.in", "state.out").output_keys("__state.return_state__")

    with pytest.raises(TypeError, match="KVStore"):
        StateRoute("state.in", "state.out").enable_return_state(object())


def test_state_route_runs_recurrent_node_without_initial_state() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="ctx",
            input_args=[KeyRef("x")],
            input_kwargs=route.input_kwargs(),
            output_key="ctx.out",
            output_keys=route.output_keys("ctx.out"),
        ),
        RecurrentContextHead(input_dim=4, hidden_dim=3),
    )
    store = KVStore({"x": torch.randn(2, 5, 4)})

    graph.run(store)

    assert store.get("ctx.out").shape == (2, 5, 4)
    assert store.get("ctx.state.out").shape == (1, 2, 3)


def test_state_route_runs_recurrent_node_with_initial_state() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="ctx",
            input_args=[KeyRef("x")],
            input_kwargs=route.input_kwargs(),
            output_key="ctx.out",
            output_keys=route.output_keys("ctx.out"),
        ),
        RecurrentContextHead(input_dim=4, hidden_dim=3),
    )
    store = KVStore(
        {
            "x": torch.randn(2, 5, 4),
            "ctx.state.in": torch.zeros(1, 2, 3),
        }
    )

    graph.run(store)

    assert store.get("ctx.out").shape == (2, 5, 4)
    assert store.get("ctx.state.out").shape == (1, 2, 3)


def test_state_route_can_carry_state_between_runs_manually() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="ctx",
            input_args=[KeyRef("x")],
            input_kwargs=route.input_kwargs(),
            output_key="ctx.out",
            output_keys=route.output_keys("ctx.out"),
        ),
        RecurrentContextHead(input_dim=4, hidden_dim=3),
    )
    store = KVStore({"x": torch.randn(2, 5, 4)})

    graph.run(store)
    first_state = store.get("ctx.state.out")
    store.set("ctx.state.in", first_state)
    graph.run(store)

    assert store.get("ctx.state.out").shape == first_state.shape


def test_state_route_carry_copies_output_state_to_input_state() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    store = KVStore()
    state = torch.randn(1, 2, 3)
    store.set("ctx.state.out", state)

    returned = route.carry(store)

    assert returned is store
    assert torch.equal(store.get("ctx.state.in"), state)


def test_state_route_carry_preserves_graph_value_metadata() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    store = KVStore()
    value = GraphValue(
        data=torch.randn(1, 2, 3),
        layout="state",
        origin="ctx",
        meta={"step": 1},
    )
    store.set_value("ctx.state.out", value)

    route.carry(store)

    carried = store.get_value("ctx.state.in")
    assert carried is value
    assert carried.layout == "state"
    assert carried.origin == "ctx"
    assert carried.meta == {"step": 1}


def test_state_route_carry_missing_output_raises_by_default() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")

    with pytest.raises(KeyError, match="ctx.state.out"):
        route.carry(KVStore())


def test_state_route_carry_missing_output_can_be_ignored() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    store = KVStore()

    returned = route.carry(store, missing_ok=True)

    assert returned is store
    assert not store.has("ctx.state.in")


def test_state_route_carry_validates_inputs() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")

    with pytest.raises(TypeError, match="KVStore"):
        route.carry(object())

    with pytest.raises(TypeError, match="missing_ok"):
        route.carry(KVStore(), missing_ok=1)


def test_state_route_carry_feeds_next_recurrent_run() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="ctx",
            input_args=[KeyRef("x")],
            input_kwargs=route.input_kwargs(),
            output_key="ctx.out",
            output_keys=route.output_keys("ctx.out"),
        ),
        RecurrentContextHead(input_dim=4, hidden_dim=3),
    )
    store = KVStore({"x": torch.randn(2, 5, 4)})

    graph.run(store)
    first_state = store.get("ctx.state.out")
    route.carry(store)
    graph.run(store)

    assert torch.equal(store.get("ctx.state.in"), first_state)
    assert store.get("ctx.state.out").shape == first_state.shape




def test_state_route_required_state_preserves_graph_dependency() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="ctx",
            input_args=[KeyRef("x")],
            input_kwargs=route.input_kwargs(state_optional=False),
            output_key="ctx.out",
            output_keys=route.output_keys("ctx.out"),
        ),
        RecurrentContextHead(input_dim=4, hidden_dim=3),
    )
    graph.add_node(
        NodeSpec(
            name="state_producer",
            input_args=[KeyRef("x")],
            output_key="ctx.state.in",
        ),
        _StateProducer(),
    )

    assert graph.execution_order() == ("state_producer", "ctx")
    assert graph.dependency_names("ctx") == ("state_producer",)

    store = KVStore({"x": torch.randn(2, 5, 4)})
    graph.run(store)

    assert store.get("ctx.out").shape == (2, 5, 4)
    assert store.get("ctx.state.in").shape == (1, 2, 3)
    assert store.get("ctx.state.out").shape == (1, 2, 3)

def test_state_route_enable_return_state_is_optional() -> None:
    route = StateRoute("ctx.state.in", "ctx.state.out", return_state_key="flag")
    graph = GraphExecutor()
    graph.add_node(
        NodeSpec(
            name="ctx",
            input_args=[KeyRef("x")],
            input_kwargs=route.input_kwargs(),
            output_key="ctx.out",
            output_keys=route.output_keys("ctx.out"),
        ),
        RecurrentContextHead(input_dim=4, hidden_dim=3),
    )
    store = KVStore({"x": torch.randn(2, 5, 4)})
    route.enable_return_state(store)

    graph.run(store)

    assert store.get("ctx.out").shape == (2, 5, 4)
    assert store.get("ctx.state.out").shape == (1, 2, 3)
