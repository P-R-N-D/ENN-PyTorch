from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from .schema import GraphValue, KeyRef
from .store import KVStore


def _validate_state_key(value: object, field_name: str) -> str:
    label = f"StateRoute.{field_name}"
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value:
        raise ValueError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{label} must not have leading or trailing whitespace.")
    return value


@dataclass(slots=True)
class StateRoute:
    """
    Runtime state key routing for stateful modules.

    ``state_input_key`` is read as an optional module input. If missing, it
    resolves to ``None``. ``state_output_key`` is the secondary output key used
    when the module returns ``(output, next_state)``. ``return_state_key`` is an
    optional KVStore flag; when it is missing, ``return_state`` defaults to
    ``True`` so recurrent heads can be routed with ``NodeSpec.output_keys``.
    """

    state_input_key: str
    state_output_key: str
    state_arg: str = "state"
    return_state_arg: str = "return_state"
    return_state_key: str = "__state.return_state__"

    def __post_init__(self) -> None:
        self.state_input_key = _validate_state_key(
            self.state_input_key,
            "state_input_key",
        )
        self.state_output_key = _validate_state_key(
            self.state_output_key,
            "state_output_key",
        )
        if self.state_input_key == self.state_output_key:
            raise ValueError(
                "StateRoute.state_input_key and state_output_key must be different "
                "to avoid self-dependency in GraphExecutor."
            )

        self.state_arg = _validate_state_key(self.state_arg, "state_arg")
        self.return_state_arg = _validate_state_key(
            self.return_state_arg,
            "return_state_arg",
        )
        if self.state_arg == self.return_state_arg:
            raise ValueError("StateRoute state_arg and return_state_arg must differ.")

        self.return_state_key = _validate_state_key(
            self.return_state_key,
            "return_state_key",
        )
        if self.return_state_key in {self.state_input_key, self.state_output_key}:
            raise ValueError(
                "StateRoute.return_state_key must be distinct from "
                "state_input_key and state_output_key."
            )

    def input_kwargs(
        self,
        existing: Mapping[str, KeyRef] | None = None,
        *,
        state_optional: bool = True,
    ) -> dict[str, KeyRef]:
        if not isinstance(state_optional, bool):
            raise TypeError("StateRoute.input_kwargs state_optional must be a bool.")
        if existing is None:
            out: dict[str, KeyRef] = {}
        elif isinstance(existing, Mapping):
            out = {}
            for key, ref in existing.items():
                norm_key = _validate_state_key(key, "input_kwargs key")
                if not isinstance(ref, KeyRef):
                    raise TypeError(
                        "StateRoute.input_kwargs values must be KeyRef instances."
                    )
                out[norm_key] = ref
        else:
            raise TypeError("StateRoute.input_kwargs existing must be a mapping.")

        conflicts = {self.state_arg, self.return_state_arg} & set(out)
        if conflicts:
            raise ValueError(
                "StateRoute input kwargs conflict with existing keys: "
                f"{sorted(conflicts)!r}"
            )

        out[self.state_arg] = KeyRef(
            self.state_input_key,
            optional=state_optional,
            default=None,
        )
        out[self.return_state_arg] = KeyRef(
            self.return_state_key,
            optional=True,
            default=True,
        )
        return out

    def output_keys(self, primary_output_key: str) -> tuple[str, str]:
        primary_output_key = _validate_state_key(
            primary_output_key,
            "primary_output_key",
        )
        reserved = {
            self.state_input_key,
            self.state_output_key,
            self.return_state_key,
        }
        if primary_output_key in reserved:
            raise ValueError(
                "StateRoute primary_output_key must differ from state/control keys."
            )
        return primary_output_key, self.state_output_key

    def enable_return_state(self, store: KVStore) -> KVStore:
        if not isinstance(store, KVStore):
            raise TypeError(f"StateRoute.enable_return_state expects KVStore, got {type(store)!r}")
        store.set(self.return_state_key, True)
        return store

    def reset(self, store: KVStore, *, missing_ok: bool = True) -> KVStore:
        """
        Clear the routed input state slot from ``store``.

        ``KVStore`` supports parent lookup, so deleting only the local key would
        still expose an inherited parent state. Reset therefore writes a local
        ``None`` value when a state is resolvable, masking any parent value.
        """
        if not isinstance(store, KVStore):
            raise TypeError(f"StateRoute.reset expects KVStore, got {type(store)!r}")
        if not isinstance(missing_ok, bool):
            raise TypeError("StateRoute.reset missing_ok must be a bool.")

        if not store.has(self.state_input_key):
            if missing_ok:
                return store
            raise KeyError(f"KVStore missing key: {self.state_input_key!r}")
        store.set(self.state_input_key, None, origin="StateRoute.reset")
        return store

    @staticmethod
    def _carried_payload(
        payload: Any,
        *,
        detach: bool,
        clone: bool,
    ) -> Any:
        if isinstance(payload, Tensor):
            if detach:
                payload = payload.detach()
            if clone:
                payload = payload.clone()
            return payload

        if isinstance(payload, tuple):
            return tuple(
                StateRoute._carried_payload(item, detach=detach, clone=clone)
                for item in payload
            )
        if isinstance(payload, list):
            return [
                StateRoute._carried_payload(item, detach=detach, clone=clone)
                for item in payload
            ]
        if isinstance(payload, Mapping):
            return {
                key: StateRoute._carried_payload(item, detach=detach, clone=clone)
                for key, item in payload.items()
            }
        return payload

    @staticmethod
    def _carried_value(
        value: GraphValue,
        *,
        detach: bool,
        clone: bool,
    ) -> GraphValue:
        data = value.data
        if not detach and not clone:
            return value

        data = StateRoute._carried_payload(data, detach=detach, clone=clone)
        return GraphValue(
            data=data,
            layout=value.layout,
            mask_key=value.mask_key,
            origin=value.origin,
            meta=dict(value.meta),
        )

    def carry(
        self,
        store: KVStore,
        *,
        missing_ok: bool = False,
        detach: bool = False,
        clone: bool = False,
    ) -> KVStore:
        """
        Copy the routed output state into the routed input state slot.

        ``detach`` and ``clone`` apply recursively to Tensor payloads inside
        tuples, lists, and mappings. Other payloads are carried unchanged. This
        helper does not reset state or
        manage stream lifecycle; those policies belong to a future stream/state
        runner.
        """
        if not isinstance(store, KVStore):
            raise TypeError(f"StateRoute.carry expects KVStore, got {type(store)!r}")
        if not isinstance(missing_ok, bool):
            raise TypeError("StateRoute.carry missing_ok must be a bool.")
        if not isinstance(detach, bool):
            raise TypeError("StateRoute.carry detach must be a bool.")
        if not isinstance(clone, bool):
            raise TypeError("StateRoute.carry clone must be a bool.")

        try:
            value = store.get_value(self.state_output_key)
        except KeyError:
            if missing_ok:
                return store
            raise
        store.set_value(self.state_input_key, self._carried_value(value, detach=detach, clone=clone))
        return store
