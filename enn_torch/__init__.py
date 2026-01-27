# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensordict import TensorDictBase

    from .nn.architecture import Model

__all__ = [
    "config",
    "core",
    "data",
    "nn",
    "runtime",
    "load_model",
    "new_model",
    "predict",
    "save_model",
    "train",
]


class _ConfigModuleLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = ModuleType(spec.name)
        module.__spec__ = spec
        module.__loader__ = self
        return module

    def exec_module(self, module: ModuleType) -> None:
        core_module = importlib.import_module("enn_torch.core.config")
        module.__dict__.update(core_module.__dict__)
        module.__dict__["__name__"] = module.__spec__.name if module.__spec__ else module.__name__
        module.__dict__["__package__"] = "enn_torch"


class _ConfigModuleFinder(importlib.abc.MetaPathFinder):
    _TARGET = "enn_torch.config"

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self._TARGET:
            return None
        return importlib.util.spec_from_loader(fullname, _ConfigModuleLoader())


def _install_config_alias() -> None:
    if "enn_torch.config" in sys.modules:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _ConfigModuleFinder):
            return
    sys.meta_path.insert(0, _ConfigModuleFinder())


def __getattr__(name: str) -> ModuleType:
    if name == "config":
        _install_config_alias()
        module = importlib.import_module("enn_torch.core.config")
        sys.modules.setdefault(f"{__name__}.config", module)
        globals()["config"] = module
        return module
    if name in {"core", "data", "nn", "runtime"}:
        return importlib.import_module(f"enn_torch.{name}")
    raise AttributeError(f"module 'enn_torch' has no attribute {name!r}")


def new_model(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflow

    return workflow.new_model(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflow

    return workflow.load_model(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> str:
    from .runtime import workflow

    return workflow.save_model(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Model:
    from .runtime import workflow

    return workflow.train(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> TensorDictBase | dict[str, TensorDictBase]:
    from .runtime import workflow

    return workflow.predict(*args, **kwargs)


_install_config_alias()
