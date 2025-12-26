"""STNet top-level package.

This package intentionally exposes a small, stable surface area:

Modules:
- stnet.core
- stnet.data
- stnet.model
- stnet.run

Convenience functions (thin wrappers around stnet.run.*):
- stnet.new_model
- stnet.load_model
- stnet.save_model
- stnet.train
- stnet.predict
- stnet.get_prediction
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "core",
    "data",
    "model",
    "run",
    # Short-call API
    "new_model",
    "load_model",
    "save_model",
    "train",
    "predict",
    "get_prediction",
]


def __getattr__(name: str) -> Any:
    # Lazy module imports keep "import stnet" cheap.
    if name in {"core", "data", "model", "run"}:
        return importlib.import_module(f"stnet.{name}")
    raise AttributeError(f"module 'stnet' has no attribute '{name}'")


def new_model(*args: Any, **kwargs: Any) -> Any:
    """Create a new model.

    Wrapper for :func:`stnet.run.io.new_model`.
    """

    from .run.io import new_model as _new_model

    return _new_model(*args, **kwargs)


def load_model(*args: Any, **kwargs: Any) -> Any:
    """Load a model checkpoint.

    Wrapper for :func:`stnet.run.io.load_model`.
    """

    from .run.io import load_model as _load_model

    return _load_model(*args, **kwargs)


def save_model(*args: Any, **kwargs: Any) -> Any:
    """Save a model checkpoint.

    Wrapper for :func:`stnet.run.io.save_model`.
    """

    from .run.io import save_model as _save_model

    return _save_model(*args, **kwargs)


def train(*args: Any, **kwargs: Any) -> Any:
    """Train a model.

    Wrapper for :func:`stnet.run.compute.train`.
    """

    from .run.compute import train as _train

    return _train(*args, **kwargs)


def predict(*args: Any, **kwargs: Any) -> Any:
    """Run predict/infer.

    Wrapper for :func:`stnet.run.compute.predict`.
    """

    from .run.compute import predict as _predict

    return _predict(*args, **kwargs)


def get_prediction(*args: Any, **kwargs: Any) -> Any:
    """Read predictions saved by :func:`stnet.predict`.

    Wrapper for :func:`stnet.run.compute.get_prediction`.
    """

    from .run.compute import get_prediction as _get_prediction

    return _get_prediction(*args, **kwargs)
