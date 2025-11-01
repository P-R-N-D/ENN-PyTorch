"""Compatibility shim for the legacy :mod:`stnet.run` package."""

from __future__ import annotations

import sys as _sys
import warnings as _warnings

from stnet import api as _api

__all__ = list(_api.__all__)

_globals = globals()
for _name in __all__:
    _globals[_name] = getattr(_api, _name)

del _globals

# Provide compatibility aliases for the submodules that previously lived here.
_sys.modules[__name__ + ".config"] = _api.config
_sys.modules[__name__ + ".io"] = _api.io
_sys.modules[__name__ + ".run"] = _api.run
_sys.modules[__name__ + ".runtime"] = _api.runtime
_sys.modules[__name__ + ".utils"] = _api.utils
_sys.modules[__name__ + ".dtypes"] = _api.datatype

_warnings.warn(
    "'stnet.run' is deprecated and will be removed in a future release; "
    "import from 'stnet.api' instead.",
    DeprecationWarning,
    stacklevel=2,
)
