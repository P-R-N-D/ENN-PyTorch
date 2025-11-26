# -*- coding: utf-8 -*-
from __future__ import annotations

from tensordict import set_list_to_stack

set_list_to_stack(True).set()

import os as _os

_os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
_os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

from . import api, backend, data, functional, model

__all__ = [
    "api",
    "backend",
    "data",
    "functional",
    "model",
]
