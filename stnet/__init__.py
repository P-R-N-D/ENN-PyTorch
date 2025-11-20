# -*- coding: utf-8 -*-
from __future__ import annotations

from tensordict import set_list_to_stack

set_list_to_stack(True).set()

from . import api, backend, data, functional, model

__all__ = [
    "api",
    "backend",
    "data",
    "functional",
    "model",
]
