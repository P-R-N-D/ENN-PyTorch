# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging

import torch
from tensordict import set_list_to_stack

from . import api, backend, data, functional, model

__all__ = [
    "api",
    "backend",
    "data",
    "functional",
    "model",
]


os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
set_list_to_stack(True).set()
