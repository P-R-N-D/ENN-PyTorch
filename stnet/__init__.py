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

class IgnoreTorchCompileMsg(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "No valid triton configs" in msg:
            return False
        if "Runtime error during autotuning" in msg:
            return False
        if "Ignoring this choice" in msg:
            return False
        return True

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").addFilter(IgnoreTorchCompileMsg())
logging.getLogger("torch._inductor.select_algorithm").addFilter(IgnoreTorchCompileMsg())
set_list_to_stack(True).set()
