# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging
import re
import warnings

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
        if "No valid triton configs" in msg: return False
        elif "Runtime error during autotuning" in msg: return False
        elif "Ignoring this choice" in msg: return False
        elif "Autotune Choices Stats" in msg: return False
        elif "triton_flex_" in msg: return False
        elif "best_time" in msg and "best_triton_pos" in msg: return False
        elif "Not enough SMs" in msg: return False
        elif "hit config.recompile_limit" in msg: return False
        elif "recompilation reasons" in msg: return False
        elif "recompiles" in msg: return False
        else: return True

os.environ.setdefault("TORCH_LOGS", "-all")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32 behavior.*")
ignored_sentences = [
    "External init callback must run in same thread as registerClient",
    "Initializing zero-element tensors is a no-op",
    "gpuGetDeviceCount failed with code",
    "torch.distributed is disabled",
    "TypedStorage is deprecated",
    "flex_attention called without torch.compile",
    "SOLUTION: Use torch.compile",
    "Not enough SMs to use max_autotune_gemm",
    "allowTF32CuDNN",
    "allowTF32CuBLAS",
    "torch._dynamo hit config.recompile_limit",
    "Detected a Jax installation",
]
ignored_pattern = "|".join([f".*{re.escape(s)}.*" for s in ignored_sentences])
warnings.filterwarnings("ignore", message=ignored_pattern)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
for logger_name in (
    "torch._inductor", 
    "torch._inductor.select_algorithm", 
    "torch._dynamo",
    "torch._dynamo.convert_frame"
):
    logger = logging.getLogger(logger_name)
    logger.addFilter(IgnoreTorchCompileMsg())
    logger.setLevel(logging.ERROR)
set_list_to_stack(True).set()
