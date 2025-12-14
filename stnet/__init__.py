# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import re
import warnings

os.environ.setdefault("TORCH_LOGS", "-all")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

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

    _DROP_SUBSTRINGS = (
        "No valid triton configs",
        "Runtime error during autotuning",
        "Ignoring this choice",
        "Autotune Choices Stats",
        "triton_flex_",
        "Not enough SMs",
        "hit config.recompile_limit",
        "recompilation reasons",
        "recompiles",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "best_time" in msg and "best_triton_pos" in msg:
            return False
        return not any(substr in msg for substr in self._DROP_SUBSTRINGS)


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


if _env_flag("STNET_DISABLE_MKLDNN", False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.backends.mkldnn.enabled = False
        except Exception:
            pass

warnings.filterwarnings(
    "ignore",
    message="Please use the new API settings to control TF32 behavior.*",
)
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
    "torch._dynamo.convert_frame",
):
    logger = logging.getLogger(logger_name)
    logger.addFilter(IgnoreTorchCompileMsg())
    logger.setLevel(logging.ERROR)

set_list_to_stack(True).set()
