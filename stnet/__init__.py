# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import logging
import os
import re
import warnings
from types import ModuleType

os.environ.setdefault("TORCH_LOGS", "-all")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import torch
try:
    from tensordict import set_list_to_stack as _set_list_to_stack
except Exception:
    _set_list_to_stack = None

__all__ = [
    "api",
    "backend",
    "data",
    "functional",
    "model",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


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

if _set_list_to_stack is not None:
    try:
        setter = _set_list_to_stack(True)
        if hasattr(setter, "set") and callable(getattr(setter, "set")):
            setter.set()
    except Exception:
        pass
