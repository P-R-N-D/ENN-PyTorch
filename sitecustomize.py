from __future__ import annotations

import logging
import re

_DCP_NOISE_RE = re.compile(
    r"("
    r"Initializing dist\.ProcessGroup in checkpoint background process"
    r"|Checkpoint background process is running"
    r"|Checkpoint background process is shutting down"
    r"|Waiting for checkpoint save request"
    r"|Received async checkpoint request"
    r"|Completed checkpoint save request"
    r"|Checkpoint save failed for checkpoint_id="
    r")",
    re.IGNORECASE,
)


class _ENNDropTorchDCPNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if not msg:
            return True
        return _DCP_NOISE_RE.search(msg) is None


def _install() -> None:
    root = logging.getLogger()
    try:
        for f in getattr(root, "filters", []) or []:
            if isinstance(f, _ENNDropTorchDCPNoise):
                return
    except Exception:
        pass
    root.addFilter(_ENNDropTorchDCPNoise())


try:
    _install()
except Exception:
    pass
