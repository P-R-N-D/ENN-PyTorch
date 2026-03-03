from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from typing import Any, Dict

from .datatypes import env_bool, env_first

_DIAG_SEEN: set[str] = set()


def diag_enabled() -> bool:
    return bool(
        env_bool(
            (
                "ENN_DIAG_BATCH_SIZES",
                "ENN_DIAG_BATCHING",
                "ENN_DIAG_BATCH",
            ),
            default=False,
        )
    )


def diag_dir() -> str:
    raw = (
        env_first(
            (
                "ENN_DIAG_BATCH_DIR",
                "ENN_DIAG_BATCHING_DIR",
                "ENN_BATCH_DIAG_DIR",
                "ENN_DIAG_DIR",
            ),
            default=None,
        )
        or ""
    )
    root = str(raw).strip()
    if not root:
        root = "/var/tmp/enn_batch_diag"

    for cand in (root, "/var/tmp/enn_batch_diag", "/tmp/enn_batch_diag"):
        try:
            os.makedirs(cand, exist_ok=True)
            test = os.path.join(cand, ".write_test")
            with open(test, "a", encoding="utf-8"):
                pass
            with contextlib.suppress(Exception):
                os.remove(test)
            return cand
        except Exception:
            continue
    return root


def diag_once(tag: str) -> bool:
    tag_s = str(tag)
    if tag_s in _DIAG_SEEN:
        return False
    _DIAG_SEEN.add(tag_s)
    return True


def _json_safe(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


def diag_emit(event: str, payload: Dict[str, Any], *, also_print: bool = True) -> None:
    if not diag_enabled():
        return

    rec = {
        "ts": time.time(),
        "pid": os.getpid(),
        "event": str(event),
        "payload": {k: _json_safe(v) for k, v in (payload or {}).items()},
    }

    try:
        out_dir = diag_dir()
        path = os.path.join(out_dir, f"batching.{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    except Exception:
        pass

    if also_print:
        try:
            msg = f"[ENN][diag] {rec['event']} {rec['payload']}"
            print(msg, file=sys.stderr, flush=True)
        except Exception:
            pass

    try:
        logging.getLogger("enn.diag").warning(
            "[ENN][diag] %s %s", str(rec["event"]), str(rec["payload"])
        )
    except Exception:
        pass
