# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import torch
from tensordict import TensorDict

from enn_torch.config import ModelConfig, PatchConfig
from enn_torch.core.tensor import extract_tensor, from_buffer
from enn_torch.runtime.io import Exporter
from enn_torch.runtime.workflow import new_model, train

from .lifecycle import build_dataset

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TL_LOG_RE = re.compile(r"(dedicated_log_torch_trace_[A-Za-z0-9_]+\.log)")


def _as_path_list(out: Any, fallback: Path) -> list[str]:
    if out is None:
        return [str(fallback)]
    if isinstance(out, (str, Path)):
        return [str(out)]
    if isinstance(out, (tuple, list)):
        flat: list[str] = []
        for item in out:
            if item is None:
                continue
            if isinstance(item, (str, Path)):
                flat.append(str(item))
            else:
                flat.append(str(item))
        return flat if flat else [str(fallback)]
    return [str(out)]


def _stats_np(arr: np.ndarray) -> dict[str, float | list[int]]:
    arr = np.asarray(arr)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _build_model_and_sample(
    device: torch.device,
) -> tuple[dict[str, Any], TensorDict, torch.nn.Module, torch.Tensor]:
    data = build_dataset("raw_data.xlsx")
    td_train = data["td_train"]
    S = data["S"]
    T = data["T"]
    patch = PatchConfig(
        is_cube=True,
        grid_size_3d=(S, T, 1),
        patch_size_3d=(1, 1, 1),
        use_padding=True,
    )
    cfg = ModelConfig(
        device=device,
        patch=patch,
        normalization_method="layernorm",
        d_model=192,
        heads=2,
        mlp_ratio=2.0,
        dropout=0.05,
        drop_path=0.05,
        spatial_depth=2,
        temporal_depth=2,
        spatial_latents=24,
        temporal_latents=24,
        modeling_type="spatiotemporal",
        compile_mode="disabled",
    )
    model = new_model(in_dim=td_train["X"].shape[1], out_shape=(S, T), config=cfg).to(
        device
    )
    if os.environ.get("ENN_DEPLOYMENT_DEBUG_EXTRA", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    ):
        with contextlib.suppress(Exception):
            model.add_task("debug_extra", mode="spatial", weight=0.25)
    print("[debug] tasks:", model.list_tasks())
    sample = td_train["X"][:4].to(device)
    return data, td_train, model, sample


def _run_isolated_export(
    fmt_name: str, out_path: str, state_path: str
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "debug.deployment",
        "--export-only",
        fmt_name,
        "--out",
        out_path,
        "--state",
        state_path,
    ]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    env.setdefault("ENN_EXPORT_FAULTHANDLER", "1")
    env.setdefault("ENN_EXPORT_TRACEBACK_AFTER_SEC", "30")
    timeout_s = 120
    if str(fmt_name).strip().lower() in ("onnx", "ort"):
        timeout_s = 600
        env.setdefault("ENN_ONNX_TRY_DYNAMO", "0")
    with contextlib.suppress(Exception):
        timeout_s = int(
            os.environ.get("ENN_EXPORT_SUBPROCESS_TIMEOUT", str(timeout_s)).strip()
            or str(timeout_s)
        )

    def _tail_text(v: object, n: int = 2000) -> str:
        if v is None:
            return ""
        if isinstance(v, bytes):
            try:
                s = v.decode("utf-8", errors="replace")
            except Exception:
                s = repr(v)
            return s[-n:]
        return str(v)[-n:]

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            os.killpg(p.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            p.kill()
        out2, err2 = ("", "")
        with contextlib.suppress(Exception):
            out2, err2 = p.communicate(timeout=5)
        return {
            "status": "error",
            "error": "subprocess export timed out",
            "stdout_tail": _tail_text(out2 or out),
            "stderr_tail": _tail_text(err2 or err),
        }
    if p.returncode == 0:

        def _parse_last_json_line(text: object) -> dict[str, Any] | None:
            if not text:
                return None
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
            for ln in reversed(lines):
                if ln.startswith("{") and ln.endswith("}"):
                    with contextlib.suppress(Exception):
                        return json.loads(ln)
            return None

        obj = _parse_last_json_line(out)
        if obj is not None:
            obj.setdefault("_stdout_tail", _tail_text(out))
            obj.setdefault("_stderr_tail", _tail_text(err))
            return obj
        return {
            "status": "error",
            "error": "subprocess returned non-json output",
            "stdout": _tail_text(out),
            "stderr": _tail_text(err),
        }
    return {
        "status": "error",
        "error": f"subprocess export failed (returncode={p.returncode})",
        "stdout_tail": _tail_text(out),
        "stderr_tail": _tail_text(err),
    }


def _ensure_state_shapes_for_scaler(
    model: torch.nn.Module, sd: dict[str, object]
) -> None:
    if not isinstance(sd, dict):
        return

    def _model_device(mod: torch.nn.Module) -> torch.device:
        with contextlib.suppress(StopIteration):
            return next(mod.parameters()).device
        with contextlib.suppress(StopIteration):
            return next(mod.buffers()).device
        return torch.device("cpu")

    def _alloc_replacement(
        existing: torch.Tensor | None,
        shape: torch.Size,
        *,
        fallback_mod: torch.nn.Module,
    ) -> torch.Tensor:
        if torch.is_tensor(existing):
            return torch.empty(shape, device=existing.device, dtype=existing.dtype)
        return torch.empty(
            shape, device=_model_device(fallback_mod), dtype=torch.float32
        )

    def _is_scaler_key(k: str) -> bool:
        return k.startswith("scaler.") or ".scaler." in k

    for full_key, val in sd.items():
        if not isinstance(full_key, str) or not _is_scaler_key(full_key):
            continue
        if not torch.is_tensor(val):
            continue
        parts = full_key.split(".")
        mod = model
        ok = True
        for p in parts[:-1]:
            if not hasattr(mod, p):
                ok = False
                break
            mod = getattr(mod, p)
            if not isinstance(mod, torch.nn.Module):
                ok = False
                break
        if not ok:
            continue
        name = parts[-1]
        tgt = val.detach()
        if hasattr(mod, "_buffers") and name in getattr(mod, "_buffers", {}):
            buf = mod._buffers.get(name)
            if torch.is_tensor(buf) and tuple(buf.shape) != tuple(tgt.shape):
                mod._buffers[name] = _alloc_replacement(
                    buf,
                    tgt.shape,
                    fallback_mod=mod,
                )
                setattr(mod, name, mod._buffers[name])
            continue
        if hasattr(mod, "_parameters") and name in getattr(mod, "_parameters", {}):
            prm = mod._parameters.get(name)
            if (
                prm is not None
                and torch.is_tensor(prm)
                and tuple(prm.shape) != tuple(tgt.shape)
            ):
                mod._parameters[name] = torch.nn.Parameter(
                    _alloc_replacement(
                        prm,
                        tgt.shape,
                        fallback_mod=mod,
                    ),
                    requires_grad=prm.requires_grad,
                )
                setattr(mod, name, mod._parameters[name])
            continue


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _truncate(s: str, n: int = 4000) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _should_run_draft(err: str) -> bool:
    e = _strip_ansi(err or "")
    return ("Constraints violated" in e) or ("torch.export.export" in e)


def _extract_tlparse_log_from_text(txt: str) -> str | None:
    if not txt:
        return None
    m = _TL_LOG_RE.search(txt)
    if not m:
        return None
    return f"/tmp/export_root/{m.group(1)}"


def _read_log_excerpt(
    path: str, *, max_lines: int = 220, tail: bool = True
) -> str | None:
    try:
        p = Path(path)
        if not p.exists():
            return None
        lines = p.read_text(errors="replace").splitlines()
        if not lines:
            return None
        chunk = lines[-max_lines:] if tail else lines[:max_lines]
        return "\n".join(chunk)
    except Exception:
        return None


def _report_to_text(ep: object) -> str:
    for attr in ("_report", "report"):
        if hasattr(ep, attr):
            try:
                return str(getattr(ep, attr))
            except Exception:
                pass
    return ""


def _draft_export_diagnostics(
    model: torch.nn.Module, sample: torch.Tensor
) -> dict[str, Any]:
    info: dict[str, Any] = {"ok": False, "attempts": []}
    info["ts"] = int(time.time())
    try:
        import torch.export as tex
    except Exception as exc:
        info["error"] = f"torch.export import failed: {exc!r}"
        return info
    try:
        from enn_torch.runtime.wrappers import _onnx_model, _TensorOutputModule
    except Exception as exc:
        info["error"] = f"could not import ONNX wrapper helpers: {exc!r}"
        return info
    Dim = getattr(tex, "Dim", None)
    dyn_candidates: list[Any] = [None]
    if Dim is not None:
        try:
            bdim = Dim("batch", min=1)
        except Exception:
            try:
                bdim = Dim("batch")
            except Exception:
                bdim = None
        if bdim is not None:
            dyn_candidates = [
                ({"x": {0: bdim}},),
                {"x": {0: bdim}},
                ({0: bdim},),
            ] + dyn_candidates
    try:
        with _onnx_model(model) as serving_model:
            wrapper = _TensorOutputModule(serving_model).eval()
            args = (sample,)
            for strict in (False, True):
                for dyn in dyn_candidates:
                    try:
                        kw = {"strict": strict}
                        if dyn is not None:
                            kw["dynamic_shapes"] = dyn
                        with warnings.catch_warnings(record=True) as wrec:
                            warnings.simplefilter("always")
                            res = tex.draft_export(wrapper, args, **kw)
                        info["ok"] = True
                        info["strict"] = strict
                        info["dynamic_shapes_repr"] = repr(dyn)
                        rep_txt = _report_to_text(res)
                        if rep_txt:
                            info["report"] = _truncate(rep_txt, 12000)
                            tl = _extract_tlparse_log_from_text(rep_txt)
                            if tl:
                                info["tlparse_log"] = tl
                                ex = _read_log_excerpt(tl)
                                if ex:
                                    info["tlparse_log_excerpt_tail"] = _truncate(
                                        ex, 12000
                                    )
                        if wrec:
                            info["warnings"] = [
                                _truncate(_strip_ansi(str(w.message)), 2000)
                                for w in wrec
                            ]
                            if "tlparse_log" not in info:
                                for w in info["warnings"]:
                                    tl = _extract_tlparse_log_from_text(w)
                                    if tl:
                                        info["tlparse_log"] = tl
                                        ex = _read_log_excerpt(tl)
                                        if ex:
                                            info["tlparse_log_excerpt_tail"] = (
                                                _truncate(ex, 12000)
                                            )
                                        break
                        for attr in (
                            "errors",
                            "constraints",
                            "guards",
                            "help",
                        ):
                            if hasattr(res, attr):
                                try:
                                    info[attr] = _truncate(repr(getattr(res, attr)))
                                except Exception:
                                    pass
                        return info
                    except Exception as e:
                        info["attempts"].append(
                            {
                                "strict": strict,
                                "dynamic_shapes_repr": repr(dyn),
                                "error": _truncate(repr(e)),
                            }
                        )
    except Exception as exc:
        info["error"] = _truncate(repr(exc))
        info["traceback"] = _truncate(traceback.format_exc(), 6000)
    return info


@contextlib.contextmanager
def _temp_env(k: str, v: str) -> Iterator[None]:
    old = os.environ.get(k)
    os.environ[k] = v
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def _export_only_main(fmt_name: str, out_path: str, state_path: str) -> int:
    with contextlib.suppress(Exception):
        if os.environ.get("ENN_EXPORT_FAULTHANDLER", "1").strip() != "0":
            import faulthandler

            faulthandler.enable(all_threads=True)
            after_s = 30
            with contextlib.suppress(Exception):
                after_s = int(
                    os.environ.get("ENN_EXPORT_TRACEBACK_AFTER_SEC", "30").strip()
                    or "30"
                )
            faulthandler.dump_traceback_later(after_s, repeat=True)
    device = torch.device("cpu")
    data, td_train, model, sample = _build_model_and_sample(device)
    sd = torch.load(state_path, map_location="cpu")
    with contextlib.suppress(Exception):
        _ensure_state_shapes_for_scaler(model, sd)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        msg = str(exc)
        if "Unexpected key(s) in state_dict" in msg:
            model.load_state_dict(sd, strict=False)
        else:
            raise
    model.eval()
    fmt = Exporter.for_export(
        Path(out_path).suffix if Path(out_path).suffix else out_path
    )
    if fmt is None:
        print(json.dumps({"status": "error", "error": "no exporter registered"}))
        return 1
    try:
        print(f"[export-only] format={fmt_name} out={out_path}", flush=True)
        save_kw = {"sample_input": sample, "dynamic_batch": True}
        if str(fmt_name).strip().lower() in {
            "onnx",
            "ort",
            "tensorrt",
            "tensorflow",
            "litert",
        }:
            save_kw["prefer_dynamo"] = bool(
                os.environ.get("ENN_ONNX_PREFER_DYNAMO", "0").strip().lower()
                in ("1", "true", "yes", "y", "on")
            )
        t0 = time.time()
        fmt.save(model, out_path, **save_kw)
        dt = time.time() - t0
        print(f"[export-only] done format={fmt_name} dt={dt:.2f}s", flush=True)
        print(json.dumps({"status": "ok"}))
        return 0
    except ImportError as exc:
        print(json.dumps({"status": "skipped", "error": repr(exc)}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": repr(exc)}))
        return 1


def export_and_validate(
    model: torch.nn.Module,
    sample: torch.Tensor,
    td_train: TensorDict,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*LeafSpec.*",
    )
    targets = {
        "pt2": out_dir / "model.pt2",
        "onnx": out_dir / "model.onnx",
        "ort": out_dir / "model.ort",
        "executorch": out_dir / "model.pte",
        "tensorflow": out_dir / "model.savedmodel",
        "litert": out_dir / "model.tflite",
        "coreml": out_dir / "model.mlmodel",
        "tensorrt": out_dir / "model.engine",
    }
    results: Dict[str, Any] = {}
    state_path = out_dir / "model.state_dict.pt"
    task_spec_path = out_dir / "task_specs.json"
    try:
        torch.save(model.state_dict(), state_path)
        with contextlib.suppress(Exception):
            task_specs = getattr(model, "task_specs", None)
            if callable(task_specs):
                task_spec_path.write_text(
                    json.dumps(task_specs(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    except Exception as exc:
        state_path = None
        results["_state_save_error"] = repr(exc)
    with _temp_env("ENN_DISABLE_PIECEWISE_CALIB", "1"):
        isolate = {
            s.strip().lower()
            for s in os.environ.get(
                "ENN_DEPLOYMENT_ISOLATE_EXPORTS", "pt2,onnx,ort"
            ).split(",")
            if s.strip()
        }
        for name, path in targets.items():
            if state_path is not None and name.strip().lower() in isolate:
                results[name] = _run_isolated_export(name, str(path), str(state_path))
                if results[name].get("status") == "ok":
                    results[name] = {"status": "ok", "paths": [str(path)]}
                elif results[name].get("status") == "skipped":
                    results[name] = {
                        "status": "skipped",
                        "reason": "missing_optional_dependency",
                        "error": results[name].get("error", "skipped"),
                    }
                continue
            fmt = Exporter.for_export(path.suffix if path.suffix else str(path))
            try:
                if fmt is None:
                    raise RuntimeError("no exporter registered")
                save_kwargs: Dict[str, Any] = {
                    "sample_input": sample,
                    "dynamic_batch": True,
                }
                out = fmt.save(model, path, **save_kwargs)
                results[name] = {
                    "status": "ok",
                    "paths": _as_path_list(out, path),
                }
            except ImportError as exc:
                if name in ("pt2", "onnx"):
                    results[name] = {"status": "error", "error": repr(exc)}
                else:
                    results[name] = {
                        "status": "skipped",
                        "reason": "missing_optional_dependency",
                        "error": repr(exc),
                    }
            except Exception as exc:
                err_s = repr(exc)
                results[name] = {"status": "error", "error": err_s}
    validation: Dict[str, Any] = {}
    try:
        y = td_train["Y"].detach().cpu().numpy()
        validation["label_stats"] = _stats_np(y)
    except Exception as exc:
        validation["label_stats_error"] = repr(exc)
    pt2_path = targets["pt2"]
    if pt2_path.exists():
        with _temp_env("ENN_DISABLE_PIECEWISE_CALIB", "1"):
            try:
                with from_buffer():
                    ep = torch.export.load(str(pt2_path))
                with torch.no_grad():
                    pt2_out = extract_tensor(ep.module()(sample))
                    torch_out = extract_tensor(
                        model.forward_export(sample)
                        if hasattr(model, "forward_export")
                        else model(sample, return_loss=False)
                    )

                def _to_numpy_materialized(t: torch.Tensor) -> np.ndarray:
                    t = extract_tensor(t).detach()
                    try:
                        from torch._subclasses.functional_tensor import (
                            disable_functional_mode,
                        )
                    except Exception:
                        disable_functional_mode = None
                    cm = (
                        disable_functional_mode()
                        if disable_functional_mode is not None
                        else contextlib.nullcontext()
                    )
                    with cm:
                        tt = t
                        if hasattr(tt, "to_local"):
                            with contextlib.suppress(Exception):
                                tt = tt.to_local()
                        tt = tt.to("cpu").contiguous()
                        base = torch.empty(
                            tuple(tt.shape), dtype=tt.dtype, device="cpu"
                        )
                        base.copy_(tt)
                        return base.numpy()

                try:
                    pt2_np = _to_numpy_materialized(pt2_out)
                    torch_np = _to_numpy_materialized(torch_out)
                    validation["pt2_mae"] = float(np.mean(np.abs(pt2_np - torch_np)))
                    validation["pt2_out_stats"] = _stats_np(pt2_np)
                except Exception as conv_exc:
                    validation["pt2_error"] = repr(conv_exc)
                    validation["pt2_out_repr"] = repr(pt2_out)
            except Exception as exc:
                validation["pt2_error"] = repr(exc)

        def _truthy(v: str) -> bool:
            return v.strip().lower() in ("1", "true", "yes", "y", "on")

        do_alt = _truthy(os.environ.get("ENN_VALIDATE_ALT_BATCH", "0"))
        if (
            "pt2_error" not in validation
            and do_alt
            and isinstance(sample, torch.Tensor)
            and sample.ndim >= 2
            and int(sample.shape[0]) > 1
        ):
            try:
                alt_n = max(1, int(sample.shape[0]) // 2)
                alt = sample[:alt_n]
                with torch.no_grad():
                    alt_pt2 = extract_tensor(ep.module()(alt))
                    alt_torch = extract_tensor(
                        model.forward_export(alt)
                        if hasattr(model, "forward_export")
                        else model(alt, return_loss=False)
                    )
                validation["pt2_mae_alt"] = float(
                    np.mean(
                        np.abs(
                            alt_pt2.detach().cpu().numpy()
                            - alt_torch.detach().cpu().numpy()
                        )
                    )
                )
            except Exception as exc:
                validation["pt2_mae_alt_error"] = repr(exc)
    onnx_res = results.get("onnx", {})
    if isinstance(onnx_res, dict) and onnx_res.get("status") == "error":
        err = str(onnx_res.get("error", ""))
        if _should_run_draft(err):
            if os.environ.get("ENN_ONNX_DRAFT_EXPORT", "1") != "0":
                validation["onnx_draft_export"] = _draft_export_diagnostics(
                    model, sample
                )
    for name in ("onnx", "ort"):
        path = targets[name]
        if path.exists():
            with _temp_env("ENN_DISABLE_PIECEWISE_CALIB", "1"):
                try:
                    import onnxruntime as ort

                    sess = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"]
                    )
                    inp_name = sess.get_inputs()[0].name
                    out = sess.run(
                        None,
                        {inp_name: sample.detach().cpu().numpy().astype(np.float32)},
                    )[0]
                    validation[f"{name}_out_stats"] = _stats_np(out)
                except Exception as exc:
                    validation[f"{name}_error"] = repr(exc)
    return {"exports": results, "validation": validation}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-only", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--state", default=None)
    args, _ = ap.parse_known_args()
    if args.export_only:
        raise SystemExit(_export_only_main(args.export_only, args.out, args.state))
    os.environ.setdefault("ENN_PREBATCH", "1")
    os.environ.setdefault("ENN_PREFETCH_FACTOR", "1")
    data, td_train, model, sample = _build_model_and_sample(torch.device("cpu"))
    S = data["S"]
    T = data["T"]
    print(f"[export] dataset groups={data['B']} grid={S}x{T}")
    print("[export] training short run (epochs=2) for exportable weights")
    train(
        model,
        td_train,
        epochs=2,
        base_lr=3e-3,
        weight_decay=1e-4,
        val_frac=0.1,
        max_nodes=1,
    )
    model.eval()
    stats = export_and_validate(model, sample, td_train, Path("export_artifacts"))

    def _json_default(o: object) -> object:
        if isinstance(o, bytes):
            try:
                return o.decode("utf-8", errors="replace")
            except Exception:
                return repr(o)
        if isinstance(o, Path):
            return str(o)
        try:
            if isinstance(o, np.generic):
                return o.item()
        except Exception:
            pass
        return repr(o)

    print(json.dumps(stats, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
