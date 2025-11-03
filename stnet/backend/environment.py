# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
import importlib
import math
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.multiprocessing as mp
from tensordict import TensorDict
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
)
from torchrl.envs import EnvBase

from ..data.nodes import SampleReader

@dataclass
class _RuntimeConfig:
    deterministic: bool = False
    allow_tf32: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    matmul_precision: Optional[str] = None
    sdpa_backends: Optional[List[str]] = None
    te_first: bool = True


_RUNTIME_CONFIG = _RuntimeConfig()


class System:
    RuntimeConfig = _RuntimeConfig

    @staticmethod
    def get_runtime_config() -> _RuntimeConfig:
        return _RUNTIME_CONFIG

    @staticmethod
    def is_main_loadable() -> bool:
        main_mod = sys.modules.get("__main__")
        if main_mod is None:
            return False
        main_path = getattr(main_mod, "__file__", None)
        if not main_path:
            return False
        try:
            main_path = os.fspath(main_path)
        except TypeError:
            return False
        if isinstance(main_path, str) and main_path.startswith("<") and main_path.endswith(">"):
            return False
        return os.path.exists(main_path)

    @staticmethod
    def initialize_python_path() -> str:
        separator = os.pathsep
        current_env = os.environ.get("PYTHONPATH", "")
        env_paths = [path for path in current_env.split(separator) if path]
        paths: List[str] = list(env_paths)
        seen: set[str] = set(env_paths)

        def _ensure_front(candidate: Path | str | None) -> None:
            if candidate is None:
                return
            try:
                path_str = os.fspath(candidate)
            except TypeError:
                return
            if not path_str:
                return
            if path_str in seen:
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                return
            seen.add(path_str)
            paths.insert(0, path_str)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        def _ensure_env(entry: str) -> None:
            if entry in seen:
                return
            seen.add(entry)
            paths.append(entry)

        try:
            package_dir = Path(__file__).resolve().parents[1]
        except Exception:
            package_dir = None
        project_dir = package_dir.parent if package_dir is not None else None
        cwd_dir: Path | None = None
        with contextlib.suppress(Exception):
            cwd_dir = Path.cwd().resolve()
        main_dir: Path | None = None
        main_module = sys.modules.get("__main__")
        if main_module is not None:
            main_file = getattr(main_module, "__file__", None)
            if main_file:
                with contextlib.suppress(Exception):
                    main_dir = Path(main_file).resolve().parent

        for candidate in (package_dir, project_dir, main_dir, cwd_dir):
            _ensure_front(candidate)

        for entry in list(sys.path):
            if not entry:
                continue
            try:
                entry_str = os.fspath(entry)
            except TypeError:
                continue
            if not entry_str:
                continue
            _ensure_env(entry_str)

        python_path = separator.join(paths)
        os.environ["PYTHONPATH"] = python_path
        return python_path

    @staticmethod
    def optimal_start_method() -> str:
        current = mp.get_start_method(allow_none=True)
        if current is not None:
            return str(current)
        for method in ("forkserver", "spawn"):
            try:
                multiprocessing.get_context(method)
            except ValueError:
                continue
            return method
        raise RuntimeError(
            "No supported multiprocessing start method "
            "(tried forkserver, spawn)."
        )

    @staticmethod
    def set_multiprocessing_env() -> None:
        try:
            mp.set_sharing_strategy("file_system")
        except RuntimeError:
            pass
        if mp.get_start_method(allow_none=True) is not None:
            return
        last_error: Optional[BaseException] = None
        for method in ("forkserver", "spawn"):
            try:
                multiprocessing.get_context(method)
            except ValueError as exc:
                last_error = exc
                continue
            try:
                for module in (multiprocessing, mp):
                    module.set_start_method(method, force=True)
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                continue
            return
        raise RuntimeError(
            "Unable to configure multiprocessing start method "
            "(tried forkserver, spawn)."
        ) from last_error

    @staticmethod
    def default_temp() -> str:
        return (
            os.environ.get("TEMP", "C:\\Windows\\Temp")
            if sys.platform.startswith("win")
            else "/tmp"
            if os.path.isdir("/tmp")
            else "/var/tmp"
        )

    @staticmethod
    def new_dir(prefix: str) -> str:
        base = System.default_temp()
        os.makedirs(base, exist_ok=True)
        directory = os.path.join(base, f"{prefix}_{os.getpid()}_{os.urandom(4).hex()}")
        os.makedirs(directory, exist_ok=True)
        return directory

    @staticmethod
    def initialize_sdpa_backends() -> List[object]:
        names = _RUNTIME_CONFIG.sdpa_backends or []
        if not names:
            return []
        try:
            from torch.nn.attention import SDPBackend
        except Exception:
            return []
        mapping = {
            "FLASH": "FLASH_ATTENTION",
            "FLASH_ATTENTION": "FLASH_ATTENTION",
            "EFFICIENT": "EFFICIENT_ATTENTION",
            "MEM_EFFICIENT": "EFFICIENT_ATTENTION",
            "CUDNN": "CUDNN_ATTENTION",
            "MATH": "MATH",
        }
        backends: List[object] = []
        for name in names:
            key = mapping.get(name, name)
            if hasattr(SDPBackend, key):
                backends.append(getattr(SDPBackend, key))
        return backends

    @staticmethod
    def is_cpu_bf16_supported() -> bool:
        try:
            mkldnn_ops = getattr(torch.ops, "mkldnn", None)
            if mkldnn_ops is not None and hasattr(
                mkldnn_ops, "_is_mkldnn_bf16_supported"
            ):
                return bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())
        except Exception:
            pass
        return False

    @staticmethod
    def is_cuda_bf16_supported() -> bool:
        try:
            if not torch.cuda.is_available():
                return False
            f = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(f):
                return bool(f())
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            return major >= 8
        except Exception:
            return False

    @staticmethod
    def get_device(
        *args: Any,
        deterministic: Optional[bool] = None,
        allow_tf32: Optional[bool] = None,
        cudnn_benchmark: Optional[bool] = None,
        matmul_precision: Optional[str] = None,
        sdpa_backends: Optional[Sequence[str]] = None,
        te_first: Optional[bool] = None,
        **kwargs: Any,
    ) -> torch.device:
        cfg = _RUNTIME_CONFIG
        if deterministic is not None:
            cfg.deterministic = bool(deterministic)
        det_flag = cfg.deterministic
        allow_val = (
            bool(allow_tf32)
            if allow_tf32 is not None
            else bool(cfg.allow_tf32)
            if cfg.allow_tf32 is not None and deterministic is None
            else False
            if det_flag
            else True
        )
        cfg.allow_tf32 = allow_val
        benchmark_val = (
            bool(cudnn_benchmark)
            if cudnn_benchmark is not None
            else bool(cfg.cudnn_benchmark)
            if cfg.cudnn_benchmark is not None and deterministic is None
            else False
            if det_flag
            else True
        )
        cfg.cudnn_benchmark = benchmark_val
        precision_val = (
            str(matmul_precision)
            if matmul_precision is not None
            else str(cfg.matmul_precision)
            if cfg.matmul_precision is not None and deterministic is None
            else "highest"
            if det_flag
            else "high"
        )
        cfg.matmul_precision = precision_val
        if sdpa_backends is not None:
            cfg.sdpa_backends = [str(x) for x in sdpa_backends]
        if te_first is not None:
            cfg.te_first = bool(te_first)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.deterministic = cfg.deterministic
            torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
            try:
                torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision(str(cfg.matmul_precision))
            except Exception:
                pass
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available():
            device = torch.device("vulkan")
        else:
            device = torch.device("cpu")
        return device

    @staticmethod
    def optimal_optimizer_params(
        device: torch.device, use_foreach: Optional[bool], use_fused: bool
    ) -> Dict[str, bool]:
        devt = device.type
        flags: Dict[str, bool] = {}
        flags["foreach"] = (
            devt in {"cuda", "xpu"} if use_foreach is None else bool(use_foreach)
        )
        if use_fused and devt in {"cuda", "xpu"}:
            flags["fused"] = True
            flags["foreach"] = False
        return flags

    @staticmethod
    def cuda_compute_capability(device: torch.device) -> Tuple[int, int]:
        if device.type != "cuda" or not torch.cuda.is_available():
            return (0, 0)
        try:
            major, minor = torch.cuda.get_device_capability(device)
        except Exception:
            return (0, 0)
        return (int(major), int(minor))

    @staticmethod
    def is_float8_supported(
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[bool, str]:
        dev = torch.device(device) if device is not None else System.get_device()
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (False, f"FP8 requires CUDA (found {dev.type})")
        major, minor = System.cuda_compute_capability(dev)
        if major <= 0:
            return (False, "Unable to query CUDA compute capability")
        if major < 9:
            return (False, f"FP8 requires sm_90+ (found sm_{major}{minor})")
        try:
            import transformer_engine.pytorch as te

            backend = getattr(te, "__name__", "transformer_engine.pytorch")
            return (True, backend)
        except Exception:
            return (False, "transformer_engine not found")

    @staticmethod
    def is_int8_supported(
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[bool, str]:
        dev = torch.device(device) if device is not None else System.get_device()
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (False, f"INT8 requires CUDA (found {dev.type})")
        major, minor = System.cuda_compute_capability(dev)
        if major <= 0:
            return (False, "Unable to query CUDA compute capability")
        if major < 7:
            return (False, f"INT8 requires sm_70+ (found sm_{major}{minor})")
        try:
            importlib.import_module("torchao.quantization")
            return (True, "torchao.quantization")
        except Exception:
            return (True, f"sm_{major}{minor}")

    @staticmethod
    def is_int4_supported(
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[bool, str]:
        dev = torch.device(device) if device is not None else System.get_device()
        if dev.type != "cuda" or not torch.cuda.is_available():
            return (False, f"INT4 requires CUDA (found {dev.type})")
        major, minor = System.cuda_compute_capability(dev)
        if major <= 0:
            return (False, "Unable to query CUDA compute capability")
        if major < 8:
            return (False, f"INT4 requires sm_80+ (found sm_{major}{minor})")
        try:
            importlib.import_module("torchao.optim")
            return (True, "torchao.optim")
        except Exception:
            with contextlib.suppress(Exception):
                importlib.import_module("torchao.prototype.low_bit_optim")
                return (True, f"sm_{major}{minor}")
        return (False, "torchao low-bit optimizers unavailable")

    @staticmethod
    def _safe_cuda_device_count() -> int:
        try:
            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except Exception:
            return 0
        return 0

    @staticmethod
    def optimal_procs() -> Dict[str, Union[int, str]]:
        n_gpu = System._safe_cuda_device_count()
        return {"nproc_per_node": n_gpu or 1, "device": "cuda" if n_gpu else "cpu"}

    @staticmethod
    def cpu_count() -> int:
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            return os.cpu_count() or 1

    @staticmethod
    def optimal_threads() -> Dict[str, Union[int, bool]]:
        n_cpu = System.cpu_count()
        n_gpu = System._safe_cuda_device_count()
        intra = max(1, min(n_cpu, int(round(0.8 * n_cpu))))
        inter = max(1, min(4, int(math.sqrt(intra))))
        workers = (
            max(2, min(8 * n_gpu, n_cpu // max(1, n_gpu)))
            if n_gpu > 0
            else max(2, min(8, n_cpu // 2))
        )
        return {
            "intraop": intra,
            "interop": inter,
            "dataloader_workers": workers,
            "prefetch_factor": 2,
            "pin_memory": bool(n_gpu > 0),
        }

    @staticmethod
    def optimize_threads() -> Dict[str, Union[int, bool]]:
        threads = System.optimal_threads()
        os.environ.setdefault("OMP_NUM_THREADS", str(threads["intraop"]))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads["intraop"]))
        try:
            torch.set_num_threads(int(threads["intraop"]))
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(int(threads["interop"]))
        except Exception:
            pass
        return threads


def get_runtime_config() -> _RuntimeConfig:
    return System.get_runtime_config()


def is_main_loadable() -> bool:
    return System.is_main_loadable()


def initialize_python_path() -> str:
    return System.initialize_python_path()


def optimal_start_method() -> str:
    return System.optimal_start_method()


def set_multiprocessing_env() -> None:
    System.set_multiprocessing_env()


def default_temp() -> str:
    return System.default_temp()


def new_dir(prefix: str) -> str:
    return System.new_dir(prefix)


def initialize_sdpa_backends() -> List[object]:
    return System.initialize_sdpa_backends()


def is_cpu_bf16_supported() -> bool:
    return System.is_cpu_bf16_supported()


def is_cuda_bf16_supported() -> bool:
    return System.is_cuda_bf16_supported()


def get_device(
    *args: Any,
    **kwargs: Any,
) -> torch.device:
    return System.get_device(*args, **kwargs)


def optimal_optimizer_params(
    device: torch.device, use_foreach: Optional[bool], use_fused: bool
) -> Dict[str, bool]:
    return System.optimal_optimizer_params(device, use_foreach, use_fused)


def cuda_compute_capability(device: torch.device) -> Tuple[int, int]:
    return System.cuda_compute_capability(device)


def is_float8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    return System.is_float8_supported(device)


def is_int8_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    return System.is_int8_supported(device)


def is_int4_supported(
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[bool, str]:
    return System.is_int4_supported(device)


def optimal_procs() -> Dict[str, Union[int, str]]:
    return System.optimal_procs()


def cpu_count() -> int:
    return System.cpu_count()


def optimal_threads() -> Dict[str, Union[int, bool]]:
    return System.optimal_threads()


def optimize_threads() -> Dict[str, Union[int, bool]]:
    return System.optimize_threads()


# --- NEW: TorchRL EnvBase 구현체 --------------------------------------------


class TorchEnvironment(EnvBase):
    """
    STNet의 memmap 데이터셋을 한-스텝 에피소드(batch 단위)로 내보내는 EnvBase.

    - reset(): 다음 배치의 {observation=X, label=Y}를 준비
    - step(): action은 무시(밴딧형). 매 스텝 done=True가 되어 1-step 에피소드
    """

    batch_locked = False  # TorchRL 벡터화 규칙: 필요 시 배치 재설정 허용

    def __init__(
        self,
        memmap_dir: str,
        *,
        in_dim: int,
        label_shape: tuple[int, ...],
        split: str = "train",
        batch_size: int = 64,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device, batch_size=torch.Size([int(batch_size)]))
        self._memmap_dir = memmap_dir
        self._split = split
        self._in_dim = int(in_dim)
        self._label_shape = tuple(label_shape)
        self._batch_size = int(batch_size)

        # 데이터 리더(각 프로세스에서 독립적으로 열림)
        self._reader = SampleReader(
            memmap_dir,
            split=split,
            batch_size=self._batch_size,
        )
        self._it = iter(self._reader)

        # ---- Env specs
        # 관측치: (in_dim,)
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(self._in_dim,), device=self.device
            ),
            shape=torch.Size([]),
            device=self.device,
        )
        # (사용하지 않지만 규약상 필요) action: 스칼라 dummy
        self.action_spec = UnboundedContinuousTensorSpec(
            shape=(1,), device=self.device
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), device=self.device
        )
        self.done_spec = DiscreteTensorSpec(
            n=2, shape=(1,), dtype=torch.bool, device=self.device
        )
        self.add_truncated_keys(("truncated",))

    # 내부: 다음 배치 로드
    def _next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            item = next(self._it)
        except StopIteration:
            # 끝나면 새로운 이터레이터로 재시작
            self._it = iter(
                SampleReader(
                    self._memmap_dir, split=self._split, batch_size=self._batch_size
                )
            )
            item = next(self._it)

        if isinstance(item, dict):
            X, Y = item["X"], item["Y"]
        else:
            X, Y = item

        # 장치 정렬
        if self.device is not None:
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)

        X = torch.atleast_2d(X)  # [B, in_dim]
        Y = torch.as_tensor(Y, device=X.device)
        return X, Y

    # reset: 초기 관측치 제공
    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        X, Y = self._next_batch()
        B = X.shape[0]
        td = TensorDict(
            {
                "observation": X,
                "label": Y,  # 지도학습 타깃을 그대로 노출
                "reward": torch.zeros(B, 1, device=X.device),
                "done": torch.zeros(B, 1, dtype=torch.bool, device=X.device),
                "truncated": torch.zeros(B, 1, dtype=torch.bool, device=X.device),
                "step_count": torch.zeros(B, 1, dtype=torch.int64, device=X.device),
            },
            batch_size=[B],
            device=X.device,
        )
        return td

    # step: 1-step 에피소드(매 스텝 done=True), 다음 배치를 next.*로 예열
    def _step(self, td: TensorDict) -> TensorDict:
        B = td.get("observation").shape[0]
        Xn, Yn = self._next_batch()
        out = td.clone(False)  # 레이아웃만 복사
        out.set("reward", torch.zeros(B, 1, device=out.device))
        out.set("done", torch.ones(B, 1, dtype=torch.bool, device=out.device))
        out.set("truncated", torch.zeros(B, 1, dtype=torch.bool, device=out.device))
        out.set("step_count", torch.ones(B, 1, dtype=torch.int64, device=out.device))
        out.set(("next", "observation"), Xn)
        out.set(("next", "label"), Yn)
        out.set(("next", "reward"), torch.zeros(B, 1, device=out.device))
        out.set(("next", "done"), torch.zeros(B, 1, dtype=torch.bool, device=out.device))
        out.set(("next", "truncated"), torch.zeros(B, 1, dtype=torch.bool, device=out.device))
        out.set(("next", "step_count"), torch.zeros(B, 1, dtype=torch.int64, device=out.device))
        return out


# ---------------------------------------------------------------------------
