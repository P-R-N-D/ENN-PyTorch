from __future__ import annotations
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import inspect
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from ..architecture.network import Model, Config


def _require(module: str, pip_hint: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError as err:
        hint = f" (try: {pip_hint})" if pip_hint else ""
        raise ImportError(f"{module} is required for this operation{hint}") from err


def new_model(
    in_dim: int,
    out_shape: Sequence[int],
    config: Config | Dict[str, Any] | None,
) -> Model:
    if config is None:
        cfg = Config()
    elif isinstance(config, dict):
        allowed = set(inspect.signature(Config).parameters.keys())
        filtered = {k: v for k, v in config.items() if k in allowed}
        cfg = Config(**filtered)
    elif isinstance(config, Config):
        cfg = config
    else:
        raise TypeError("config must be Config or dict or None")
    return Model(in_dim, tuple(int(x) for x in out_shape), config=cfg)


def load_model(
    checkpoint_path: str,
    in_dim: Optional[int] = None,
    out_shape: Optional[Sequence[int]] = None,
    config: Config | Dict[str, Any] | None = None,
    map_location: Optional[str] = None,
) -> Model:
    if os.path.isdir(checkpoint_path):
        if in_dim is None or out_shape is None:
            raise ValueError(
                "Loading from a checkpoint directory requires in_dim and out_shape."
            )
        model = new_model(int(in_dim), tuple(out_shape), config)
        opts = StateDictOptions(full_state_dict=True)
        m_sd = get_model_state_dict(model, options=opts)
        load(state_dict={"model": m_sd}, storage_reader=FileSystemReader(checkpoint_path))
        set_model_state_dict(model, m_sd, options=StateDictOptions(strict=False))
        return model
    obj = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=False)
    meta_in_dim = obj.get("in_dim") if isinstance(obj, dict) else None
    meta_out_shape = obj.get("out_shape") if isinstance(obj, dict) else None
    meta_cfg = obj.get("config") if isinstance(obj, dict) else None
    use_in_dim = int(in_dim if in_dim is not None else meta_in_dim)
    use_out_shape = tuple(out_shape if out_shape is not None else (meta_out_shape or ()))
    use_config = (
        config
        if config is not None
        else Config(**meta_cfg) if isinstance(meta_cfg, dict) else Config()
    )
    model = new_model(use_in_dim, use_out_shape, use_config)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(sd)
    return model


def save_model(
    model: Model,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None
) -> str:
    cfg_obj = getattr(model, "_Model__config", None)
    cfg_dict = asdict(cfg_obj) if isinstance(cfg_obj, Config) else asdict(Config())
    payload: Dict[str, Any] = {
        "version": 1,
        "in_dim": int(model.in_dim),
        "out_shape": tuple(int(x) for x in model.out_shape),
        "config": cfg_dict,
        "state_dict": model.state_dict(),
        "pytorch_version": torch.__version__,
    }
    if optimizer is not None:
        if hasattr(optimizer, "state_dict"):
            try:
                payload["optimizer_state_dict"] = optimizer.state_dict()
            except (RuntimeError, NotImplementedError, ValueError, AttributeError):
                pass
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


class _ExportCompat(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_flat, _ = self.net(x, labels_flat=None, net_loss=None)
        return y_flat


def _infer_tensor_shape(
    model: nn.Module,
    sample_input: Optional[torch.Tensor]
) -> Tuple[int, Tuple[int, ...]]:
    in_dim: Optional[int] = None
    out_shape: Optional[Tuple[int, ...]] = None
    if hasattr(model, "in_dim"):
        try:
            in_dim = int(getattr(model, "in_dim"))
        except (TypeError, ValueError):
            in_dim = None
    if hasattr(model, "out_shape"):
        try:
            _shape = getattr(model, "out_shape")
            out_shape = tuple(int(x) for x in _shape)
        except (TypeError, ValueError):
            out_shape = None
    if (in_dim is None or out_shape is None) and sample_input is not None:
        dev = next((p.device for p in model.parameters() if p is not None), torch.device("cpu"))
        sample = sample_input.to(dev)
        model.eval()
        with torch.no_grad():
            if sample.ndim == 1:
                sample = sample.unsqueeze(0)
            y_flat, _ = model(sample, labels_flat=None, net_loss=None)
        if in_dim is None:
            in_dim = int(sample.numel() // sample.shape[0])
        if out_shape is None:
            out_shape = tuple(int(x) for x in y_flat.shape[1:])
            if len(out_shape) == 1:
                out_shape = (out_shape[0],)
    if in_dim is None or out_shape is None:
        raise RuntimeError("failed to infer I/O shapes")
    return int(in_dim), tuple(out_shape)


def _pad_sample(model: nn.Module, sample_input: Optional[torch.Tensor]) -> torch.Tensor:
    if sample_input is not None:
        return sample_input
    in_dim, _ = _infer_tensor_shape(model, sample_input)
    try:
        p = next(model.parameters())
        dtype, device = p.dtype, p.device
    except StopIteration:
        dtype, device = torch.float32, torch.device("cpu")
    return torch.zeros(1, in_dim, dtype=dtype, device=device)


def _to_tensorflow_dtype(t: torch.dtype) -> Any:
    _require("tensorflow", "pip install tensorflow")
    import tensorflow as tf
    mapping = {
        torch.float32: tf.float32,
        torch.float: tf.float32,
        torch.float16: tf.float16,
        torch.bfloat16: tf.bfloat16,
        torch.int64: tf.int64,
        torch.int32: tf.int32,
        torch.int16: tf.int16,
        torch.int8: tf.int8,
        torch.uint8: tf.uint8,
        torch.bool: tf.bool,
    }
    return mapping.get(t, tf.float32)


def to_onnx(
    model: nn.Module,
    onnx_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    **kwargs: Any
) -> str:
    _require("onnx", "pip install onnx")
    wrapper = _ExportCompat(model).eval()
    sample = _pad_sample(model, sample_input)
    input_names = ["features"]
    output_names = ["preds_flat"]
    dynamic_axes = {"features": {0: "batch"}, "preds_flat": {0: "batch"}} if dynamic_batch else None
    export_error = getattr(torch.onnx, "OnnxExporterError", RuntimeError)
    fallback_errors = (RuntimeError,) if export_error is RuntimeError else (RuntimeError, export_error)
    try:
        torch.onnx.export(
            wrapper,
            sample,
            onnx_path,
            dynamo=True,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except fallback_errors:
        torch.onnx.export(
            wrapper,
            sample,
            onnx_path,
            dynamo=False,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    return onnx_path


def to_ort(
    model: nn.Module,
    ort_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    onnx_path: Optional[str] = None,
    opset_version: int = 18,
    optimization_level: str = "all",
    optimization_style: str = "fixed",
    target_platform: Optional[str] = None,
    save_optimized_onnx_model: bool = False,
    **kwargs: Any
) -> Tuple[str, Optional[str]]:
    _require("onnxruntime", "pip install onnxruntime")
    import onnxruntime as ort
    onnx_path = onnx_path or str(Path(ort_path).with_suffix(".onnx"))
    if not Path(onnx_path).exists():
        sample = _pad_sample(model, sample_input)
        to_onnx(model, onnx_path, sample_input=sample, opset_version=opset_version)
    opt_map = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    level = opt_map.get(optimization_level.lower(), ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    disabled_optimizers = (
        ["NchwcTransformer"] if level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL and target_platform != "amd64"
        else None
    )
    optimized_onnx_path: Optional[str] = None
    if save_optimized_onnx_model:
        optimized_onnx_path = str(Path(ort_path).with_suffix(".optimized.onnx"))
        so_onnx = ort.SessionOptions()
        so_onnx.optimized_model_filepath = optimized_onnx_path
        so_onnx.graph_optimization_level = level
        if optimization_style.lower() == "runtime":
            so_onnx.add_session_config_entry("optimization.minimal_build_optimizations", "apply")
        _ = ort.InferenceSession(
            onnx_path,
            sess_options=so_onnx,
            providers=["CPUExecutionProvider"],
            disabled_optimizers=disabled_optimizers,
        )
    so = ort.SessionOptions()
    so.optimized_model_filepath = ort_path
    so.graph_optimization_level = level
    so.add_session_config_entry("session.save_model_format", "ORT")
    if optimization_style.lower() == "runtime":
        so.add_session_config_entry("optimization.minimal_build_optimizations", "save")
    ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
        disabled_optimizers=disabled_optimizers,
    )
    return ort_path, optimized_onnx_path


def to_nnef(
    model: nn.Module,
    nnef_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    onnx_path: Optional[str] = None,
    opset_version: int = 18,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    io_transpose: bool = False,
    compress: bool = True,
    fold_constants: bool = True,
    optimize: bool = True,
    keep_io_names: bool = True,
    **kwargs: Any
) -> str:
    onnx_path = onnx_path or str(Path(nnef_path).with_suffix(".onnx"))
    if not Path(onnx_path).exists():
        sample = _pad_sample(model, sample_input)
        to_onnx(model, onnx_path, sample_input=sample, opset_version=opset_version)
    try:
        import importlib

        importlib.import_module("nnef_tools.convert")
    except ImportError as exc:
        raise ImportError("nnef-tools[onnx] required") from exc
    if input_shapes is None:
        sample = _pad_sample(model, sample_input)
        input_shapes = {"features": tuple(int(x) for x in sample.shape)}
    cmd = [
        os.sys.executable, "-m", "nnef_tools.convert",
        "--input-format", "onnx",
        "--output-format", "nnef",
        "--input-model", onnx_path,
        "--output-model", nnef_path,
        "--input-shapes", json.dumps(input_shapes),
    ]
    if keep_io_names:
        cmd.append("--keep-io-names")
    if io_transpose:
        cmd.append("--io-transpose")
    if fold_constants:
        cmd.append("--fold-constants")
    if optimize:
        cmd.append("--optimize")
    if compress:
        cmd.append("--compress")
    subprocess.run(cmd, check=True)
    return nnef_path


def to_tensorrt(
    model: nn.Module,
    engine_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    onnx_path: Optional[str] = None,
    opset_version: int = 18,
    fp16: bool = True,
    int8: bool = False,
    calibrator: Optional[Any] = None,
    workspace_size_bytes: int = 1 << 30,
    opt_batch: Optional[int] = None,
    max_batch: int = 8,
    **kwargs: Any
) -> str:
    onnx_path = onnx_path or str(Path(engine_path).with_suffix(".onnx"))
    if not Path(onnx_path).exists():
        sample = _pad_sample(model, sample_input)
        to_onnx(model, onnx_path, sample_input=sample, opset_version=opset_version)
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError("tensorrt required") from exc
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_bytes)
        else:
            config.max_workspace_size = workspace_size_bytes
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("TensorRT failed to parse ONNX")
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if int8:
            if not builder.platform_has_fast_int8:
                warnings.warn("INT8 not supported; ignoring")
            else:
                has_int8_layers = any(
                    network.get_layer(i).precision == trt.int8 for i in range(network.num_layers)
                )
                if calibrator is None and not has_int8_layers:
                    warnings.warn("INT8 calibrator not provided")
                config.set_flag(trt.BuilderFlag.INT8)
                if calibrator is not None:
                    config.set_int8_calibrator(calibrator)
        input_tensor = network.get_input(0)
        sample = _pad_sample(model, sample_input)
        s = tuple(int(x) for x in sample.shape)
        min_shape = (1, *s[1:])
        opt_shape = (opt_batch or s[0], *s[1:])
        max_shape = (max_batch, *s[1:])
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("engine build failed")
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
    return engine_path


def to_gguf(
    model: nn.Module,
    gguf_path: str,
    *args: Any,
    hf_model_dir: Optional[str] = None,
    llama_cpp_dir: Optional[str] = None,
    quantize: Optional[str] = None,
    extra_convert_args: Optional[Sequence[str]] = None,
    **kwargs: Any
) -> str:
    if hf_model_dir is None:
        raise NotImplementedError("set hf_model_dir for LLMs")
    repo = Path(llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR", "llama.cpp"))
    convert_py = repo / "convert.py"
    if not convert_py.exists():
        raise FileNotFoundError(f"llama.cpp convert.py not found: {convert_py}")
    cmd = [os.sys.executable, str(convert_py), hf_model_dir, "--outfile", gguf_path]
    if extra_convert_args:
        cmd.extend(extra_convert_args)
    subprocess.run(cmd, check=True)
    if quantize:
        quant_bin = repo / "quantize"
        if os.name == "nt":
            quant_bin = quant_bin.with_suffix(".exe")
        if not quant_bin.exists():
            warnings.warn("quantize binary not found; skipping")
            return gguf_path
        out_q = str(Path(gguf_path).with_suffix(f".{quantize}.gguf"))
        subprocess.run([str(quant_bin), gguf_path, out_q, quantize], check=True)
        return out_q
    return gguf_path


def to_coreml(
    model: nn.Module,
    coreml_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    convert_to: str = "mlprogram",
    compute_units: str = "ALL",
    minimum_deployment_target: Optional[str] = None,
    **kwargs: Any
) -> str:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError("coremltools required") from exc
    sample = _pad_sample(model, sample_input)
    wrapper = _ExportCompat(model).eval()
    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, sample)
    cu_map = {
        "ALL": getattr(ct.ComputeUnit, "ALL", None),
        "CPU_ONLY": getattr(ct.ComputeUnit, "CPU_ONLY", None),
        "CPU_AND_GPU": getattr(ct.ComputeUnit, "CPU_AND_GPU", None),
        "CPU_AND_NE": getattr(ct.ComputeUnit, "CPU_AND_NE", None),
    }
    selected_cu = cu_map.get(compute_units.upper(), cu_map["ALL"])
    kwargs_dict: Dict[str, Any] = {
        "inputs": [ct.TensorType(shape=tuple(int(x) for x in sample.shape))],
        "convert_to": convert_to,
        "compute_units": selected_cu,
    }
    if minimum_deployment_target:
        target = getattr(ct.target, minimum_deployment_target, None)
        if target is not None:
            kwargs_dict["minimum_deployment_target"] = target
    mlmodel = ct.convert(scripted, **kwargs_dict)
    mlmodel.save(coreml_path)
    return coreml_path


def to_litert(
    model: nn.Module,
    tflite_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    onnx_path: Optional[str] = None,
    opset_version: int = 18,
    allow_fp16: bool = False,
    int8_quantize: bool = False,
    representative_dataset: Optional[Callable] = None,
    prefer_onnx2tf: bool = True,
    **kwargs: Any
) -> str:
    onnx_path = onnx_path or str(Path(tflite_path).with_suffix(".onnx"))
    if not Path(onnx_path).exists():
        sample = _pad_sample(model, sample_input)
        to_onnx(model, onnx_path, sample_input=sample, opset_version=opset_version)
    if prefer_onnx2tf:
        try:
            out_dir = Path(tflite_path).with_suffix("")
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                os.sys.executable, "-m", "onnx2tf",
                "-i", onnx_path,
                "-o", str(out_dir),
                "--copy_onnx_input_output_names_to_tflite",
            ]
            subprocess.run(cmd, check=True)
            tflites = list(out_dir.glob("*.tflite"))
            if not tflites:
                raise RuntimeError("onnx2tf did not produce .tflite")
            shutil.copyfile(tflites[0], tflite_path)
            return tflite_path
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            warnings.warn(f"onnx2tf failed; fallback: {exc}")
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError as exc:
        raise ImportError("onnx, onnx-tf, tensorflow required") from exc
    import tensorflow as tf
    model_onnx = onnx.load(onnx_path)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpd:
        saved_model_dir = Path(tmpd) / "saved_model"
        prepare(model_onnx).export_graph(str(saved_model_dir))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if allow_fp16:
            converter.target_spec.supported_types = [tf.float16]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8_quantize:
            if representative_dataset is None:
                raise ValueError("representative_dataset required for INT8")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
    return tflite_path


def to_executorch(
    model: nn.Module,
    pte_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    backend: str = "xnnpack",
    dynamic_shapes: Optional[Dict[str, Dict[int, Any]]] = None,
    separate_weights: bool = False,
    weights_tag: str = "model",
    **kwargs: Any
) -> Tuple[str, Optional[str]]:
    try:
        from executorch.exir import to_edge_transform_and_lower
    except ImportError as exc:
        raise ImportError("executorch required") from exc
    partitioners = []
    b = backend.lower()
    if b == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioners = [XnnpackPartitioner()]
    elif b == "coreml":
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        partitioners = [CoreMLPartitioner()]
    elif b == "vulkan":
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        partitioners = [VulkanPartitioner()]
    else:
        raise ValueError(f"unknown backend: {backend}")
    wrapper = _ExportCompat(model).eval()
    sample = _pad_sample(model, sample_input)
    from torch.export import export
    exported_program = export(wrapper, (sample,), dynamic_shapes=dynamic_shapes)
    transform_passes = None
    if separate_weights:
        from functools import partial
        from executorch.exir.passes.external_constants_pass import (
            delegate_external_constants_pass,
        )
        transform_passes = [
            partial(
                delegate_external_constants_pass,
                ep=exported_program,
                gen_tag_fn=lambda _: weights_tag,
            )
        ]
    lowered = to_edge_transform_and_lower(
        exported_program, transform_passes=transform_passes, partitioner=partitioners
    )
    et_prog = lowered.to_executorch()
    Path(pte_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pte_path, "wb") as f:
        f.write(et_prog.buffer)
    ptd_path: Optional[str] = None
    if separate_weights:
        out_dir = Path(pte_path).parent
        et_prog.write_tensor_data_to_file(out_dir)
        ptd_path = str(out_dir / f"{weights_tag}.ptd")
    return pte_path, ptd_path


def to_script(
    model: nn.Module,
    ts_path: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    use_trace: bool = False,
    freeze: bool = True,
    **kwargs: Any
) -> str:
    wrapper = _ExportCompat(model).eval()
    sample = _pad_sample(model, sample_input)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, sample) if use_trace else torch.jit.script(wrapper)
    if freeze:
        scripted = torch.jit.freeze(scripted)
    Path(ts_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(ts_path)
    return ts_path


def to_tensorflow(
    model: nn.Module,
    export_dir: str,
    *args: Any,
    sample_input: Optional[torch.Tensor] = None,
    onnx_path: Optional[str] = None,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    prefer_onnx2tf: bool = True,
    **kwargs: Any
) -> str:
    os.makedirs(export_dir, exist_ok=True)
    tmpdir = None
    if onnx_path is None:
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="to_tensorflow_")
        onnx_path = os.path.join(tmpdir, "model.onnx")
        to_onnx(
            model,
            onnx_path,
            sample_input=sample_input,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
        )
    saved_dir = export_dir
    converted = False
    errors: list[str] = []
    if prefer_onnx2tf:
        try:
            from onnx2tf import convert
        except ImportError as exc:
            errors.append(f"onnx2tf import failed: {exc!r}")
        else:
            try:
                convert(
                    input_onnx_file_path=onnx_path,
                    output_folder_path=saved_dir,
                    copy_onnx_input_output_names_to_tflite=True,
                    non_verbose=True,
                )
                converted = True
            except (RuntimeError, ValueError, OSError) as exc:
                cmd = [os.sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", saved_dir]
                try:
                    subprocess.run(cmd, check=True)
                    converted = True
                except (OSError, subprocess.CalledProcessError) as cli_exc:
                    errors.append(f"onnx2tf CLI failed: {cli_exc!r}; original: {exc!r}")
    if not converted:
        try:
            import onnx
            from onnx_tf.backend import prepare
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(saved_dir)
            converted = True
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            errors.append(f"onnx-tf failed: {exc!r}")
    if not converted:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("convert failed: " + " / ".join(errors))
    try:
        import tensorflow as tf
        loaded = tf.saved_model.load(saved_dir)
        if "serving_default" not in loaded.signatures:
            if sample_input is None:
                raise RuntimeError("provide sample_input for TF signature wrap")
            tf_dtype = _to_tensorflow_dtype(sample_input.dtype)
            sig_shape = tuple(int(x) for x in sample_input.shape)
            tensor_spec = tf.TensorSpec(shape=sig_shape, dtype=tf_dtype, name="features")

            class _Wrapper(tf.Module):
                def __init__(self, inner: Any) -> None:
                    super().__init__()
                    self.inner = inner

                @tf.function(input_signature=[tensor_spec])
                def __call__(self, features: Any) -> Any:
                    if hasattr(self.inner, "infer"):
                        y = self.inner.infer(features)
                    elif hasattr(self.inner, "__call__"):
                        y = self.inner(features)
                    else:
                        av = list(self.inner.signatures.values())
                        if av:
                            y = av[0](features=features)
                        else:
                            raise RuntimeError("no callable signature")
                    return y if isinstance(y, dict) else {"preds_flat": y}

            wrapper = _Wrapper(loaded)
            tf.saved_model.save(wrapper, saved_dir, signatures=wrapper.__call__.get_concrete_function())
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return saved_dir