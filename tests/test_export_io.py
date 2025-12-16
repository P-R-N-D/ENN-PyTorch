import builtins
from pathlib import Path
from typing import List, Tuple, Type

import torch
import pytest

from stnet.api import io
from stnet.api.config import ModelConfig
from stnet.backend import export


def _small_model() -> torch.nn.Module:
    cfg = ModelConfig(
        depth=1,
        heads=1,
        spatial_depth=1,
        temporal_depth=1,
        spatial_latents=4,
        temporal_latents=4,
        mlp_ratio=1.0,
        drop_path=0.0,
        dropout=0.0,
    )
    model = io.new_model(4, (2,), cfg)
    for p in model.parameters():
        torch.nn.init.constant_(p, 0.5)
    return model


def _stub_save(traces: List[str]):
    def _save(self, model: torch.nn.Module, dst: Path, *args, **kwargs) -> Tuple[Path, ...]:
        traces.append(f"{type(self).__name__}:{dst}")
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"stub")
        return (dst,)

    return _save


def test_torch_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = _small_model()
    save_path = tmp_path / "model.pt"
    io.save_model(model, save_path)

    loaded = io.load_model(save_path, in_dim=4, out_shape=(2,))
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)


@pytest.mark.parametrize(
    "ext,fmt",
    [
        (".onnx", export.Onnx),
        (".engine", export.TensorRT),
        (".tflite", export.LiteRT),
        (".tf", export.TensorFlow),
        (".mlmodel", export.CoreML),
    ],
)
def test_export_formats_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ext: str, fmt: Type[export.Format]) -> None:
    traces: List[str] = []
    monkeypatch.setattr(fmt, "save", _stub_save(traces))
    model = _small_model()
    out_path = tmp_path / f"export{ext}"
    result = io.save_model(model, out_path)

    assert Path(result).exists()
    assert traces and traces[0].endswith(str(out_path))
