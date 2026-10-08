"""Exporting a model that was quantized on a GPU (the tensors live on CUDA until they are written)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
sys.path.insert(0, str(Path(__file__).parent))

from _tiny import calib, tiny

from mqde.export import get_exporter
from mqde.export.real_check import real_check
from mqde.quant.blockwise import BlockwiseQuantizer
from mqde.quant.engine import RecipeSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize(("fmt", "recipe"), [("gptq", "w4a16"), ("awq", "w4a16"), ("compressed-tensors", "w4a16"),
                                             ("compressed-tensors", "fp8")])
def test_export_after_cuda_quantization(fmt: str, recipe: str, tmp_path: Path) -> None:
    ad = tiny()
    ad.model.to("cuda")
    ad.device = torch.device("cuda")  # from_model recorded the CPU at construction
    BlockwiseQuantizer(ad, RecipeSpec.load(recipe), "rtn").run(calib())
    get_exporter(fmt).export(ad.model, tmp_path)
    assert real_check(ad.model, tmp_path)["codes_identical"]
