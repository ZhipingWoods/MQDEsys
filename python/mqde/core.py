"""Thin, typed wrapper over the C++ extension ``mqde._core`` (JSON results are decoded here)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from mqde import _core  # type: ignore[attr-defined]  # compiled extension, built by CMake


def version() -> str:
    return str(_core.version())


def inspect_model(path: str, tensors: bool = False) -> dict[str, Any]:
    return json.loads(_core.inspect_model(path, tensors))


def estimate(
    model_path: str,
    recipe: str,
    hardware: str = "nvidia/rtx4090",
    backend: str = "hf_transformers",
    **runtime: Any,
) -> dict[str, Any]:
    return json.loads(_core.estimate(model_path, recipe, hardware, backend, json.dumps(runtime)))


def load_recipe(name: str) -> dict[str, Any]:
    return json.loads(_core.load_recipe(name))


def probe() -> list[dict[str, Any]]:
    return json.loads(_core.probe())


def fake_quant(x: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    return _core.fake_quant(np.ascontiguousarray(x, dtype=np.float32), json.dumps(spec))


def quantize_int(x: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    return _core.quantize_int(np.ascontiguousarray(x, dtype=np.float32), json.dumps(spec))


def fp8_round(x: np.ndarray, fmt: str = "e4m3") -> np.ndarray:
    return _core.fp8_round(np.ascontiguousarray(x, dtype=np.float32), fmt)


def measure(ref: np.ndarray, approx: np.ndarray) -> dict[str, float | None]:
    return json.loads(
        _core.measure(np.ascontiguousarray(ref, dtype=np.float32), np.ascontiguousarray(approx, dtype=np.float32))
    )


def tensor_stats(x: np.ndarray) -> dict[str, float]:
    return json.loads(_core.tensor_stats(np.ascontiguousarray(x, dtype=np.float32)))


# ---------------------------------------------------------------- bit packing (T-0405)
def pack_int(
    layout: str, codes: np.ndarray, zeros: np.ndarray, scales: np.ndarray, bits: int, v1: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """codes [out, in] / zeros, scales [out, groups] -> GPTQ or AWQ (qweight int32, qzeros int32, scales fp16 [groups, out])."""
    return _core.pack_int(
        layout,
        np.ascontiguousarray(codes, dtype=np.uint8),
        np.ascontiguousarray(zeros, dtype=np.uint8),
        np.ascontiguousarray(scales, dtype=np.float32),
        bits,
        v1,
    )


def unpack_int(
    layout: str, qweight: np.ndarray, qzeros: np.ndarray, out: int, in_features: int, group_size: int, bits: int, v1: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    return _core.unpack_int(
        layout, np.ascontiguousarray(qweight, dtype=np.int32), np.ascontiguousarray(qzeros, dtype=np.int32),
        out, in_features, group_size, bits, v1,
    )


def gguf_quantize(x: np.ndarray, type_: str) -> np.ndarray:
    return _core.gguf_quantize(np.ascontiguousarray(x, dtype=np.float32).reshape(-1), type_)


def gguf_dequantize(data: np.ndarray, type_: str, n: int) -> np.ndarray:
    return _core.gguf_dequantize(np.ascontiguousarray(data, dtype=np.uint8), type_, n)
