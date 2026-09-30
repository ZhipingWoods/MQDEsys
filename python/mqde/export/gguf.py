"""GGUF export for llama.cpp (T-0705).

1. ``convert_f16``: llama.cpp's own ``convert_hf_to_gguf.py`` writes an F16 GGUF with the architecture and
   tokenizer metadata (we do not re-implement that mapping).
2. ``requantize``: every 2-D ``blk.*.weight`` tensor is re-encoded with ``_core.gguf_quantize``, which is
   bit-identical to ggml's ``quantize_row_*_ref`` (what ``llama-quantize --pure`` does without an imatrix).
   Tensors whose row length is not a multiple of the block size fall back to a 32-element format, like
   llama-quantize does (its fallbacks are Q5_0 / IQ4_NL, which we do not implement, so we use Q4_0).
Embeddings, the output head and norms keep the converter's F16/F32.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mqde import core

BLOCK = {"q4_0": 32, "q8_0": 32, "q4_k": 256, "q2_k": 256}
FALLBACK = {"q4_k": "q4_0", "q2_k": "q4_0", "q4_0": "q8_0", "q8_0": None}
GGML_ID = {"q4_0": 2, "q8_0": 8, "q2_k": 10, "q4_k": 12}
FILE_TYPE = {"q4_0": 2, "q8_0": 7, "q2_k": 10, "q4_k": 14}  # llama_ftype: MOSTLY_Q4_0 / Q8_0 / Q2_K / Q4_K_S
LINEAR = re.compile(r"^blk\.\d+\..+\.weight$")


def llama_cpp_dir() -> Path:
    for c in [os.environ.get("MQDE_LLAMACPP_SRC", ""), *sorted(map(str, Path("/cache/wuzhiping").glob("llama.cpp*")))]:
        if c and (Path(c) / "convert_hf_to_gguf.py").is_file():
            return Path(c)
    raise FileNotFoundError("llama.cpp source with convert_hf_to_gguf.py not found (set MQDE_LLAMACPP_SRC)")


def convert_f16(hf_dir: str | Path, out_path: str | Path, src: Path | None = None) -> Path:
    src = src or llama_cpp_dir()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{src / 'gguf-py'}:{os.environ.get('PYTHONPATH', '')}"}
    cmd = [sys.executable, str(src / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "f16", "--outfile", str(out)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"convert_hf_to_gguf failed ({r.returncode}): {r.stderr[-2000:]}")
    return out


def requantize(src_gguf: str | Path, dst_gguf: str | Path, type_: str) -> dict[str, Any]:
    import gguf

    type_ = type_.lower()
    if type_ not in BLOCK:
        raise ValueError(f"unsupported GGUF type '{type_}' ({', '.join(BLOCK)})")
    r = gguf.GGUFReader(str(src_gguf))
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(str(dst_gguf), arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        val = f.contents()
        if name == "general.file_type":
            val = FILE_TYPE[type_]
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            w.add_key_value(name, val, f.types[0], sub_type=f.types[-1])
        else:
            w.add_key_value(name, val, f.types[0])
    counts: dict[str, int] = {}
    fallback: list[str] = []
    for t in r.tensors:
        data = np.asarray(t.data)
        ne0 = int(t.shape[0])
        src_fp = t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
        if LINEAR.match(t.name) and len(t.shape) == 2 and src_fp:
            ty: str | None = type_
            while ty is not None and ne0 % BLOCK[ty]:
                ty = FALLBACK[ty]
            if ty is not None:
                if ty != type_:
                    fallback.append(t.name)
                rows = data.size // ne0
                q = core.gguf_quantize(data.astype(np.float32).reshape(-1), ty).reshape(rows, -1)
                w.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType(GGML_ID[ty]))
                counts[ty] = counts.get(ty, 0) + 1
                continue
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
        counts[t.tensor_type.name.lower()] = counts.get(t.tensor_type.name.lower(), 0) + 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"path": str(dst_gguf), "type": type_, "tensor_types": counts, "fallback": fallback,
            "bytes": Path(dst_gguf).stat().st_size}


def dequantize_linear_weights(path: str | Path) -> dict[str, np.ndarray]:
    """name -> float32 [rows, ne0] for every quantized blk.* weight, via our dequantizer (== ggml)."""
    import gguf

    out = {}
    for t in gguf.GGUFReader(str(path)).tensors:
        name = t.tensor_type.name.lower()
        if name in BLOCK:
            ne0 = int(t.shape[0])
            n = int(np.prod(t.shape))
            out[t.name] = core.gguf_dequantize(np.asarray(t.data).reshape(-1).view(np.uint8), name, n).reshape(-1, ne0)
    return out
