"""End-to-end llama.cpp deploy check (T-0705): HF checkpoint -> F16 GGUF -> Q4_K GGUF -> llama-perplexity,
llama-bench and llama-server load test.

Default model is a tiny randomly initialised Qwen2 (2 layers, hidden 256) that reuses a real tokenizer, so the
run takes seconds and every linear row length is a multiple of 256 (Q4_K applies with no fallback). Pass
--model to use a real checkpoint instead. Exit code is non-zero on any failure.

    python tests/integration/deploy_llamacpp.py --tokenizer /path/to/Qwen2.5-0.5B-Instruct [--gpu-layers 0]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from mqde.backend.llamacpp import bench, perplexity
from mqde.backend.llamacpp_server import LlamaServer, _request, serve_bench
from mqde.export.gguf import convert_f16, requantize


def tiny_model(tokenizer: str, out: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(tokenizer)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).half()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF checkpoint dir (default: tiny random Qwen2)")
    ap.add_argument("--tokenizer", help="tokenizer dir for the tiny model")
    ap.add_argument("--type", default="q4_k")
    ap.add_argument("--gpu-layers", type=int, default=0, help="-ngl (0 = CPU only)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workdir")
    a = ap.parse_args()
    if not a.model and not a.tokenizer:
        ap.error("--model or --tokenizer is required")

    wd = Path(a.workdir or tempfile.mkdtemp(prefix="mqde-llamacpp-"))
    wd.mkdir(parents=True, exist_ok=True)
    hf = Path(a.model) if a.model else tiny_model(a.tokenizer, wd / "hf")
    f16 = convert_f16(hf, wd / "model-f16.gguf")
    q = requantize(f16, wd / f"model-{a.type}.gguf", a.type)
    rep: dict = {"workdir": str(wd), "gpu_layers": a.gpu_layers, "quant": {k: q[k] for k in ("tensor_types", "fallback", "bytes")}}
    fails: list[str] = []
    if a.type not in q["tensor_types"]:
        fails.append(f"no {a.type} tensors written: {q['tensor_types']}")

    # deterministic text with enough tokens for 2 chunks of 128
    text = wd / "text.txt"
    text.write_text(" ".join(f"The number {i} is followed by {i + 1}." for i in range(400)))
    for name, path in [("f16", f16), (a.type, Path(q["path"]))]:
        p = perplexity(path, text, ctx=128, chunks=2, threads=a.threads, gpu_layers=a.gpu_layers, timeout=600)
        b = bench(path, pp=128, tg=16, threads=a.threads, reps=1, gpu_layers=a.gpu_layers, timeout=600)
        rep[name] = {"ppl": p["ppl"], "pp128": b.get("pp128_tokens_per_s"), "tg16": b.get("tg16_tokens_per_s"), "backend": b.get("backend")}
        if not (math.isfinite(p["ppl"]) and p["ppl"] > 1):
            fails.append(f"{name}: bad ppl {p['ppl']}")

    with LlamaServer(
        q["path"], wd / "server.log", threads=a.threads, ctx=1024, parallel=2, gpu_layers=a.gpu_layers, startup_timeout=120
    ) as srv:
        # fixed-length smoke: a random model may emit EOS at once, so ignore it and require n tokens back
        r = _request(
            srv.port, "POST", "/completion", {"prompt": "Hello", "n_predict": 8, "ignore_eos": True, "temperature": 0}, timeout=120
        )
        n = (r.get("timings") or {}).get("predicted_n") or r.get("tokens_predicted")
        load = serve_bench(srv.port, concurrency=2, requests=4, prompt_len=64, n_predict=16, timeout=120)
    rep["server"] = {
        "startup_s": srv.startup_s,
        "smoke_tokens": n,
        "ok": load.get("ok"),
        "tpot_p50_ms": (load.get("tpot_ms") or {}).get("p50"),
        "tok_s": load.get("output_tokens_per_s"),
    }
    if n != 8:
        fails.append(f"smoke returned {n} tokens, expected 8")
    if load.get("ok") != 4:
        fails.append(f"load: {load.get('ok')}/4 ok, errors {load.get('errors')}")

    rep["fails"] = fails
    print(json.dumps(rep, indent=2))
    print("DEPLOY OK" if not fails else "DEPLOY FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
