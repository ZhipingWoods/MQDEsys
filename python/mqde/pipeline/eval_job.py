"""Evaluate an already-quantized checkpoint (no quantization, no parent job required).

Two ways to name the checkpoint:

* ``model.quantized_path: /path/to/checkpoint`` (with ``model.path`` pointing at the base model for the
  tokenizer / architecture when the checkpoint does not carry them itself), or
* ``artifact: {job, recipe, format}`` / ``{job, format: gguf, type}`` for an export a finished job left behind.

Stages: validate -> eval_prepare -> eval_hf | eval_llamacpp -> (deploy_vllm | deploy_sglang |
deploy_llamacpp, only when ``stages`` includes ``deploy``) -> report.

``eval_hf`` loads the checkpoint with transformers (compressed-tensors works out of the box; AWQ / GPTQ
need their own loaders installed) and scores PPL on ``evaluation.dataset``.  GGUF checkpoints go through
llama.cpp's own ``llama-perplexity``.  A serving backend is optional: it adds served PPL, load numbers and
lm-eval / EvalScope benchmarks, but an offline checkpoint must not require one.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from mqde.backend.llamacpp import bench, perplexity as llamacpp_perplexity
from mqde.data.text import BUILTIN_EVAL, eval_ids, load_text_source
from mqde.eval.offline import run_benchmarks
from mqde.eval.ppl import perplexity
from mqde.eval.registry import suite as benchmark_suite
from mqde.pipeline.deploy_job import load_tokenizer, resolve_artifact
from mqde.pipeline.envinfo import device_name
from mqde.pipeline.events import EventSink
from mqde.pipeline.llm_job import LLMJob, _stages

HF_FORMATS = {"awq", "gptq", "compressed-tensors", "fp8", "bitsandbytes"}
FORMATS = HF_FORMATS | {"gguf", "auto", "fp16", "bf16", "fp32"}


def _benchmarks(e: dict[str, Any]) -> list[str]:
    """evaluation.suite / evaluation.benchmarks -> an ordered, de-duplicated list."""
    names: list[str] = []
    if e.get("suite"):
        names += benchmark_suite(str(e["suite"]))
    b = e.get("benchmarks")
    if isinstance(b, str):
        names += benchmark_suite(b)
    elif isinstance(b, list):
        names += [str(x) for x in b]
    seen: set[str] = set()
    return [x for x in names if not (x in seen or seen.add(x))]


class HFLogits:
    """Just the causal-LM logits an offline PPL needs -- no QuantModel/family assumptions.

    Transformers reads a checkpoint's ``quantization_config`` itself (compressed-tensors out of the box;
    AWQ/GPTQ when their loaders are installed), so an existing quantized directory is enough.
    """

    def __init__(self, path: str | Path, device: str = "cpu", dtype: torch.dtype = torch.float32,
                 tokenizer_path: str | Path | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype, trust_remote_code=False)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or path), trust_remote_code=False)
        # Some tiny / base tokenizers (GPT-2 fixtures, some BERT-derived checkpoints) do not define a
        # pad token. Generation batches one prompt at a time in the fallback path, but transformers
        # still validates padding when padding=True is passed by callers.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids.to(self.device)).logits

    def encode(self, text: str) -> list[int]:
        """Token ids for a benchmark prompt; no special tokens (the scorer compares prefix lengths)."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Greedy, batch-1 decoding. Chat templates are used when the tokenizer defines one."""
        texts = []
        for p in prompts:
            if getattr(self.tokenizer, "chat_template", None):
                texts.append(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
            else:
                texts.append(p)
        enc = self.tokenizer(texts, return_tensors="pt", padding=self.tokenizer.pad_token is not None).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)


def detect_format(path: str | Path) -> str:
    """What kind of checkpoint this is, from the file layout (never trust the extension alone)."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return "gguf"
    for name in ("config.json",):
        cfg = p / name
        if cfg.is_file():
            try:
                d = json.loads(cfg.read_text())
            except Exception:
                continue
            qc = d.get("quantization_config") or {}
            method = str(qc.get("quant_method") or qc.get("quantization_method") or "").lower()
            if method:
                if method in ("compressed-tensors", "compressed_tensors"):
                    return "compressed-tensors"
                return method
            if "quantization_config" in d and not method:
                return "compressed-tensors"
    for name in ("quantize_config.json", "quant_config.json"):
        q = p / name
        if q.is_file():
            try:
                d = json.loads(q.read_text())
            except Exception:
                continue
            return str(d.get("quant_method") or d.get("quantization_method") or "gptq").lower()
    return "fp16"


@dataclass
class EvalJobSpec:
    name: str
    model_path: str  # base model: tokenizer / architecture when the checkpoint does not carry them
    quantized_path: str | None = None  # an already-quantized checkpoint on local disk
    quant_format: str = "auto"  # auto | awq | gptq | compressed-tensors | gguf | fp8 | ...
    artifact: dict[str, Any] | None = None  # alternatively: an export a finished job left behind
    stages: list[str] | None = None  # ["evaluate"] by default; "deploy" adds serving benchmarks
    eval_seq_len: int = 512
    eval_windows: int = 40
    eval_dataset: str = BUILTIN_EVAL
    benchmarks: list[str] = field(default_factory=list)  # offline suite / benchmark names or JSONL paths
    benchmark_limit: int | None = None
    benchmark_max_new_tokens: int = 64
    benchmark_max_windows: int | None = None  # None reuses eval_windows for wikitext2_ppl
    device: str = "cpu"
    hardware: str = ""
    vllm: dict[str, Any] | None = None
    sglang: dict[str, Any] | None = None
    llamacpp: dict[str, Any] | None = None
    output_dir: str = "runs"
    llamacpp_serve: dict[str, Any] | None = None
    llamacpp_gpu_layers: int | None = None

    def __post_init__(self) -> None:
        if self.llamacpp is not None:
            self.llamacpp_serve = self.llamacpp.get(
                "serve", {"concurrency": 4, "requests": 16, "prompt_len": 256, "n_predict": 64})
            self.llamacpp_gpu_layers = self.llamacpp.get("gpu_layers")

    @property
    def plan(self) -> list[str]:
        return _stages(self.stages) if self.stages is not None else ["evaluate"]

    @property
    def deploys(self) -> bool:
        return "deploy" in self.plan

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalJobSpec:
        d = yaml.safe_load(Path(path).read_text())
        m, e, dep = d["model"], d.get("evaluation", {}), d.get("deploy", {})
        return cls(
            name=d.get("metadata", {}).get("name", Path(path).stem),
            model_path=m.get("path") or "",
            quantized_path=m.get("quantized_path"),
            quant_format=str(m.get("quant_format", "auto")),
            artifact=dict(d["artifact"]) if isinstance(d.get("artifact"), dict) else None,
            stages=_stages(d.get("stages")),
            eval_seq_len=int(e.get("seq_len", 512)),
            eval_windows=int(e.get("max_windows", 40)),
            eval_dataset=str(e.get("dataset") or BUILTIN_EVAL),
            benchmarks=_benchmarks(e),
            benchmark_limit=(int(e["limit"]) if e.get("limit") is not None else None),
            benchmark_max_new_tokens=int(e.get("max_new_tokens", 64)),
            benchmark_max_windows=(int(e["max_windows"]) if e.get("max_windows") is not None else None),
            device="cpu" if dep.get("device") == "cpu" else "cuda:0",
            hardware=str(dep.get("hardware", "")),
            vllm=(dep.get("vllm") if isinstance(dep.get("vllm"), dict) else ({} if dep.get("vllm") else None)),
            sglang=(dep.get("sglang") if isinstance(dep.get("sglang"), dict) else ({} if dep.get("sglang") else None)),
            llamacpp=dep.get("llamacpp") if isinstance(dep.get("llamacpp"), dict) else None,
            output_dir=d.get("output_dir", "runs"),
        )


class EvalJob(LLMJob):
    spec: EvalJobSpec  # type: ignore[assignment]

    def __init__(self, spec: EvalJobSpec, job_id: str | None = None) -> None:
        self.spec = spec
        self.job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.out = Path(spec.output_dir) / self.job_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.ev = EventSink(self.job_id, self.out / "events.jsonl")
        self.report: dict[str, Any] = {"job_id": self.job_id, "kind": "llm_eval",
                                       "spec": spec.__dict__, "stages": [], "plan": spec.plan}

    def _serve_dir(self, ckpt: Path) -> Path:
        d = self.out / "serve"
        d.mkdir(exist_ok=True)
        return d

    def _gguf_dir(self) -> Path:
        return self._serve_dir(Path())

    def _resolve(self) -> dict[str, Any]:
        s = self.spec
        if s.artifact:
            art = resolve_artifact(s.artifact, s.output_dir)
            path = Path(art["path"])
            fmt = str(art["format"])
            out = {**art, "path": str(path), "format": fmt, "source": "artifact"}
        else:
            if not s.quantized_path:
                raise ValueError("no checkpoint: set model.quantized_path or artifact")
            path = Path(s.quantized_path)
            if not path.exists():
                raise ValueError(f"model.quantized_path: {path} does not exist")
            fmt = s.quant_format if s.quant_format != "auto" else detect_format(path)
            out = {"path": str(path), "format": fmt, "source": "path"}
        if fmt in ("auto",):
            out["format"] = detect_format(path)
        out["format"] = str(out["format"]).lower()
        if out["format"] not in FORMATS:
            raise ValueError(f"unsupported quant_format '{out['format']}' (known: {sorted(FORMATS)})")
        # A quantized checkpoint usually carries its tokenizer; fall back to the base model when it does not.
        if (path / "tokenizer_config.json").is_file():
            tok = path
        elif s.model_path:
            tok = Path(s.model_path)
        else:
            raise ValueError(f"model.quantized_path ({path}) has no tokenizer_config.json; set model.path to the "
                             "base model so the tokenizer can be read from it")
        out["tokenizer_path"] = str(tok)
        out["base_model_path"] = s.model_path
        return out

    def _eval_hf(self, ckpt: dict[str, Any], ids: torch.Tensor) -> dict[str, Any]:
        """Offline PPL through transformers. Missing AWQ/GPTQ loaders are reported, not disguised as a score."""
        s = self.spec
        cuda = torch.cuda.is_available() and s.device != "cpu"
        dtype = torch.float16 if cuda else torch.float32
        self.ev.log(f"eval_hf: loading {ckpt['path']} ({ckpt['format']}, {dtype})")
        try:
            ad = HFLogits(ckpt["path"], device=s.device, dtype=dtype, tokenizer_path=ckpt.get("tokenizer_path"))
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            hint = ""
            if ckpt["format"] in ("awq", "gptq"):
                hint = (" (install the matching loader -- autoawq for AWQ, gptqmodel/auto-gptq for GPTQ -- "
                        "or configure deploy.vllm / deploy.sglang and add 'deploy' to stages)")
            self.ev.log(f"eval_hf: skipped: {reason}{hint}", level="warning")
            return {"status": "skipped", "reason": reason + hint}
        try:
            r = perplexity(ad.logits, ids, s.eval_seq_len, s.eval_windows,
                           progress=lambda i, n: self.ev.emit("progress", value=i / n, msg="eval_hf") if i % 10 == 0 else None)
            self.ev.emit("metric", name="eval.hf.ppl", value=round(r["ppl"], 4), msg="eval_hf")
            out: dict[str, Any] = {"status": "succeeded", **r, "format": ckpt["format"], "dtype": str(dtype),
                                   "device": device_name(s.device)}
            if s.benchmarks:
                self.ev.log(f"offline benchmarks: {', '.join(s.benchmarks)}")
                out["benchmarks"] = run_benchmarks(
                    s.benchmarks, encode=ad.encode, logits_fn=ad.logits, generate=ad.generate, ppl_ids=ids,
                    limit=s.benchmark_limit, seq_len=s.eval_seq_len,
                    max_windows=s.benchmark_max_windows or s.eval_windows,
                    max_new_tokens=s.benchmark_max_new_tokens,
                    progress=lambda name, i, n: self.ev.emit("progress", value=i / n, msg=f"benchmark {name}"))
                for name, b in out["benchmarks"]["benchmarks"].items():
                    if isinstance(b.get("metrics"), dict):
                        for metric, value in b["metrics"].items():
                            if isinstance(value, (int, float)):
                                self.ev.emit("metric", name=f"eval.{name}.{metric}", value=value, msg=name)
            return out
        finally:
            del ad

    def _eval_llamacpp(self, ckpt: dict[str, Any]) -> dict[str, Any]:
        """GGUF PPL through llama.cpp's own binary; no server needed."""
        s = self.spec
        d = self.out / "eval"
        d.mkdir(exist_ok=True)
        text = d / "eval-corpus.txt"
        text.write_text(load_text_source(s.eval_dataset).text)
        threads = torch.get_num_threads()
        p = llamacpp_perplexity(ckpt["path"], text, ctx=s.eval_seq_len, chunks=s.eval_windows, threads=threads,
                                gpu_layers=self.spec.llamacpp_gpu_layers)
        b = bench(ckpt["path"], pp=s.eval_seq_len, tg=64, threads=threads, reps=2, gpu_layers=self.spec.llamacpp_gpu_layers)
        self.ev.emit("metric", name="eval.llamacpp.ppl", value=p["ppl"], msg="eval_llamacpp")
        return {"status": "succeeded", **p, **{k: v for k, v in b.items() if k.endswith("per_s")},
                "backend": b.get("backend"), "threads": threads, "gpu_layers": self.spec.llamacpp_gpu_layers,
                "format": "gguf"}

    def _run(self) -> None:
        s = self.spec
        if "quantize" in s.plan or "export" in s.plan:
            raise ValueError("eval_job only evaluates an existing checkpoint; use mqde.pipeline.llm_job for "
                             f"quantize/export stages (got {s.plan})")
        self.report["env"] = {"torch": torch.__version__, "device": s.device, "device_name": device_name(s.device),
                              "hardware": s.hardware}
        self.ev.log(f"stages: {' -> '.join(s.plan)}")
        ckpt = self.report["checkpoint"] = self._stage("validate", self._resolve,
                                                       s.quantized_path or json.dumps(s.artifact))
        ids = self._stage("eval_prepare", lambda: eval_ids(load_tokenizer(ckpt["tokenizer_path"]),
                                                           source=s.eval_dataset))
        self.report["data"] = {"eval": {"dataset": s.eval_dataset, "windows": s.eval_windows,
                                        "seq_len": s.eval_seq_len, "benchmarks": s.benchmarks}}
        out: dict[str, Any] = {}
        if ckpt["format"] == "gguf":
            out["llamacpp"] = self._stage("eval_llamacpp", lambda: self._eval_llamacpp(ckpt), "llama.cpp")
        else:
            out["hf"] = self._stage("eval_hf", lambda: self._eval_hf(ckpt, ids), ckpt["format"])
        self.report["eval"] = out
        self._save()
        base_ppl = next((r.get("ppl") for r in out.values() if r.get("status") == "succeeded" and r.get("ppl")), None)

        if s.deploys:
            self.report["deploy"] = {}
            ck = Path(ckpt["path"])
            if s.vllm is not None:
                self.report["deploy"]["vllm"] = self._stage(
                    "deploy_vllm", lambda: self._vllm_serve(ck, ids, base_ppl), ck.name)
                self._save()
            if s.sglang is not None:
                self.report["deploy"]["sglang"] = self._stage(
                    "deploy_sglang", lambda: self._sglang_serve(ck, ids, base_ppl), ck.name)
                self._save()
            if s.llamacpp is not None:
                if ckpt["format"] != "gguf":
                    self.report["deploy"]["llamacpp"] = {
                        "status": "skipped", "reason": f"llama.cpp serves GGUF, this checkpoint is {ckpt['format']}"}
                else:
                    self.report["deploy"]["llamacpp"] = self._stage(
                        "deploy_llamacpp",
                        lambda: self._llamacpp_serve(ck, ckpt.get("type", "gguf"), torch.get_num_threads()),
                        ckpt.get("type", "gguf"))
        served = any(v.get("status") in ("succeeded", "degraded") for v in (self.report.get("deploy") or {}).values())
        if base_ppl is None and not served:
            reasons = [f"{k}: {(v or {}).get('reason') or (v or {}).get('error') or (v or {}).get('status')}"
                       for k, v in out.items()]
            raise RuntimeError("could not evaluate the checkpoint: " + "; ".join(reasons) +
                               ". Configure a serving backend with 'deploy' in stages for checkpoints this "
                               "environment cannot load offline.")
        self._stage("report", self._save)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an already-quantized checkpoint (no quantization)")
    ap.add_argument("job", help="job YAML")
    ap.add_argument("--device", default=None)
    ap.add_argument("--benchmarks", default=None,
                    help="offline benchmark suite/names/JSONL, e.g. quick or wikitext2_ppl,smoke_mcq")
    ap.add_argument("--limit", type=int, default=None, help="max samples per benchmark")
    ap.add_argument("--job-id", default=None, help="assigned by mqde-server")
    a = ap.parse_args()
    spec = EvalJobSpec.from_yaml(a.job)
    if a.device:
        spec.device = a.device
    if a.benchmarks:
        spec.benchmarks = _benchmarks({"benchmarks": a.benchmarks})
    if a.limit is not None:
        spec.benchmark_limit = a.limit
    job = EvalJob(spec, job_id=a.job_id)
    rep = job.run()
    ev = rep.get("eval") or {}
    print(json.dumps({"job_id": job.job_id, "checkpoint": (rep.get("checkpoint") or {}).get("path"),
                      "ppl": {k: v.get("ppl") for k, v in ev.items()},
                      "report": str(job.out / "report.json")}))


if __name__ == "__main__":
    main()
