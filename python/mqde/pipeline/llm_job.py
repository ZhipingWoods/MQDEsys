"""LLM quantization + evaluation job (HF causal LMs: Qwen / LLaMA / ...).

Stages (same event protocol as the VLA job so the server/UI show it identically):
  validate -> inspect -> calib_prepare -> eval_baseline (pretrain PPL)
  -> per recipe: quantize (blockwise sequential, chosen algorithm, mixed-precision policy)
               -> eval_fake_wo_kv (weights+acts quantized, KV in FP)
               -> eval_fake (… plus KV cache quantized, when the recipe has kv)
  -> report

Eval positions follow the design doc: pretrain / fake_quant_wo_kv / fake_quant.

    python -m mqde.pipeline.llm_job --model /cache/wuzhiping/hub/Qwen2.5-0.5B-Instruct \
        --recipes w4a16,w4a8 --algorithm gptq --device cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from mqde.data.text import calib_batches, eval_ids
from mqde.eval.logits_probe import capture_reference, compare_to_reference, split_windows
from mqde.eval.ppl import perplexity
from mqde.models.hf_llm import HFCausalLM
from mqde.pipeline.events import EventSink
from mqde.quant import kv as kvq
from mqde.quant.blockwise import BlockwiseQuantizer, MixedPrecisionPolicy
from mqde.quant.engine import RecipeSpec
from mqde.quant.modules import replace_module
from mqde.quant.numerics import QuantSpec


def _opt_dict(v: Any) -> dict[str, Any] | None:
    """YAML option that is either a mapping (settings), a truthy flag (defaults) or absent / false (off)."""
    return dict(v) if isinstance(v, dict) else ({} if v else None)


@dataclass
class LLMJobSpec:
    name: str
    model_path: str
    recipes: list[str]
    algorithm: str = "recipe"
    calib_samples: int = 32
    calib_seq_len: int = 512
    eval_seq_len: int = 512
    eval_windows: int = 40
    device: str = "cpu"  # deploy.device: shared fallback for quant_device/eval_device when they are not set
    # T-1304/T-1306: quantize on one device, evaluate on another (e.g. calibration/GPTQ's sequential error
    # feedback on CPU for determinism, PPL/export on GPU for speed). Each independently falls back to
    # `device` in from_yaml, so an unmodified job (only deploy.device set) gets quant_device == eval_device
    # == device, i.e. today's single-device behavior exactly -- _run() only moves the live model between
    # them when they differ.
    quant_device: str = "cpu"
    eval_device: str = "cpu"
    hardware: str = ""  # deploy.hardware (T-1308): a hardware_db id the submitter picked, purely informational
    act_static: str | None = None
    kv_method: str = "kivi"  # keys per-channel: int8 is lossless on Qwen2 (outlier key channels), naive per-token is not
    kv_granularity: str = "per_token"
    kv_override: dict[str, Any] | None = None  # e.g. {format: int, bits: 8}
    mixed_precision: dict[str, Any] | None = None
    export: list[str] = field(default_factory=list)  # gptq | awq: real checkpoints + fake/real check (S09/S10)
    llamacpp: list[str] = field(default_factory=list)  # GGUF types to deploy on llama.cpp: q8_0 q5_0 q4_0 iq4_nl q4_k q2_k
    # llama.cpp -ngl: None keeps llama.cpp's default, 0 forces CPU, 99 offloads every layer (Metal / CUDA build)
    llamacpp_gpu_layers: int | None = None
    # llama-server per GGUF type: smoke request + closed-loop load (TTFT/TPOT); skipped when the binary is missing
    llamacpp_serve: dict[str, Any] | None = field(default_factory=lambda: {"concurrency": 4, "requests": 16,
                                                                           "prompt_len": 256, "n_predict": 64})
    # S11/S12 on vLLM per exported checkpoint (needs a CUDA device and a vLLM binary, else skipped): {} for defaults, or
    # max_model_len, gpu_memory_utilization, enforce_eager, concurrency, requests, prompt_len, n_predict, ppl_windows
    # and lm_eval: {tasks: [gsm8k], limit: 50, num_fewshot, concurrency, seed, max_gen_toks, timeout} runs the official lm-eval CLI
    # against the served model (ADR-0002; skipped when MQDE_LM_EVAL_BIN / lm-eval is not found)
    vllm: dict[str, Any] | None = None
    # T-0704: the same S11/S12 on SGLang per exported checkpoint (own environment, MQDE_SGLANG_PYTHON; skipped when it
    # is missing or there is no CUDA device): {} for defaults, or max_model_len, mem_fraction_static,
    # disable_cuda_graph, concurrency, requests, prompt_len, n_predict, ppl_windows. None = off.
    sglang: dict[str, Any] | None = None
    # Output-distribution check next to PPL (eval/logits_probe.py): KL and Top-1 agreement against the unquantized model.
    # {} turns it on with defaults, or {windows: 8, k: 256, exact: false}; None = off. top-k KL is a lower bound of the exact KL
    # (about 11% low on Qwen2.5-0.5B at k=256), exact: true keeps every logit (about 1.2 GiB per 8 windows of 512).
    logits_probe: dict[str, Any] | None = None
    sensitivity: str | None = None  # block | type: per-group FP fallback ranking + mixed-precision suggestion
    sensitivity_windows: int = 8
    suggest_target: float = 0.5  # share of the PPL degradation the suggested rules should win back
    # T-1304/T-1306: run calibration + quantization only, skip every stage that evaluates quality (PPL,
    # logits probe, sensitivity) or that reports a quality delta (export's real_vs_fake_ppl_pct, vLLM/
    # llama.cpp serving). export/llamacpp/vllm/sensitivity/logits_probe are still read from the spec but
    # ignored (logged, not silently dropped) when this is set -- validate_jobspec rejects the combination
    # before a server-submitted job gets this far; a direct CLI invocation is trusted to not combine them.
    quantize_only: bool = False
    # T-1304: advanced per-job override of the recipe's own per-group size (quant.group_size in the YAML,
    # the wizard's "高级参数" panel). None = use whatever each recipe's own data/recipes/*.yaml says.
    # Applies only to recipes whose weight granularity is already per_group -- a recipe quantizing
    # per_tensor/per_channel has no group_size concept, so the override is a no-op for it (logged, not a
    # validation error: a job can mix recipes that do and don't use it).
    group_size_override: int | None = None
    seed: int = 0
    output_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMJobSpec:
        d = yaml.safe_load(Path(path).read_text())
        q, c, e, dep = d["quant"], d.get("calibration", {}), d.get("evaluation", {}), d.get("deploy", {})
        dep_device = dep.get("device", "cpu")
        return cls(
            name=d.get("metadata", {}).get("name", Path(path).stem),
            model_path=d["model"]["path"],
            recipes=q["recipe"] if isinstance(q["recipe"], list) else [q["recipe"]],
            algorithm=str(q.get("algorithm", "recipe")),
            calib_samples=int(c.get("n_samples", 32)),
            calib_seq_len=int(c.get("seq_len", 512)),
            eval_seq_len=int(e.get("seq_len", 512)),
            eval_windows=int(e.get("max_windows", 40)),
            device=dep_device,
            # top-level quant_device / evaluation.device (schemas/job_spec.schema.json, T-1305): each falls
            # back to deploy.device, so a job that only sets deploy.device gets both equal to it.
            quant_device=str(d.get("quant_device", dep_device)),
            eval_device=str(e.get("device", dep_device)),
            hardware=str(dep.get("hardware", "")),
            act_static=q.get("act_static"),
            kv_method=q.get("kv", {}).get("method", "kivi") if isinstance(q.get("kv"), dict) else "kivi",
            kv_granularity=q.get("kv", {}).get("granularity", "per_token") if isinstance(q.get("kv"), dict) else "per_token",
            kv_override=q.get("kv") if isinstance(q.get("kv"), dict) and "format" in q.get("kv", {}) else None,
            mixed_precision=q.get("mixed_precision"),
            export=list(d.get("export", {}).get("formats", [])),
            llamacpp=list((dep.get("llamacpp") or {}).get("types", [])),
            llamacpp_gpu_layers=(dep.get("llamacpp") or {}).get("gpu_layers"),
            llamacpp_serve=(dep.get("llamacpp") or {}).get("serve", {"concurrency": 4, "requests": 16,
                                                                    "prompt_len": 256, "n_predict": 64}),
            vllm=(dep.get("vllm") if isinstance(dep.get("vllm"), dict) else ({} if dep.get("vllm") else None)),
            sglang=(dep.get("sglang") if isinstance(dep.get("sglang"), dict) else ({} if dep.get("sglang") else None)),
            logits_probe=_opt_dict((d.get("analysis") or {}).get("logits_probe")),
            sensitivity=(d.get("analysis") or {}).get("sensitivity"),
            sensitivity_windows=int((d.get("analysis") or {}).get("sensitivity_windows", 8)),
            quantize_only=bool(e.get("skip", False)),
            group_size_override=int(q["group_size"]) if q.get("group_size") is not None else None,
            output_dir=d.get("output_dir", "runs"),
        )


def served_max_len(cfg: dict[str, Any], eval_seq_len: int, prompt_len: int, n_predict: int) -> int:
    """max_model_len for the vLLM server: enough for the PPL windows and the load test, and for lm-eval's few-shot prompts."""
    n = max(int(cfg.get("max_model_len", 1024)), eval_seq_len + 16, prompt_len + n_predict + 16)
    if cfg.get("lm_eval"):  # 5-shot gsm8k prompts are about 1.4k tokens and the server rejects longer ones
        n = max(n, int(cfg["lm_eval"].get("max_model_len", 4096)))
    return n


def wants_lm_eval_baseline(vllm: dict[str, Any] | None) -> bool:
    """Task accuracy of a quantized model means nothing without the unquantized model's on the same tasks: on by default."""
    cfg = (vllm or {}).get("lm_eval")
    if not cfg:
        return False
    return bool(cfg.get("baseline", True))


class LLMJob:
    def __init__(self, spec: LLMJobSpec, job_dir: str | Path | None = None, job_id: str | None = None) -> None:
        self.spec = spec
        self.job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.out = Path(job_dir) if job_dir else Path(spec.output_dir) / self.job_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.ev = EventSink(self.job_id, self.out / "events.jsonl")
        self.report: dict[str, Any] = {"job_id": self.job_id, "kind": "llm", "spec": spec.__dict__, "stages": []}
        self._fp_small: float | None = None

    def _save(self) -> None:
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2, default=str))

    def _serve_dir(self, ckpt: Path) -> Path:
        """Where a serving stage writes its logs and lm-eval output: next to the checkpoint this job exported."""
        return ckpt.parent

    def _gguf_dir(self) -> Path:
        return self.out / "export" / "gguf"

    def _stage(self, name: str, fn: Callable[[], Any], msg: str = "") -> Any:
        t0 = self.ev.start_stage(name, msg=msg) if msg else self.ev.start_stage(name)
        try:
            r = fn()
        except Exception as e:
            self.ev.finish_stage(t0, "failed", error=f"{type(e).__name__}: {e}")
            self.report.setdefault("stages", []).append({"stage": name, "status": "failed", "error": str(e)})
            raise
        self.ev.finish_stage(t0)
        self.report.setdefault("stages", []).append({"stage": name, "status": "succeeded", "seconds": round(time.perf_counter() - t0, 3)})
        return r

    def run(self) -> dict[str, Any]:
        self.ev.emit("job_state", state="RUNNING")
        try:
            self._run()
            self.ev.emit("job_state", state="SUCCEEDED")
        except Exception as e:
            self.report["error"] = f"{type(e).__name__}: {e}"
            self.ev.emit("job_state", state="FAILED", msg=self.report["error"])
            raise
        finally:
            self._save()
            self.ev.close()
        return self.report

    def _on_quant_progress(self, e: dict[str, Any]) -> None:
        if e["kind"] == "layer":
            self.ev.emit("trace", **e)
        else:
            self.ev.emit("progress", value=(e["index"] + 1) / e["total"], msg=f"block {e['index'] + 1}/{e['total']}")

    def _ppl(self, ad: HFCausalLM, ids: torch.Tensor, label: str) -> dict[str, float]:
        s = self.spec
        def on_window(i: int, n: int) -> None:
            if i % 10 == 0:
                self.ev.emit("progress", value=i / n, msg=label)

        r = perplexity(ad.logits, ids, s.eval_seq_len, s.eval_windows, progress=on_window)
        self.ev.emit("metric", name=f"{label}.ppl", value=round(r["ppl"], 4), msg=label)
        return r

    def _sensitivity(self, ad: HFCausalLM, ids: torch.Tensor, q: BlockwiseQuantizer, recipe: str) -> dict[str, Any]:
        from mqde.trace.sensitivity import fallback_sensitivity, suggest

        s = self.spec
        metric = lambda: perplexity(ad.logits, ids, s.eval_seq_len, s.sensitivity_windows)["ppl"]  # noqa: E731
        if self._fp_small is None:  # the FP model on the same windows, measured once
            fqs = {n: ad.model.get_submodule(n) for n in q.originals}
            try:
                for n, m in q.originals.items():
                    replace_module(ad.model, n, m)
                self._fp_small = metric()
            finally:
                for n, m in fqs.items():
                    replace_module(ad.model, n, m)
        def on_group(i: int, n: int, g: str) -> None:
            self.ev.emit("progress", value=i / n, msg=f"{recipe} FP {g}")

        assert s.sensitivity is not None  # only called when the stage is enabled
        sens = fallback_sensitivity(ad.model, q.originals, metric, by=s.sensitivity, progress=on_group)
        sens["metric"] = f"ppl@{s.sensitivity_windows}x{s.eval_seq_len}"
        sens["metric_fp"] = self._fp_small
        sens["suggestion"] = suggest(sens, self._fp_small, target_share=s.suggest_target)
        top = sens["ranking"][:3]
        self.ev.log(f"{recipe}: most sensitive {s.sensitivity}s {top}; suggestion {sens['suggestion']['groups']}")
        return sens

    def _llamacpp(self) -> dict[str, Any]:
        """Deploy the FP model as GGUF (F16 + requested block types) on llama.cpp and measure PPL and speed.
        GGUF quantizes the original weights with llama.cpp's own numerics; it is independent of the recipes."""
        from mqde.backend.llamacpp import bench, perplexity
        from mqde.data.text import wikitext_text
        from mqde.export.gguf import convert_f16, requantize

        s = self.spec
        d = self.out / "export" / "gguf"
        d.mkdir(parents=True, exist_ok=True)
        text = d / "wiki.test.txt"
        text.write_text(wikitext_text("test"))
        threads = torch.get_num_threads()
        f16 = convert_f16(s.model_path, d / "model-f16.gguf")
        protocol = f"llama-perplexity ctx {s.eval_seq_len}, {s.eval_windows} chunks, scores the second half of each chunk"
        ngl = s.llamacpp_gpu_layers
        out: dict[str, Any] = {"protocol": protocol, "threads": threads, "gpu_layers": ngl, "results": {}}
        for ty in ["f16", *s.llamacpp]:
            path = f16 if ty == "f16" else d / f"model-{ty}.gguf"
            info = {"type": ty, "bytes": path.stat().st_size} if ty == "f16" else requantize(f16, path, ty)
            self.ev.log(f"llama.cpp {ty}: perplexity + bench")
            p = perplexity(path, text, ctx=s.eval_seq_len, chunks=s.eval_windows, threads=threads, gpu_layers=ngl)
            b = bench(path, pp=s.eval_seq_len, tg=64, threads=threads, reps=2, gpu_layers=ngl)
            out.setdefault("backend", b.get("backend"))
            out["results"][ty] = {**info, **p, **{k: v for k, v in b.items() if k.endswith("per_s")}, "path": str(path)}
            self.ev.emit("metric", name=f"llamacpp.{ty}.ppl", value=p["ppl"], msg=f"llama.cpp {ty}")
            if s.llamacpp_serve:
                out["results"][ty]["serve"] = self._llamacpp_serve(path, ty, threads)
                self._save()
        return out

    def _llamacpp_serve(self, path: Path, ty: str, threads: int) -> dict[str, Any]:
        """llama-server on 127.0.0.1: startup, one chat smoke request, closed-loop streaming load."""
        from mqde.backend.llamacpp_server import LlamaServer, serve_bench, server_bin, smoke

        cfg = dict(self.spec.llamacpp_serve or {})
        try:
            server_bin()
        except FileNotFoundError as e:
            self.ev.log(f"llama-server {ty}: skipped ({e})")
            return {"status": "skipped", "reason": str(e)}
        conc = int(cfg.get("concurrency", 4))
        n_predict = int(cfg.get("n_predict", 64))
        prompt_len = int(cfg.get("prompt_len", 256))
        ctx = int(cfg.get("ctx", conc * (prompt_len + n_predict + 64)))
        srv = LlamaServer(path, self._gguf_dir() / f"server-{ty}.log", threads=threads, ctx=ctx, parallel=conc,
                          gpu_layers=self.spec.llamacpp_gpu_layers)
        self.ev.log(f"llama-server {ty}: starting on {srv.port} (ctx {ctx}, {conc} slots)")
        try:
            with srv:
                sm = smoke(srv.port)
                self.ev.log(f"llama-server {ty}: smoke -> {sm['text']!r}")
                load = serve_bench(srv.port, concurrency=conc, requests=int(cfg.get("requests", 16)),
                                   prompt_len=prompt_len, n_predict=n_predict)
                props = srv.props()
        except Exception as e:
            from mqde.backend.llamacpp_server import diagnose

            self.ev.log(f"llama-server {ty}: failed: {e}")
            return {"status": "failed", "error": str(e)[:1500], "diagnosis": diagnose(srv.log())}
        if load.get("ok"):
            self.ev.emit("metric", name=f"llamacpp.{ty}.serve_tps", value=load["output_tokens_per_s"],
                         msg=f"llama-server {ty} @ {conc}")
        return {"status": "succeeded" if sm["ok"] and load.get("ok") == load["requests"] else "degraded",
                "startup_s": srv.startup_s, "args": srv.args, "props": props, "smoke": sm, "load": load}

    def _write_html(self) -> None:
        """Self-contained report.en.html / report.zh.html next to report.json; a failure here never fails the job."""
        from mqde.report.html import render_html

        for lang in ("en", "zh"):
            try:
                (self.out / f"report.{lang}.html").write_text(render_html(self.report, lang), encoding="utf-8")
            except Exception as e:  # report rendering is a convenience, not a result
                self.ev.log(f"report.{lang}.html not written: {type(e).__name__}: {e}", level="warning")

    def _export(self, ad: HFCausalLM, ids: torch.Tensor, recipe: str, fake_ppl: float) -> dict[str, Any]:
        from mqde.export import get_exporter
        from mqde.export.real_check import real_check, use_real_weights

        out: dict[str, Any] = {}
        for fmt in self.spec.export:
            ex = get_exporter(fmt)
            problems = ex.check(ad.model)
            if problems:
                out[fmt] = {"status": "unsupported", "problems": [{"layer": n, "reason": r} for n, r in problems[:20]],
                            "n_problems": len(problems)}
                self.ev.log(f"{recipe}: {fmt} export skipped: {problems[0][1]} ({len(problems)} layers)", level="warning")
                continue
            d = self.out / "export" / f"{recipe.lower()}-{fmt}"
            res = self._stage("export", lambda: ex.export(ad.model, d, self.spec.model_path), f"{recipe}/{fmt}")
            chk = self._stage("real_check", lambda: real_check(ad.model, d), f"{recipe}/{fmt}")
            chk.pop("per_layer")

            def real_ppl() -> dict[str, float]:
                with use_real_weights(ad.model, d):
                    return self._ppl(ad, ids, f"{recipe}.real_{fmt}")

            rp = self._stage("eval_real", real_ppl, f"{recipe}/{fmt}")
            out[fmt] = {"status": "exported", **res.to_dict(), "real_check": chk, "real_ppl": rp["ppl"],
                        "real_vs_fake_ppl_pct": 100.0 * (rp["ppl"] / fake_ppl - 1.0)}
            if self.spec.vllm is not None:
                out[fmt]["vllm"] = self._stage("deploy_vllm", lambda: self._vllm_serve(d, ids, rp["ppl"]), f"{recipe}/{fmt}")
            if self.spec.sglang is not None:
                out[fmt]["sglang"] = self._stage("deploy_sglang", lambda: self._sglang_serve(d, ids, rp["ppl"]), f"{recipe}/{fmt}")
        return out

    def _vllm_serve(self, ckpt: Path, ids: torch.Tensor, real_ppl: float) -> dict[str, Any]:
        """S11 + S12: serve the exported checkpoint with vLLM, smoke it, load it, and score it with its own logprobs."""
        from mqde.backend.base import openai_models, openai_smoke
        from mqde.backend.vllm_server import VllmServer, vllm_bin
        from mqde.eval.loadtest import LoadSpec, run_load
        from mqde.eval.served_ppl import served_perplexity

        s, cfg = self.spec, dict(self.spec.vllm or {})
        try:
            vllm_bin()
        except FileNotFoundError as e:
            self.ev.log(f"vLLM {ckpt.name}: skipped ({e})")
            return {"status": "skipped", "reason": str(e)}
        if not torch.cuda.is_available():
            self.ev.log(f"vLLM {ckpt.name}: skipped (no CUDA device)")
            return {"status": "skipped", "reason": "no CUDA device"}
        conc, n_predict, prompt_len = int(cfg.get("concurrency", 4)), int(cfg.get("n_predict", 64)), int(cfg.get("prompt_len", 256))
        max_len = served_max_len(cfg, s.eval_seq_len, prompt_len, n_predict)
        srv = VllmServer(ckpt, self._serve_dir(ckpt) / f"vllm-{ckpt.name}.log", max_model_len=max_len, dtype="float16",
                         gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.3)),
                         enforce_eager=bool(cfg.get("enforce_eager", True)), max_num_seqs=max(conc, 8))
        self.ev.log(f"vLLM {ckpt.name}: starting on {srv.port} (max_model_len {max_len})")
        try:
            with srv:
                sm = openai_smoke(srv.port)
                self.ev.log(f"vLLM {ckpt.name}: smoke -> {sm['text']!r}")
                load = run_load(srv.port, LoadSpec(concurrency=conc, requests=int(cfg.get("requests", 16)),
                                                   prompt_len=prompt_len, n_predict=n_predict))
                sp = served_perplexity(srv.port, ids.tolist(), s.eval_seq_len, int(cfg.get("ppl_windows", s.eval_windows)),
                                       concurrency=conc)
                lm = (self._lm_eval(srv.port, ckpt, self._serve_dir(ckpt) / f"lm-eval-{ckpt.name}", cfg["lm_eval"], openai_models(srv.port)[0]["id"])
                      if cfg.get("lm_eval") else None)
                props, engine = srv.props(), srv.engine_info()
        except Exception as e:
            self.ev.log(f"vLLM {ckpt.name}: failed: {e}", level="warning")
            return {"status": "failed", "error": str(e)[:1500], "diagnosis": srv.diagnose()}
        if load.get("ok"):
            self.ev.emit("metric", name=f"vllm.{ckpt.name}.serve_tps", value=load["output_tokens_per_s"], msg=f"vLLM {ckpt.name}")
        self.ev.emit("metric", name=f"vllm.{ckpt.name}.served_ppl", value=sp["ppl"], msg=f"vLLM {ckpt.name}")
        good = bool(sm["ok"]) and load.get("ok") == load["spec"]["requests"]
        return {"status": "succeeded" if good else "degraded", "startup_s": srv.startup_s, "args": srv.args, "props": props,
                "engine": engine, "smoke": sm, "load": load, "served_ppl": sp["ppl"], "served_windows": sp["windows"],
                "served_vs_real_ppl_pct": 100.0 * (sp["ppl"] / real_ppl - 1.0), **({"lm_eval": lm} if lm is not None else {})}

    def _sglang_serve(self, ckpt: Path, ids: torch.Tensor, real_ppl: float) -> dict[str, Any]:
        """S11 + S12 on SGLang: serve the exported checkpoint, smoke it, load it, score it with its own logprobs."""
        from mqde.backend.base import openai_smoke
        from mqde.backend.sglang_server import SglangServer, sglang_python
        from mqde.eval.loadtest import LoadSpec, run_load
        from mqde.eval.served_ppl import served_perplexity

        s, cfg = self.spec, dict(self.spec.sglang or {})
        try:
            sglang_python()
        except FileNotFoundError as e:
            self.ev.log(f"SGLang {ckpt.name}: skipped ({e})")
            return {"status": "skipped", "reason": str(e)}
        if not torch.cuda.is_available():
            self.ev.log(f"SGLang {ckpt.name}: skipped (no CUDA device)")
            return {"status": "skipped", "reason": "no CUDA device"}
        conc, n_predict, prompt_len = int(cfg.get("concurrency", 4)), int(cfg.get("n_predict", 64)), int(cfg.get("prompt_len", 256))
        max_len = served_max_len(cfg, s.eval_seq_len, prompt_len, n_predict)
        srv = SglangServer(ckpt, self._serve_dir(ckpt) / f"sglang-{ckpt.name}.log", max_model_len=max_len, dtype="float16",
                           mem_fraction_static=float(cfg.get("mem_fraction_static", 0.3)),
                           disable_cuda_graph=bool(cfg.get("disable_cuda_graph", False)), max_running_requests=max(conc, 8))
        self.ev.log(f"SGLang {ckpt.name}: starting on {srv.port} (context_length {max_len})")
        try:
            with srv:
                sm = openai_smoke(srv.port)
                self.ev.log(f"SGLang {ckpt.name}: smoke -> {sm['text']!r}")
                load = run_load(srv.port, LoadSpec(concurrency=conc, requests=int(cfg.get("requests", 16)),
                                                   prompt_len=prompt_len, n_predict=n_predict))
                sp = served_perplexity(srv.port, ids.tolist(), s.eval_seq_len, int(cfg.get("ppl_windows", s.eval_windows)),
                                       concurrency=conc, api="sglang")
                props, engine = srv.props(), srv.engine_info()
        except Exception as e:
            self.ev.log(f"SGLang {ckpt.name}: failed: {e}", level="warning")
            return {"status": "failed", "error": str(e)[:1500], "diagnosis": srv.diagnose()}
        if load.get("ok"):
            self.ev.emit("metric", name=f"sglang.{ckpt.name}.serve_tps", value=load["output_tokens_per_s"], msg=f"SGLang {ckpt.name}")
        self.ev.emit("metric", name=f"sglang.{ckpt.name}.served_ppl", value=sp["ppl"], msg=f"SGLang {ckpt.name}")
        good = bool(sm["ok"]) and load.get("ok") == load["spec"]["requests"]
        return {"status": "succeeded" if good else "degraded", "startup_s": srv.startup_s, "args": srv.args, "props": props,
                "engine": engine, "smoke": sm, "load": load, "served_ppl": sp["ppl"], "served_windows": sp["windows"],
                "served_vs_real_ppl_pct": 100.0 * (sp["ppl"] / real_ppl - 1.0)}

    def _lm_eval_baseline(self) -> dict[str, Any]:
        """The same lm-eval run against the unquantized model, served by vLLM with the same dtype as the exported checkpoints."""
        from mqde.backend.base import openai_models
        from mqde.backend.vllm_server import VllmServer, vllm_bin

        s, cfg = self.spec, dict(self.spec.vllm or {})
        try:
            vllm_bin()
        except FileNotFoundError as e:
            return {"status": "skipped", "reason": str(e)}
        if not torch.cuda.is_available():
            return {"status": "skipped", "reason": "no CUDA device"}
        ckpt = Path(s.model_path)
        max_len = served_max_len(cfg, s.eval_seq_len, int(cfg.get("prompt_len", 256)), int(cfg.get("n_predict", 64)))
        srv = VllmServer(ckpt, self.out / "vllm-baseline.log", max_model_len=max_len, dtype="float16",
                         gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.3)),
                         enforce_eager=bool(cfg.get("enforce_eager", True)), max_num_seqs=max(int(cfg.get("concurrency", 4)), 8))
        self.ev.log(f"vLLM baseline {ckpt.name}: starting on {srv.port} (max_model_len {max_len})")
        try:
            with srv:
                res = self._lm_eval(srv.port, ckpt, self.out / "lm-eval-baseline", cfg["lm_eval"], openai_models(srv.port)[0]["id"])
        except Exception as e:
            self.ev.log(f"vLLM baseline: failed: {e}", level="warning")
            return {"status": "failed", "error": str(e)[:1500], "diagnosis": srv.diagnose()}
        return res

    def _lm_eval(self, port: int, model_dir: Path, out_dir: Path, cfg: dict[str, Any], served_name: str) -> dict[str, Any]:
        """S12 task accuracy: the official lm-eval CLI against the server that is already up (ADR-0002). Never fails the job."""
        from mqde.eval.lm_eval_adapter import run_lm_eval

        tasks = [str(t) for t in cfg.get("tasks", ["gsm8k"])]
        try:
            res = run_lm_eval(port, served_name, model_dir, tasks, out_dir, limit=cfg.get("limit", 50),
                              num_fewshot=cfg.get("num_fewshot"), concurrency=int(cfg.get("concurrency", 8)),
                              seed=int(cfg.get("seed", 1234)), max_gen_toks=cfg.get("max_gen_toks"),
                              timeout=float(cfg.get("timeout", 7200)))
        except FileNotFoundError as e:
            self.ev.log(f"lm-eval {model_dir.name}: skipped ({e})")
            return {"status": "skipped", "reason": str(e)}
        except Exception as e:
            self.ev.log(f"lm-eval {model_dir.name}: failed: {e}", level="warning")
            return {"status": "failed", "error": str(e)[:1500]}
        for task, t in res["tasks"].items():
            for key, m in t["metrics"].items():
                self.ev.emit("metric", name=f"lm_eval.{model_dir.name}.{task}.{key}", value=m["value"], msg=f"lm-eval {task}")
        return {"status": "succeeded", **res}

    def _run(self) -> None:
        s = self.spec
        torch.manual_seed(s.seed)
        # GPTQ's Cholesky/GEMM results depend on the BLAS thread count (bit-level), which its sequential error
        # feedback amplifies (~1% PPL for W4 on Qwen2.5-0.5B): the count is part of the reproducibility key.
        def device_name(device: str) -> str:
            return torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") and torch.cuda.is_available() else "cpu"

        self.report["env"] = {
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "quant_device": s.quant_device, "eval_device": s.eval_device,
            # T-1308: the real measured device name (not just "cuda:0"), so Compare.tsx can facet
            # throughput/error by actual hardware across jobs instead of by an opaque device string.
            "quant_device_name": device_name(s.quant_device), "eval_device_name": device_name(s.eval_device),
            "hardware": s.hardware,
        }
        recipes = self._stage("validate", lambda: [RecipeSpec.load(r) for r in s.recipes], ",".join(s.recipes))
        if s.group_size_override is not None:
            for r in recipes:
                if r.weight.granularity == "per_group":
                    self.ev.log(f"group_size override: {r.name} {r.weight.group_size} -> {s.group_size_override}")
                    r.weight = dataclasses.replace(r.weight, group_size=s.group_size_override)  # QuantSpec is frozen
                else:
                    self.ev.log(f"group_size override ignored for {r.name}: granularity is {r.weight.granularity}, not per_group")
        ad = self._stage("inspect", lambda: HFCausalLM(s.model_path, device=s.quant_device))
        self.report["model"] = {"path": s.model_path, "blocks": len(ad.get_blocks()), "linears": len(ad.qmodel.units)}
        prep = lambda: (calib_batches(ad.tokenizer, s.calib_samples, s.calib_seq_len, s.seed), eval_ids(ad.tokenizer))  # noqa: E731
        cal, ids = self._stage("calib_prepare", prep)
        self.report["data"] = {"calib": {"dataset": "wikitext-2 train", "n": len(cal), "seq_len": s.calib_seq_len},
                               "eval": {"dataset": "wikitext-2 test", "windows": s.eval_windows, "seq_len": s.eval_seq_len}}
        probe_windows, probe_ref = None, None
        base: dict[str, float] | None = None
        # T-1304/T-1306: quant_device/eval_device split. Moving only happens when they actually differ, so
        # an unmodified job (both equal to deploy.device) never calls HFCausalLM.to() at all -- same
        # single-device behavior as before this existed, bit-for-bit (see the BLAS-thread-count comment
        # above: GPTQ's result depends on exactly this kind of incidental state staying unchanged).
        moves_devices = s.quant_device != s.eval_device
        if s.quantize_only:
            self.ev.log("quantize_only: skipping eval_baseline, logits probe, sensitivity, export and deploy")
        else:
            if moves_devices:
                ad.to(s.eval_device)
            base = self._stage("eval_baseline", lambda: self._ppl(ad, ids, "pretrain"))
            if wants_lm_eval_baseline(s.vllm):
                self.report["baseline_lm_eval"] = self._stage("lm_eval_baseline", self._lm_eval_baseline)
            if s.logits_probe is not None:
                pc = s.logits_probe
                probe_windows = split_windows(ids, s.eval_seq_len, int(pc.get("windows", 8)))
                probe_ref = self._stage("logits_reference", lambda: capture_reference(
                    ad.logits, probe_windows, k=int(pc.get("k", 256)), exact=bool(pc.get("exact", False))), "pretrain")
                self.report["logits_probe"] = {"windows": int(probe_windows.shape[0]), "seq_len": s.eval_seq_len, "k": probe_ref.k,
                                               "exact": probe_ref.exact, "reference_mib": probe_ref.nbytes() / 2**20}
            self.report["baseline"] = base
            if moves_devices:
                ad.to(s.quant_device)  # back for calibration/quantize below
        self._save()
        policy = MixedPrecisionPolicy.from_dict(s.mixed_precision)
        self.report["recipes"] = {}
        for recipe in recipes:
            if not recipe.weight.enabled:
                continue
            algo = recipe.default_algorithm if s.algorithm in ("", "recipe") else s.algorithm
            rep: dict[str, Any] = {"recipe": recipe.name, "algorithm": algo, "weight": recipe.weight.to_dict(), "act": recipe.act.to_dict()}
            q = BlockwiseQuantizer(ad, recipe, algo, policy=policy, act_static=s.act_static,
                                   progress=self._on_quant_progress)
            res = self._stage("quantize", lambda: q.run(cal), f"{recipe.name}/{algo}")
            rep["layers"] = [t.to_dict() for t in res.traces]
            rep["skipped"] = res.skipped
            rep["quantize_seconds"] = res.seconds
            if s.quantize_only:
                # No evaluation ran, so there is no PPL to report -- worst_layers still comes from the
                # quantize-time trace (weight SQNR, or output SQNR when the algorithm captured activations),
                # not from an eval stage, so it is still meaningful here.
                rep["eval"] = {}
                rep["summary"] = {
                    "ppl": None,
                    "ppl_delta_pct": None,
                    "kv_delta_pct": None,
                    "worst_layers": sorted(({"name": t.name, "out_sqnr_db": t.out_sqnr_db, "w_sqnr_db": t.w_sqnr_db} for t in res.traces),
                                           key=lambda d: d["out_sqnr_db"] if d["out_sqnr_db"] is not None else d["w_sqnr_db"])[:10],
                }
                q.restore()
                self.report["recipes"][recipe.name] = rep
                self._save()
                continue
            assert base is not None
            if moves_devices:
                ad.to(s.eval_device)
            wo = self._stage("eval_fake_wo_kv", lambda: self._ppl(ad, ids, f"{recipe.name}.fake_quant_wo_kv"), recipe.name)
            rep["eval"] = {"fake_quant_wo_kv": wo}
            if probe_ref is not None:
                assert probe_windows is not None
                rep["logits"] = self._stage("eval_logits", lambda: compare_to_reference(ad.logits, probe_windows, probe_ref), recipe.name)
                self.ev.emit("metric", name=f"{recipe.name}.kl", value=round(rep["logits"]["kl"], 5), msg=recipe.name)
            if s.sensitivity:
                rep["sensitivity"] = self._stage("sensitivity", lambda: self._sensitivity(ad, ids, q, recipe.name), recipe.name)
            kv_spec = QuantSpec.from_dict(s.kv_override) if s.kv_override else QuantSpec()
            if not kv_spec.enabled:
                kv_spec = _recipe_kv(recipe.name)
            kq = kvq.from_spec(kv_spec, s.kv_method, s.kv_granularity)
            if kq is not None:
                def with_kv() -> dict[str, float]:
                    with kvq.kv_quant(ad.model, kq):
                        return self._ppl(ad, ids, f"{recipe.name}.fake_quant")
                rep["eval"]["fake_quant"] = self._stage("eval_fake", with_kv, recipe.name)
                rep["kv"] = {**kv_spec.to_dict(), "method": s.kv_method, "granularity": s.kv_granularity}
            if s.export:
                rep["export"] = self._export(ad, ids, recipe.name, wo["ppl"])
            fq = rep["eval"].get("fake_quant", wo)
            rep["summary"] = {
                "ppl": fq["ppl"],
                "ppl_delta_pct": 100.0 * (fq["ppl"] / base["ppl"] - 1.0),
                "kv_delta_pct": 100.0 * (fq["ppl"] / wo["ppl"] - 1.0) if "fake_quant" in rep["eval"] else None,
                # output SQNR when activations were captured, else weight SQNR (RTN weight-only skips capture)
                "worst_layers": sorted(({"name": t.name, "out_sqnr_db": t.out_sqnr_db, "w_sqnr_db": t.w_sqnr_db} for t in res.traces),
                                       key=lambda d: d["out_sqnr_db"] if d["out_sqnr_db"] is not None else d["w_sqnr_db"])[:10],
            }
            if moves_devices:
                ad.to(s.quant_device)  # back before restore(): the detached originals it swaps in were
                                        # never moved and are still sitting on the quant device (see
                                        # HFCausalLM.to()'s docstring)
            q.restore()
            self.report["recipes"][recipe.name] = rep
            self._save()
        if s.llamacpp and s.quantize_only:
            self.ev.log("quantize_only: skipping llama.cpp deploy")
        elif s.llamacpp:
            self.report["llamacpp"] = self._stage("deploy_llamacpp", self._llamacpp, ",".join(s.llamacpp))
            self._save()
        self._stage("report", self._save)
        self._write_html()


def _recipe_kv(name: str) -> QuantSpec:
    from mqde import core

    return QuantSpec.from_dict(core.load_recipe(name)["kv"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", help="job YAML")
    ap.add_argument("--model")
    ap.add_argument("--recipes")
    ap.add_argument("--algorithm")
    ap.add_argument("--device", help="sets quant-device and eval-device together; either flag below overrides just one")
    ap.add_argument("--quant-device")
    ap.add_argument("--eval-device")
    ap.add_argument("--act-static", choices=["minmax", "ema", "percentile"])
    ap.add_argument("--kv", help="e.g. int8, int4, fp8 (overrides the recipe kv)")
    ap.add_argument("--kv-method", default=None, choices=sorted(kvq.KV_REGISTRY))
    ap.add_argument("--eval-windows", type=int)
    ap.add_argument("--calib-samples", type=int)
    ap.add_argument("--mixed-precision", help="inline JSON or a YAML/JSON file: {rules: [{match, bits, skip}]}")
    ap.add_argument("--export", help="comma separated: gptq, awq, compressed-tensors")
    ap.add_argument("--sensitivity", choices=["block", "type"])
    ap.add_argument("--logits-probe", nargs="?", const="{}", help="KL / Top-1 agreement against the unquantized model; "
                    "optional JSON, e.g. '{\"windows\": 8, \"k\": 256, \"exact\": true}'")
    ap.add_argument("--llamacpp", help="GGUF types to deploy on llama.cpp, e.g. q8_0,q4_k")
    ap.add_argument("--vllm", nargs="?", const="{}", help="serve each exported checkpoint with vLLM (S11) and score it (S12); "
                    "optional JSON, e.g. '{\"gpu_memory_utilization\": 0.4}'")
    ap.add_argument("--sglang", nargs="?", const="{}", help="serve each exported checkpoint with SGLang and score it; "
                    "optional JSON, e.g. '{\"mem_fraction_static\": 0.4}'")
    ap.add_argument("--llamacpp-gpu-layers", type=int, help="llama.cpp -ngl (0 = CPU, 99 = offload all on Metal/CUDA)")
    ap.add_argument("--llamacpp-serve", help="llama-server load test as JSON, e.g. '{\"concurrency\": 8}'; 'off' disables")
    ap.add_argument("--sensitivity-windows", type=int)
    ap.add_argument("--quantize-only", action="store_true", help="skip every evaluation/export/deploy stage")
    ap.add_argument("--group-size", type=int, help="override per_group recipes' group_size (no-op for a recipe that isn't per_group)")
    ap.add_argument("--job-dir")
    ap.add_argument("--job-id", help="server-assigned id; output goes to <output_dir>/<job-id>")
    a = ap.parse_args()
    spec = LLMJobSpec.from_yaml(a.spec) if a.spec else LLMJobSpec(name="llm", model_path=a.model, recipes=["w4a16"])
    if a.device is not None:
        spec.device = spec.quant_device = spec.eval_device = a.device
    for k, v in {"model_path": a.model, "algorithm": a.algorithm, "quant_device": a.quant_device, "eval_device": a.eval_device,
                 "act_static": a.act_static, "eval_windows": a.eval_windows, "calib_samples": a.calib_samples, "kv_method": a.kv_method}.items():
        if v is not None:
            setattr(spec, k, v)
    if a.recipes:
        spec.recipes = a.recipes.split(",")
    if a.kv:
        spec.kv_override = (
            {"format": "fp8_e4m3", "bits": 8} if a.kv.startswith("fp8") else {"format": "int", "bits": int(a.kv[3:]), "symmetric": False}
        )
    if a.export:
        spec.export = a.export.split(",")
    if a.sensitivity:
        spec.sensitivity = a.sensitivity
    if a.llamacpp:
        spec.llamacpp = a.llamacpp.split(",")
    if a.vllm is not None:
        spec.vllm = json.loads(a.vllm)
    if a.sglang is not None:
        spec.sglang = json.loads(a.sglang)
    if a.logits_probe is not None:
        spec.logits_probe = json.loads(a.logits_probe)
    if a.llamacpp_gpu_layers is not None:
        spec.llamacpp_gpu_layers = a.llamacpp_gpu_layers
    if a.llamacpp_serve:
        spec.llamacpp_serve = None if a.llamacpp_serve == "off" else json.loads(a.llamacpp_serve)
    if a.sensitivity_windows:
        spec.sensitivity_windows = a.sensitivity_windows
    if a.quantize_only:
        spec.quantize_only = True
    if a.group_size is not None:
        spec.group_size_override = a.group_size
    if a.mixed_precision:
        mp = Path(a.mixed_precision)
        spec.mixed_precision = yaml.safe_load(mp.read_text() if mp.is_file() else a.mixed_precision)
    job = LLMJob(spec, a.job_dir, a.job_id)
    rep = job.run()
    summary = {k: v["summary"] | {"algorithm": v["algorithm"]} for k, v in rep["recipes"].items()}
    print(json.dumps({"job_id": job.job_id, "baseline_ppl": rep["baseline"]["ppl"], "recipes": summary}, indent=1, default=str))


if __name__ == "__main__":
    main()
