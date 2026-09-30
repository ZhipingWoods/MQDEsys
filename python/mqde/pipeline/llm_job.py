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
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from mqde.data.text import calib_batches, eval_ids
from mqde.eval.ppl import perplexity
from mqde.models.hf_llm import HFCausalLM
from mqde.pipeline.events import EventSink
from mqde.quant import kv as kvq
from mqde.quant.blockwise import BlockwiseQuantizer, MixedPrecisionPolicy
from mqde.quant.engine import RecipeSpec
from mqde.quant.modules import replace_module
from mqde.quant.numerics import QuantSpec


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
    device: str = "cpu"
    act_static: str | None = None
    kv_method: str = "kivi"  # keys per-channel: int8 is lossless on Qwen2 (outlier key channels), naive per-token is not
    kv_granularity: str = "per_token"
    kv_override: dict[str, Any] | None = None  # e.g. {format: int, bits: 8}
    mixed_precision: dict[str, Any] | None = None
    export: list[str] = field(default_factory=list)  # gptq | awq: real checkpoints + fake/real check (S09/S10)
    llamacpp: list[str] = field(default_factory=list)  # GGUF types to deploy on llama.cpp (CPU): q8_0 q4_0 q4_k q2_k
    sensitivity: str | None = None  # block | type: per-group FP fallback ranking + mixed-precision suggestion
    sensitivity_windows: int = 8
    suggest_target: float = 0.5  # share of the PPL degradation the suggested rules should win back
    seed: int = 0
    output_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMJobSpec:
        d = yaml.safe_load(Path(path).read_text())
        q, c, e, dep = d["quant"], d.get("calibration", {}), d.get("evaluation", {}), d.get("deploy", {})
        return cls(
            name=d.get("metadata", {}).get("name", Path(path).stem),
            model_path=d["model"]["path"],
            recipes=q["recipe"] if isinstance(q["recipe"], list) else [q["recipe"]],
            algorithm=str(q.get("algorithm", "recipe")),
            calib_samples=int(c.get("n_samples", 32)),
            calib_seq_len=int(c.get("seq_len", 512)),
            eval_seq_len=int(e.get("seq_len", 512)),
            eval_windows=int(e.get("max_windows", 40)),
            device=dep.get("device", "cpu"),
            act_static=q.get("act_static"),
            kv_method=q.get("kv", {}).get("method", "kivi") if isinstance(q.get("kv"), dict) else "kivi",
            kv_granularity=q.get("kv", {}).get("granularity", "per_token") if isinstance(q.get("kv"), dict) else "per_token",
            kv_override=q.get("kv") if isinstance(q.get("kv"), dict) and "format" in q.get("kv", {}) else None,
            mixed_precision=q.get("mixed_precision"),
            export=list(d.get("export", {}).get("formats", [])),
            llamacpp=list((dep.get("llamacpp") or {}).get("types", [])),
            sensitivity=(d.get("analysis") or {}).get("sensitivity"),
            sensitivity_windows=int((d.get("analysis") or {}).get("sensitivity_windows", 8)),
            output_dir=d.get("output_dir", "runs"),
        )


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

    def _stage(self, name: str, fn, msg: str = "") -> Any:
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
        r = perplexity(ad.logits, ids, s.eval_seq_len, s.eval_windows,
                       progress=lambda i, n: self.ev.emit("progress", value=i / n, msg=label) if i % 10 == 0 else None)
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
        sens = fallback_sensitivity(ad.model, q.originals, metric, by=s.sensitivity,
                                    progress=lambda i, n, g: self.ev.emit("progress", value=i / n, msg=f"{recipe} FP {g}"))
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
        out: dict[str, Any] = {"protocol": f"llama-perplexity ctx {s.eval_seq_len}, {s.eval_windows} chunks, scores the second half of each chunk",
                               "threads": threads, "results": {}}
        for ty in ["f16", *s.llamacpp]:
            path = f16 if ty == "f16" else d / f"model-{ty}.gguf"
            info = {"type": ty, "bytes": path.stat().st_size} if ty == "f16" else requantize(f16, path, ty)
            self.ev.log(f"llama.cpp {ty}: perplexity + bench")
            p = perplexity(path, text, ctx=s.eval_seq_len, chunks=s.eval_windows, threads=threads)
            b = bench(path, pp=s.eval_seq_len, tg=64, threads=threads, reps=2)
            out["results"][ty] = {**info, **p, **{k: v for k, v in b.items() if k.endswith("per_s")}, "path": str(path)}
            self.ev.emit("metric", name=f"llamacpp.{ty}.ppl", value=p["ppl"], msg=f"llama.cpp {ty}")
        return out

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
        return out

    def _run(self) -> None:
        s = self.spec
        torch.manual_seed(s.seed)
        # GPTQ's Cholesky/GEMM results depend on the BLAS thread count (bit-level), which its sequential error
        # feedback amplifies (~1% PPL for W4 on Qwen2.5-0.5B): the count is part of the reproducibility key.
        self.report["env"] = {"torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": s.device}
        recipes = self._stage("validate", lambda: [RecipeSpec.load(r) for r in s.recipes], ",".join(s.recipes))
        ad = self._stage("inspect", lambda: HFCausalLM(s.model_path, device=s.device))
        self.report["model"] = {"path": s.model_path, "blocks": len(ad.get_blocks()), "linears": len(ad.qmodel.units)}
        cal, ids = self._stage("calib_prepare", lambda: (calib_batches(ad.tokenizer, s.calib_samples, s.calib_seq_len, s.seed), eval_ids(ad.tokenizer)))
        self.report["data"] = {"calib": {"dataset": "wikitext-2 train", "n": len(cal), "seq_len": s.calib_seq_len},
                               "eval": {"dataset": "wikitext-2 test", "windows": s.eval_windows, "seq_len": s.eval_seq_len}}
        base = self._stage("eval_baseline", lambda: self._ppl(ad, ids, "pretrain"))
        self.report["baseline"] = base
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
            wo = self._stage("eval_fake_wo_kv", lambda: self._ppl(ad, ids, f"{recipe.name}.fake_quant_wo_kv"), recipe.name)
            rep["eval"] = {"fake_quant_wo_kv": wo}
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
            q.restore()
            self.report["recipes"][recipe.name] = rep
            self._save()
        if s.llamacpp:
            self.report["llamacpp"] = self._stage("deploy_llamacpp", self._llamacpp, ",".join(s.llamacpp))
            self._save()
        self._stage("report", self._save)


def _recipe_kv(name: str) -> QuantSpec:
    from mqde import core

    return QuantSpec.from_dict(core.load_recipe(name)["kv"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", help="job YAML")
    ap.add_argument("--model")
    ap.add_argument("--recipes")
    ap.add_argument("--algorithm")
    ap.add_argument("--device")
    ap.add_argument("--act-static", choices=["minmax", "ema", "percentile"])
    ap.add_argument("--kv", help="e.g. int8, int4, fp8 (overrides the recipe kv)")
    ap.add_argument("--kv-method", default=None, choices=sorted(kvq.KV_REGISTRY))
    ap.add_argument("--eval-windows", type=int)
    ap.add_argument("--calib-samples", type=int)
    ap.add_argument("--mixed-precision", help="inline JSON or a YAML/JSON file: {rules: [{match, bits, skip}]}")
    ap.add_argument("--export", help="comma separated: gptq,awq")
    ap.add_argument("--sensitivity", choices=["block", "type"])
    ap.add_argument("--llamacpp", help="GGUF types to deploy on llama.cpp, e.g. q8_0,q4_k")
    ap.add_argument("--sensitivity-windows", type=int)
    ap.add_argument("--job-dir")
    ap.add_argument("--job-id", help="server-assigned id; output goes to <output_dir>/<job-id>")
    a = ap.parse_args()
    spec = LLMJobSpec.from_yaml(a.spec) if a.spec else LLMJobSpec(name="llm", model_path=a.model, recipes=["w4a16"])
    for k, v in {"model_path": a.model, "algorithm": a.algorithm, "device": a.device, "act_static": a.act_static,
                 "eval_windows": a.eval_windows, "calib_samples": a.calib_samples, "kv_method": a.kv_method}.items():
        if v is not None:
            setattr(spec, k, v)
    if a.recipes:
        spec.recipes = a.recipes.split(",")
    if a.kv:
        spec.kv_override = {"format": "fp8_e4m3", "bits": 8} if a.kv.startswith("fp8") else {"format": "int", "bits": int(a.kv[3:]), "symmetric": False}
    if a.export:
        spec.export = a.export.split(",")
    if a.sensitivity:
        spec.sensitivity = a.sensitivity
    if a.llamacpp:
        spec.llamacpp = a.llamacpp.split(",")
    if a.sensitivity_windows:
        spec.sensitivity_windows = a.sensitivity_windows
    if a.mixed_precision:
        mp = Path(a.mixed_precision)
        spec.mixed_precision = yaml.safe_load(mp.read_text() if mp.is_file() else a.mixed_precision)
    job = LLMJob(spec, a.job_dir, a.job_id)
    rep = job.run()
    print(json.dumps({"job_id": job.job_id, "baseline_ppl": rep["baseline"]["ppl"],
                      "recipes": {k: v["summary"] | {"algorithm": v["algorithm"]} for k, v in rep["recipes"].items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
