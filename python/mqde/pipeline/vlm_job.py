"""VLM quantization + evaluation job (vision-language models such as MiniCPM-V 4.6).

Stages (same event protocol as the LLM / VLA jobs, so the server and the UI treat it alike):
  validate -> inspect -> calib_prepare -> reference -> [attribution] -> quantize -> eval_quantized -> report

What is measured, and against what: the unquantized model answers every evaluation question (greedy, image + text). Those answer
tokens are the reference. A quantized model is then scored *teacher-forced* on the very same answer tokens, which gives, over the
answer positions only (the image and the question are conditioning, not output):

  kl / top1_agreement   how far the next-token distribution moved, and how often the top choice is unchanged (logits probe);
  answer_ppl            perplexity of the reference answers (their negative log-likelihood under the model);
  generation_*          the quantized model's own greedy answer, compared with the reference token for token;
  gt_accuracy           for sources with a known answer (the synthetic scenes): is the right word in the model's answer.

`attribution` quantizes the vision tower and the language model one at a time to say which of them the damage comes from.
`quantize_only` stops after quantization (the layer-by-layer error trace is still produced).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import re
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from mqde.data.vlm import VLMSample, load_split
from mqde.eval.logits_probe import capture_sample_reference, compare_to_sample_reference
from mqde.pipeline.envinfo import device_name
from mqde.pipeline.events import EventSink
from mqde.quant.blockwise import MixedPrecisionPolicy
from mqde.quant.engine import FakeQuantEngine, RecipeSpec
from mqde.quant.suggest import suggest_skips

TOPK = 64  # reference size per position: the top-64 logits and the log-sum-exp (a lower bound of the exact KL, see logits_probe)


@dataclass
class VLMJobSpec:
    name: str
    model_path: str
    recipes: list[str]
    algorithm: str = "recipe"  # rtn | gptq | awq | smoothquant | ...; "recipe" = each recipe's default
    calib_source: str = "synthetic"
    calib_samples: int = 16
    eval_source: str = "synthetic"
    eval_samples: int = 12
    max_new_tokens: int = 24
    device: str = "cuda:0"
    hardware: str = ""
    attribution: bool = True
    # calibrate + quantize only: skips the reference, attribution and evaluation (all three need the unquantized answers)
    quantize_only: bool = False
    group_size_override: int | None = None  # quant.group_size: overrides every per_group recipe's group size
    mixed_precision: dict[str, Any] | None = None  # quant.mixed_precision: {rules: [{match, skip}]}; matching layers stay unquantized
    seed: int = 0
    output_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> VLMJobSpec:
        d = yaml.safe_load(Path(path).read_text())
        q, c, e, dep = d["quant"], d.get("calibration", {}), d.get("evaluation", {}), d.get("deploy", {})
        return cls(
            name=d.get("metadata", {}).get("name", Path(path).stem),
            model_path=d["model"]["path"],
            recipes=q["recipe"] if isinstance(q["recipe"], list) else [q["recipe"]],
            algorithm=str(q.get("algorithm", "recipe")),
            calib_source=c.get("dataset", "synthetic"),
            calib_samples=int(c.get("n_samples", 16)),
            eval_source=e.get("dataset", c.get("dataset", "synthetic")),
            eval_samples=int(e.get("n_samples", 12)),
            max_new_tokens=int(e.get("max_new_tokens", 24)),
            device=dep.get("device", "cuda:0"),
            hardware=str(dep.get("hardware", "")),
            attribution=bool(e.get("attribution", True)),
            quantize_only=bool(e.get("skip", False)),
            group_size_override=int(q["group_size"]) if q.get("group_size") is not None else None,
            mixed_precision=q.get("mixed_precision"),
            seed=int(d.get("seed", 0)),
            output_dir=d.get("output_dir", "runs"),
        )


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _forms(truth: str) -> list[str]:
    """The answer as the model may write it: a single digit also as its English word, and the word also as the digit."""
    t = truth.strip().lower()
    if t.isdigit() and int(t) < 10 and len(t) == 1:
        return [t, NUMBER_WORDS[int(t)]]
    if t in NUMBER_WORDS:
        return [t, str(NUMBER_WORDS.index(t))]
    return [t]


def answer_matches(text: str, truth: str) -> bool:
    """Is the right answer a whole word of the model's reply ("The color is Red." matches "red"; "33" does not match "3";
    "three" matches "3")."""
    low = text.lower()
    return any(re.search(rf"(?<![0-9A-Za-z]){re.escape(f)}(?![0-9A-Za-z])", low) is not None for f in _forms(truth)) if text else False


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


class _Scorer:
    """Teacher-forced logits for every evaluation sample, remembering each sample's answer negative log-likelihood. The probe
    calls it exactly once per sample, so the perplexity costs no extra forward pass."""

    def __init__(self, adapter: Any, inputs: list[dict[str, torch.Tensor]], answers: list[list[int]]) -> None:
        self.adapter, self.inputs, self.answers = adapter, inputs, answers
        self.nll: list[float] = [0.0] * len(answers)

    def __call__(self, i: int) -> torch.Tensor:
        logits = self.adapter.answer_logits(self.inputs[i], self.answers[i])
        n = len(self.answers[i])
        logp = torch.log_softmax(logits[0, :n].float(), dim=-1)
        tgt = torch.tensor(self.answers[i], device=logp.device)
        self.nll[i] = float(-logp.gather(-1, tgt[:, None]).sum())
        return logits

    def ppl(self) -> float:
        return math.exp(sum(self.nll) / sum(len(a) for a in self.answers))


class VLMJob:
    def __init__(self, spec: VLMJobSpec, job_id: str | None = None, adapter_factory: Callable[[VLMJobSpec], Any] | None = None) -> None:
        self.spec = spec
        self.job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.out = Path(spec.output_dir) / self.job_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.ev = EventSink(self.job_id, self.out / "events.jsonl")
        self.adapter_factory = adapter_factory or self._default_adapter
        self.report: dict[str, Any] = {"apiVersion": "mqde/v1", "kind": "vlm", "job_id": self.job_id, "spec": spec.__dict__, "stages": []}

    @staticmethod
    def _default_adapter(spec: VLMJobSpec) -> Any:
        from mqde.models.minicpmv import MiniCPMV

        return MiniCPMV(spec.model_path, device=spec.device)

    # ------------------------------------------------------------------ helpers
    def _stage(self, name: str, fn: Callable[[], Any], **kw: Any) -> Any:
        t0 = self.ev.start_stage(name, **kw)
        try:
            result = fn()
        except Exception as e:
            self.ev.finish_stage(t0, "failed", error=f"{type(e).__name__}: {e}")
            self.report["stages"].append({"stage": name, "status": "failed", "error": str(e)})
            raise
        self.ev.finish_stage(t0)
        self.report["stages"].append({"stage": name, "status": "succeeded", "seconds": round(time.perf_counter() - t0, 3)})
        return result

    def _save(self) -> None:
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2, default=str))

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        s = self.spec
        torch.manual_seed(s.seed)
        self.ev.emit("job_state", state="RUNNING", msg=f"job {self.job_id}")
        self.report["env"] = {
            "git_sha": _git_sha(), "torch": torch.__version__, "cuda": torch.version.cuda, "host": platform.node(),
            "device": s.device, "hardware": s.hardware, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": device_name(s.device),
        }
        try:
            self._run()
            self.report["verdict"] = "succeeded"
            self.ev.emit("job_state", state="SUCCEEDED", msg=str(self.out / "report.json"))
        except Exception as e:
            self.report["verdict"] = "failed"
            self.report["error"] = f"{type(e).__name__}: {e}"
            self.ev.emit("job_state", state="FAILED", msg=self.report["error"])
            raise
        finally:
            self._save()
            self.ev.close()
        return self.report

    def _run(self) -> None:
        s = self.spec
        recipes = self._stage("validate", lambda: [RecipeSpec.load(r) for r in s.recipes], msg=",".join(s.recipes))
        policy = MixedPrecisionPolicy.from_dict(s.mixed_precision)
        if any(pol.bits is not None for _, pol in policy.rules):
            raise ValueError("quant.mixed_precision: 'bits' rules are not supported for VLM jobs (only skip rules)")
        if s.group_size_override is not None:
            for r in recipes:
                if r.weight.granularity == "per_group":
                    self.ev.log(f"group_size override: {r.name} {r.weight.group_size} -> {s.group_size_override}")
                    r.weight = dataclasses.replace(r.weight, group_size=s.group_size_override)  # QuantSpec is frozen
                else:
                    self.ev.log(f"group_size override ignored for {r.name}: granularity is {r.weight.granularity}, not per_group")

        def load() -> Any:
            ad = self.adapter_factory(s)
            self.report["model"] = {"family": ad.family_name, "components": ad.qmodel.summary(), "n_linears": len(ad.qmodel.units)}
            return ad

        ad = self._stage("inspect", load)

        def data() -> tuple[list[VLMSample], list[VLMSample]]:
            calib, ev = load_split(s.calib_source, s.calib_samples, s.eval_source, s.eval_samples, seed=s.seed)
            self.report["data"] = {"calib": {"source": s.calib_source, "n": len(calib)},
                                   "eval": {"source": s.eval_source, "n": len(ev), "has_gt": any(x.answer for x in ev),
                                            "questions": sorted({x.question for x in ev})[:12]}}
            return calib, ev

        calib, evals = self._stage("calib_prepare", data)
        calib_inputs = [ad.preprocess(x) for x in calib]
        eval_inputs = [ad.preprocess(x) for x in evals]
        self._save()

        # reference: the unquantized model's own answers, and its distributions over them
        ref: Any = None
        answers: list[list[int]] = []
        if not s.quantize_only:
            def reference() -> Any:
                t0 = time.perf_counter()
                for inp in eval_inputs:
                    answers.append(ad.generate(inp, s.max_new_tokens))
                gen_s = time.perf_counter() - t0
                scorer = _Scorer(ad, eval_inputs, answers)
                masks = [torch.tensor([True] * len(a) + [False]) for a in answers]  # the last position predicts what follows the answer
                sref = capture_sample_reference(scorer, len(evals), masks, k=TOPK)
                texts = [ad.decode(a) for a in answers]
                gt = [(x, t) for x, t in zip(evals, texts, strict=True) if x.answer]
                self.report["baseline"] = {
                    "answer_ppl": scorer.ppl(), "n_samples": len(evals), "n_answer_tokens": sum(len(a) for a in answers),
                    "gen_seconds_per_sample": gen_s / len(evals), "reference_mib": sref.nbytes() / 2**20,
                    "gt_samples": len(gt), "gt_accuracy": (sum(answer_matches(t, x.answer or "") for x, t in gt) / len(gt)) if gt else None,
                    "examples": [{"question": x.question, "answer": t, "truth": x.answer} for x, t in zip(evals, texts, strict=True)][:8],
                }
                self.ev.emit("metric", name="baseline.answer_ppl", value=round(scorer.ppl(), 4), msg="unquantized answer perplexity")
                return sref

            ref = self._stage("reference", reference)
            self._save()

        def calibrate() -> None:
            for inp in calib_inputs:
                ad.prompt_forward(inp)

        def evaluate(with_generation: bool) -> dict[str, Any]:
            scorer = _Scorer(ad, eval_inputs, answers)
            probe = compare_to_sample_reference(scorer, ref)
            out: dict[str, Any] = {**{k: probe[k] for k in ("kl", "top1_agreement", "ref_top1_prob_drop", "positions")},
                                   "answer_ppl": scorer.ppl()}
            if with_generation:
                gens = [ad.generate(inp, s.max_new_tokens) for inp in eval_inputs]
                pre = [next((i for i, (x, y) in enumerate(zip(g, a, strict=False)) if x != y), min(len(g), len(a))) / max(1, len(a))
                       for g, a in zip(gens, answers, strict=True)]
                texts = [ad.decode(g) for g in gens]
                gt = [(x, t) for x, t in zip(evals, texts, strict=True) if x.answer]
                out.update(generation_exact_match=sum(g == a for g, a in zip(gens, answers, strict=True)) / len(answers),
                           generation_prefix_match=sum(pre) / len(pre),
                           gt_accuracy=(sum(answer_matches(t, x.answer or "") for x, t in gt) / len(gt)) if gt else None,
                           examples=[{"question": x.question, "answer": t, "truth": x.answer}
                                     for x, t in zip(evals, texts, strict=True)][:8])
            return out

        self.report["recipes"] = {}
        for recipe in recipes:
            if not recipe.weight.enabled:
                continue
            algo = recipe.default_algorithm if s.algorithm in ("", "recipe") else s.algorithm
            rep: dict[str, Any] = {"recipe": recipe.name, "weight": recipe.weight.to_dict(), "act": recipe.act.to_dict(),
                                   "maturity": recipe.maturity, "algorithm": algo}
            def on_layer(e: dict[str, Any]) -> None:
                if e["kind"] == "layer":
                    self.ev.emit("trace", **e)

            engine = FakeQuantEngine(ad.qmodel, recipe, progress=on_layer, algorithm=algo)

            def usable(mods: list[str]) -> list[Any]:
                return [u for u in engine.select_units(mods) if not policy.lookup(u.name, u.block).skip]

            modalities = [m for m in dict.fromkeys(recipe.modalities) if usable([m])]
            rep["skipped"] = sorted(u.name for u in engine.select_units(list(recipe.modalities)) if policy.lookup(u.name, u.block).skip)

            if s.attribution and not s.quantize_only:
                def attribution() -> list[dict[str, Any]]:
                    rows = []
                    for mod in modalities:
                        units = usable([mod])
                        engine.apply(units, engine.capture(units, calibrate, max_rows=256) if algo != "rtn" else None)
                        m = evaluate(False)
                        engine.restore()
                        engine.traces.clear()
                        rows.append({"modality": mod, "units": len(units), "kl": m["kl"],
                                     "top1_agreement": m["top1_agreement"], "answer_ppl": m["answer_ppl"]})
                        self.ev.emit("metric", name=f"{recipe.name}.attribution.{mod}.kl", value=m["kl"], msg=f"{recipe.name} only-{mod}")
                    return rows

                rep["attribution"] = self._stage("attribution", attribution, msg=recipe.name)

            def quantize() -> list[dict[str, Any]]:
                units = usable(modalities)
                cache = engine.capture(units, calibrate, max_rows=256)
                traces = engine.apply(units, cache)
                del cache
                return [t.to_dict() for t in traces]

            rep["layers"] = self._stage("quantize", quantize, msg=recipe.name)
            if s.quantize_only:
                rep["eval"], rep["summary"] = {}, self._summarize(rep, None)
            else:
                rep["eval"] = self._stage("eval_quantized", lambda: evaluate(True), msg=recipe.name)
                for k in ("kl", "top1_agreement", "answer_ppl"):
                    self.ev.emit("metric", name=f"{recipe.name}.{k}", value=rep["eval"][k], msg=f"{recipe.name} {k}")
                rep["summary"] = self._summarize(rep, self.report["baseline"])
            engine.restore()
            self.report["recipes"][recipe.name] = rep
            self._save()
        self._stage("report", self._save)

    @staticmethod
    def _summarize(rep: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
        layers = [t for t in rep["layers"] if t.get("out_sqnr_db") is not None]
        worst = sorted(layers, key=lambda t: t["out_sqnr_db"])[:10]
        by_comp: dict[str, list[float]] = {}
        for t in layers:
            by_comp.setdefault(t["component"], []).append(t["out_sqnr_db"])
        out: dict[str, Any] = {
            "suggestion": suggest_skips(layers, bits=int((rep.get("weight") or {}).get("bits") or 4)),
            "worst_layers": [{"name": t["name"], "out_sqnr_db": t["out_sqnr_db"], "act_kurtosis": t["act_kurtosis"]} for t in worst],
            "component_mean_out_sqnr_db": {k: sum(v) / len(v) for k, v in by_comp.items()},
        }
        if base is None:  # quantize_only: nothing was evaluated
            return {**out, "kl": None, "top1_agreement": None, "answer_ppl": None, "answer_ppl_delta_pct": None,
                    "generation_exact_match": None, "generation_prefix_match": None, "gt_accuracy": None, "gt_accuracy_delta": None,
                    "worst_stage": None}
        e, attr = rep["eval"], rep.get("attribution") or []
        acc, acc0 = e.get("gt_accuracy"), base.get("gt_accuracy")
        return {**out, "kl": e["kl"], "top1_agreement": e["top1_agreement"], "answer_ppl": e["answer_ppl"],
                "answer_ppl_delta_pct": 100.0 * (e["answer_ppl"] / base["answer_ppl"] - 1.0),
                "generation_exact_match": e["generation_exact_match"], "generation_prefix_match": e["generation_prefix_match"],
                "gt_accuracy": acc, "gt_accuracy_delta": (acc - acc0) if acc is not None and acc0 is not None else None,
                "worst_stage": max(attr, key=lambda r: r["kl"])["modality"] if attr else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a VLM quantization evaluation job")
    ap.add_argument("job", help="job YAML")
    ap.add_argument("--device", default=None)
    ap.add_argument("--recipes", default=None, help="override recipes, comma separated")
    ap.add_argument("--algorithm", default=None)
    ap.add_argument("--eval-samples", type=int, default=None)
    ap.add_argument("--calib-samples", type=int, default=None)
    ap.add_argument("--quantize-only", action="store_true")
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--job-id", default=None, help="assigned by mqde-server")
    a = ap.parse_args()
    spec = VLMJobSpec.from_yaml(a.job)
    for k, v in {"device": a.device, "algorithm": a.algorithm, "eval_samples": a.eval_samples, "calib_samples": a.calib_samples,
                 "group_size_override": a.group_size}.items():
        if v:
            setattr(spec, k, v)
    if a.recipes:
        spec.recipes = a.recipes.split(",")
    if a.quantize_only:
        spec.quantize_only = True
    job = VLMJob(spec, job_id=a.job_id)
    rep = job.run()
    print(json.dumps({"job_id": job.job_id, "report": str(job.out / "report.json"),
                      "recipes": {k: v["summary"] for k, v in rep["recipes"].items()}}, indent=2, default=str))


if __name__ == "__main__":
    main()
