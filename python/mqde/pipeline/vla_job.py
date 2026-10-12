"""End-to-end VLA quantization evaluation job.

Stages (subset of docs/ARCHITECTURE.md §5.1):
  S00 validate      parse + check the job spec
  S01 preflight     C++ estimator: memory / KV / state / concurrency / latency floor
  S03 inspect       load model, build the block view
  S04 calib_prepare load calibration + evaluation samples
  S05 eval_baseline BF16 reference: VL embeddings + actions (fixed flow-matching noise), latency, memory
  S06 attribution   quantize one modality at a time -> end-to-end action error per stage
  S07 quantize      full recipe with per-layer error trace on captured activations
  S08 eval_fake     action / embedding error vs. baseline (+ vs. ground truth when available)
  S14 report        JSON report + events for the UI

Metrics
  action_rel_l2   ||a_q - a_ref|| / ||a_ref|| over the predicted action chunk
  action_cos      cosine(a_q, a_ref)
  vl_cos          cosine of the VLM hidden state fed to the action head
  gt_mae          MAE vs. dataset actions (first k steps, first d dims), both BF16 and quantized
  noise_floor     BF16 vs. BF16 with a different noise seed: the model's own stochastic spread,
                  so quantization error can be judged relative to it
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mqde import core
from mqde.data.vla import load_split
from mqde.models.minicpm_vla import MiniCPMVLA, VLASample
from mqde.pipeline.events import EventSink
from mqde.quant.engine import FakeQuantEngine, RecipeSpec, compare


@dataclass
class JobSpec:
    name: str
    model_path: str
    recipes: list[str]
    algorithm: str = "recipe"  # rtn | gptq | awq | smoothquant | awq+gptq | ...; "recipe" = recipe default
    calib_source: str = "synthetic"
    calib_samples: int = 16
    eval_source: str = "synthetic"
    eval_samples: int = 16
    hardware: str = "nvidia/rtx4090"
    device: str = "cuda:0"
    concurrency: list[int] = field(default_factory=lambda: [1, 4, 16])
    attribution: bool = True
    extra_modalities: list[str] = field(default_factory=list)  # e.g. ["action"]
    # T-1304/T-1306: calibrate + quantize only, skip S05 eval_baseline, S06 attribution and S08 eval_fake --
    # all three need the BF16 reference predictions eval_baseline produces. S01 preflight (pure C++
    # estimation, no model forward pass) and S07 quantize still run.
    quantize_only: bool = False
    # T-1304: advanced per-job override of per_group recipes' group_size; see llm_job.py's LLMJobSpec
    # field of the same name for the full rationale (identical semantics here).
    group_size_override: int | None = None
    seed: int = 0
    output_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> JobSpec:
        d = yaml.safe_load(Path(path).read_text())
        m, q, c, e, dep = d["model"], d["quant"], d.get("calibration", {}), d.get("evaluation", {}), d.get("deploy", {})
        recipes = q["recipe"] if isinstance(q["recipe"], list) else [q["recipe"]]
        return cls(
            name=d.get("metadata", {}).get("name", Path(path).stem),
            model_path=m["path"],
            recipes=recipes,
            algorithm=str(q.get("algorithm", "recipe")),
            calib_source=c.get("dataset", "synthetic"),
            calib_samples=int(c.get("n_samples", 16)),
            eval_source=e.get("dataset", c.get("dataset", "synthetic")),
            eval_samples=int(e.get("n_samples", 16)),
            hardware=dep.get("hardware", "nvidia/rtx4090"),
            device=dep.get("device", "cuda:0"),
            concurrency=list(e.get("concurrency", [1, 4, 16])),
            attribution=bool(e.get("attribution", True)),
            extra_modalities=list(q.get("also_quantize", [])),
            quantize_only=bool(e.get("skip", False)),
            group_size_override=int(q["group_size"]) if q.get("group_size") is not None else None,
            seed=int(d.get("seed", 0)),
            output_dir=d.get("output_dir", "runs"),
        )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def _gt_mae(pred: torch.Tensor, gt: np.ndarray | None) -> float | None:
    if gt is None:
        return None
    k, d = min(pred.shape[0], gt.shape[0]), min(pred.shape[1], gt.shape[1])
    return float(np.abs(pred[:k, :d].cpu().numpy() - gt[:k, :d]).mean())


class VLAJob:
    def __init__(self, spec: JobSpec, job_id: str | None = None) -> None:
        self.spec = spec
        self.job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.out = Path(spec.output_dir) / self.job_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.ev = EventSink(self.job_id, self.out / "events.jsonl")
        self.report: dict[str, Any] = {
            "apiVersion": "mqde/v1",
            "kind": "JobReport",
            "job_id": self.job_id,
            "spec": spec.__dict__,
            "stages": [],
        }

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

    def _run_eval(
        self, vla: MiniCPMVLA, samples: list[VLASample], inputs: list[dict], seed_offset: int = 0
    ) -> list[dict[str, torch.Tensor]]:
        outs = []
        for i, (s, inp) in enumerate(zip(samples, inputs, strict=True)):
            outs.append(vla.predict(s, seed=self.spec.seed + i + seed_offset, inputs=inp))
        return outs

    def _score(self, ref: list[dict], cur: list[dict], samples: list[VLASample]) -> dict[str, Any]:
        per = []
        for r, c, s in zip(ref, cur, samples, strict=True):
            a = compare(r["actions"], c["actions"])
            v = compare(r["vl_embs"], c["vl_embs"])
            per.append(
                {
                    "action_rel_l2": a["rel_l2"],
                    "action_cos": a["cosine"],
                    "action_max_abs": a["max_abs"],
                    "vl_cos": v["cosine"],
                    "vl_sqnr_db": v["sqnr_db"],
                    "gt_mae": _gt_mae(c["actions"], s.gt_action),
                }
            )

        def agg(k: str) -> float | None:
            vals = [float(v) for p in per if (v := p[k]) is not None]
            return float(np.mean(vals)) if vals else None

        return {k: agg(k) for k in per[0]} | {"per_sample": per}

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        spec = self.spec
        torch.manual_seed(spec.seed)
        self.ev.emit("job_state", state="RUNNING", msg=f"job {self.job_id}")
        self.report["env"] = {
            "git_sha": _git_sha(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(torch.device(spec.device)) if torch.cuda.is_available() else "cpu",
            "host": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "mqde_core": core.version(),
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
        spec = self.spec

        # S00 validate
        recipes = self._stage("validate", lambda: [RecipeSpec.load(r) for r in spec.recipes], msg=",".join(spec.recipes))
        if spec.group_size_override is not None:
            for r in recipes:
                if r.weight.granularity == "per_group":
                    self.ev.log(f"group_size override: {r.name} {r.weight.group_size} -> {spec.group_size_override}")
                    r.weight.group_size = spec.group_size_override
                else:
                    self.ev.log(f"group_size override ignored for {r.name}: granularity is {r.weight.granularity}, not per_group")

        # S01 preflight (C++ estimator)
        def preflight() -> dict[str, Any]:
            model_spec = core.inspect_model(spec.model_path)
            est = {}
            for r in [*spec.recipes, "bf16"]:
                est[r] = {
                    str(c): core.estimate(
                        spec.model_path, r, spec.hardware, concurrency=c, images_per_step=2, extra_modalities=spec.extra_modalities
                    )
                    for c in spec.concurrency
                }
                e1 = est[r][str(spec.concurrency[0])]
                self.ev.emit("metric", name=f"estimate.{r}.peak_bytes", value=e1["peak_bytes"], msg=f"estimate {r} peak")
            return {"model": model_spec, "estimates": est}

        self.report["preflight"] = self._stage("preflight", preflight)
        self._save()

        # S03 inspect / load
        def load() -> MiniCPMVLA:
            torch.cuda.reset_peak_memory_stats(spec.device) if torch.cuda.is_available() else None
            vla = MiniCPMVLA(spec.model_path, device=spec.device)
            self.report["model"] = {"blocks": vla.qmodel.summary(), "n_linears": len(vla.qmodel.units)}
            return vla

        vla = self._stage("inspect", load)

        # S04 calibration + evaluation data
        def data() -> tuple[list[VLASample], list[VLASample]]:
            calib, ev = load_split(spec.calib_source, spec.calib_samples, spec.eval_source, spec.eval_samples, seed=spec.seed)
            overlap = {id(s.images[0]) for s in calib} & {id(s.images[0]) for s in ev}
            assert not overlap, "calibration and evaluation samples overlap"
            self.ev.log(f"calib {len(calib)} samples from {spec.calib_source}; eval {len(ev)} from {spec.eval_source}")
            self.report["data"] = {
                "calib": {"source": spec.calib_source, "n": len(calib)},
                "eval": {
                    "source": spec.eval_source,
                    "n": len(ev),
                    "has_gt": any(s.gt_action is not None for s in ev),
                    "tasks": sorted({s.text for s in ev})[:20],
                    "views": len(ev[0].images) if ev else 0,
                },
            }
            return calib, ev

        calib, evals = self._stage("calib_prepare", data)
        calib_inputs = [vla.preprocess(s) for s in calib]
        eval_inputs = [vla.preprocess(s) for s in evals]

        # S05 baseline
        def baseline() -> list[dict]:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(spec.device)
            t0 = time.perf_counter()
            ref = self._run_eval(vla, evals, eval_inputs)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) / len(evals) * 1e3
            noise = self._run_eval(vla, evals, eval_inputs, seed_offset=10_000)
            floor = self._score(ref, noise, evals)
            self.report["baseline"] = {
                "latency_ms_per_step": lat,
                "peak_mem_bytes": torch.cuda.max_memory_allocated(spec.device),
                "gt_mae": floor["gt_mae"]
                if floor["gt_mae"] is None
                else float(np.mean([_gt_mae(r["actions"], s.gt_action) for r, s in zip(ref, evals, strict=True)])),
                "noise_floor": {k: v for k, v in floor.items() if k != "per_sample"},
                "context_tokens": int(eval_inputs[0]["input_ids"].shape[1]),
            }
            self.ev.emit("metric", name="baseline.latency_ms", value=round(lat, 2), msg="BF16 latency/step")
            self.ev.emit("metric", name="baseline.noise_floor.action_rel_l2", value=floor["action_rel_l2"], msg="flow-matching noise floor")
            return ref

        ref: list[dict] | None = None
        if spec.quantize_only:
            self.ev.log("quantize_only: skipping eval_baseline, attribution and eval_fake")
        else:
            ref = self._stage("eval_baseline", baseline)
        self._save()

        def calibrate() -> None:
            for s, inp in zip(calib, calib_inputs, strict=True):
                vla.predict(s, seed=spec.seed, inputs=inp)

        self.report["recipes"] = {}
        for recipe in recipes:
            rep: dict[str, Any] = {"recipe": recipe.name, "weight": recipe.weight.to_dict(),
                                   "act": recipe.act.to_dict(), "maturity": recipe.maturity}
            modalities = list(dict.fromkeys(recipe.modalities + spec.extra_modalities))
            algo = recipe.default_algorithm if spec.algorithm in ("", "recipe") else spec.algorithm
            rep["algorithm"] = algo
            def on_trace(e: dict[str, Any]) -> None:
                if e["kind"] == "layer":
                    self.ev.emit("trace", **e)

            engine = FakeQuantEngine(vla.qmodel, recipe, progress=on_trace, algorithm=algo)
            if not recipe.weight.enabled:
                continue

            # S06 per-modality attribution (needs the BF16 reference from S05, so quantize_only skips it too)
            if spec.attribution and not spec.quantize_only:
                assert ref is not None
                def attribution() -> list[dict[str, Any]]:
                    rows = []
                    for mod in modalities:
                        units = engine.select_units([mod])
                        if not units:
                            continue
                        engine.apply(units, engine.capture(units, calibrate, max_rows=256) if algo != "rtn" else None)
                        cur = self._run_eval(vla, evals, eval_inputs)
                        score = self._score(ref, cur, evals)
                        engine.restore()
                        engine.traces.clear()
                        rows.append({"modality": mod, "units": len(units), **{k: v for k, v in score.items() if k != "per_sample"}})
                        self.ev.emit("metric", name=f"{recipe.name}.attribution.{mod}.action_rel_l2",
                                     value=score["action_rel_l2"], msg=f"{recipe.name} only-{mod}")
                    return rows

                rep["attribution"] = self._stage("attribution", attribution, msg=recipe.name)

            # S07 quantize with per-layer trace on captured calibration activations
            def quantize() -> dict[str, Any]:
                units = engine.select_units(modalities)
                cache = engine.capture(units, calibrate, max_rows=256)
                traces = engine.apply(units, cache)
                del cache
                return {"units": len(units), "layers": [t.to_dict() for t in traces]}

            q = self._stage("quantize", quantize, msg=recipe.name)
            rep["layers"] = q["layers"]

            if spec.quantize_only:
                rep["eval"] = {}
                rep["summary"] = self._summarize(rep, quantize_only=True)
                engine.restore()
                self.report["recipes"][recipe.name] = rep
                self._save()
                continue

            # S08 eval fake-quant
            def eval_fake() -> dict[str, Any]:
                assert ref is not None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(spec.device)
                t0 = time.perf_counter()
                cur = self._run_eval(vla, evals, eval_inputs)
                torch.cuda.synchronize()
                score = self._score(ref, cur, evals)
                score["latency_ms_per_step_fake"] = (time.perf_counter() - t0) / len(evals) * 1e3
                for k in ("action_rel_l2", "action_cos", "vl_cos", "gt_mae"):
                    self.ev.emit("metric", name=f"{recipe.name}.{k}", value=score[k], msg=f"{recipe.name} {k}")
                return score

            rep["eval"] = self._stage("eval_fake", eval_fake, msg=recipe.name)
            rep["summary"] = self._summarize(rep)
            engine.restore()
            self.report["recipes"][recipe.name] = rep
            self._save()

        self._stage("report", self._save)

    def _summarize(self, rep: dict[str, Any], quantize_only: bool = False) -> dict[str, Any]:
        layers = [t for t in rep["layers"] if t.get("out_sqnr_db") is not None]
        worst = sorted(layers, key=lambda t: t["out_sqnr_db"])[:10]
        by_comp: dict[str, list[float]] = {}
        for t in layers:
            by_comp.setdefault(t["component"], []).append(t["out_sqnr_db"])
        # quantize_only ran neither S05 (no self.report["baseline"]) nor S06/S08 (no attribution/eval on rep),
        # so the action-error fields below have nothing to compute from; worst_layers/component means still
        # come entirely from the quantize-time trace captured on calibration data, not from an eval stage.
        if quantize_only:
            return {
                "worst_layers": [
                    {"name": t["name"], "out_sqnr_db": t["out_sqnr_db"], "act_kurtosis": t["act_kurtosis"]} for t in worst
                ],
                "component_mean_out_sqnr_db": {k: float(np.mean(v)) for k, v in by_comp.items()},
                "worst_stage": None,
                "action_rel_l2_over_noise_floor": None,
            }
        attr = rep.get("attribution") or []
        worst_stage = max(attr, key=lambda r: r["action_rel_l2"] or 0.0)["modality"] if attr else None
        nf = self.report["baseline"]["noise_floor"]["action_rel_l2"] or 0.0
        return {
            "worst_layers": [
                {"name": t["name"], "out_sqnr_db": t["out_sqnr_db"], "act_kurtosis": t["act_kurtosis"]} for t in worst
            ],
            "component_mean_out_sqnr_db": {k: float(np.mean(v)) for k, v in by_comp.items()},
            "worst_stage": worst_stage,
            "action_rel_l2_over_noise_floor": (rep["eval"]["action_rel_l2"] / nf) if nf else None,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a VLA quantization evaluation job")
    ap.add_argument("job", help="job YAML")
    ap.add_argument("--device", default=None)
    ap.add_argument("--recipes", default=None, help="override recipes, comma separated")
    ap.add_argument("--algorithm", default=None, help="rtn | gptq | awq | smoothquant | awq+gptq | smoothquant+gptq")
    ap.add_argument("--eval-samples", type=int, default=None)
    ap.add_argument("--calib-samples", type=int, default=None)
    ap.add_argument("--quantize-only", action="store_true", help="skip eval_baseline, attribution and eval_fake")
    ap.add_argument("--group-size", type=int, default=None, help="override per_group recipes' group_size")
    ap.add_argument("--job-id", default=None, help="assigned by mqde-server")
    args = ap.parse_args()
    spec = JobSpec.from_yaml(args.job)
    if args.device:
        spec.device = args.device
    if args.recipes:
        spec.recipes = args.recipes.split(",")
    if args.algorithm:
        spec.algorithm = args.algorithm
    if args.eval_samples:
        spec.eval_samples = args.eval_samples
    if args.calib_samples:
        spec.calib_samples = args.calib_samples
    if args.quantize_only:
        spec.quantize_only = True
    if args.group_size is not None:
        spec.group_size_override = args.group_size
    job = VLAJob(spec, job_id=args.job_id)
    rep = job.run()
    summary = {k: v["summary"] | ({} if spec.quantize_only else {"action_rel_l2": v["eval"]["action_rel_l2"]})
              for k, v in rep["recipes"].items()}
    print(json.dumps({"job_id": job.job_id, "report": str(job.out / "report.json"), "recipes": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
