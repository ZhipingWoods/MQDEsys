"""World-model / video-generation (DiT) quantization job (T-1005, experimental).

Stages (same event protocol as the LLM / VLM / VLA jobs):
  validate -> inspect -> calib_prepare -> eval_baseline -> [attribution] -> quantize -> eval_quantized -> report

What is measured: the unquantized denoiser samples every evaluation clip with a fixed seed. Those
clips are the reference. A quantized denoiser samples with the *same* seeds, so the difference is
attributable to quantization rather than diffusion noise:

  psnr_vs_ref / mse_vs_ref   fidelity to the FP output (the primary proxy metric)
  psnr_vs_gt  / mse_vs_gt    fidelity to the real synthetic future, when the source has one
  noise_floor                two FP runs with different seeds: the model's own sampling spread
  latency / peak memory      per clip and per frame

``attribution`` quantizes the video backbone and (for EWM) the action path one at a time.
``quantize_only`` stops after quantization; the layer trace is still produced.

LPIPS is intentionally optional: importing ``lpips`` is not a declared dependency, so the report sets
``lpips_vs_ref: null`` with a note instead of silently dropping the metric.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import resource
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mqde.data.world import WorldSample, load_split
from mqde.pipeline.envinfo import device_name
from mqde.pipeline.events import EventSink
from mqde.quant.blockwise import MixedPrecisionPolicy
from mqde.quant.engine import FakeQuantEngine, RecipeSpec
from mqde.quant.suggest import suggest_skips


@dataclass
class WorldJobSpec:
    name: str
    model_path: str
    recipes: list[str]
    algorithm: str = "recipe"
    calib_source: str = "synthetic-video"
    calib_samples: int = 8
    eval_source: str = "synthetic-video"
    eval_samples: int = 8
    frames: int = 4
    sample_steps: int = 8
    device: str = "cuda:0"
    hardware: str = ""
    attribution: bool = True
    quantize_only: bool = False
    group_size_override: int | None = None
    mixed_precision: dict[str, Any] | None = None
    also_quantize: list[str] = dataclasses.field(default_factory=list)
    native_serve: dict[str, Any] | None = None
    action_conditioned: bool = False
    action_dim: int = 128
    seed: int = 0
    output_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorldJobSpec:
        d = yaml.safe_load(Path(path).read_text())
        m, q, c, e, dep = d["model"], d["quant"], d.get("calibration", {}), d.get("evaluation", {}), d.get("deploy", {})
        domain = str(d.get("domain", m.get("type", "world")))
        return cls(
            name=d.get("metadata", {}).get("name", Path(path).stem),
            model_path=m["path"],
            recipes=q["recipe"] if isinstance(q["recipe"], list) else [q["recipe"]],
            algorithm=str(q.get("algorithm", "recipe")),
            calib_source=c.get("dataset", "synthetic-video"),
            calib_samples=int(c.get("n_samples", 8)),
            eval_source=e.get("dataset", c.get("dataset", "synthetic-video")),
            eval_samples=int(e.get("n_samples", 8)),
            frames=int(e.get("frames", c.get("frames", 4))),
            sample_steps=int(e.get("sample_steps", 8)),
            device=dep.get("device", "cuda:0"),
            hardware=str(dep.get("hardware", "")),
            attribution=bool(e.get("attribution", True)),
            quantize_only=bool(e.get("skip", False)),
            group_size_override=int(q["group_size"]) if q.get("group_size") is not None else None,
            mixed_precision=q.get("mixed_precision"),
            also_quantize=[str(x) for x in (q.get("also_quantize") or [])],
            native_serve=(dep.get("native") if isinstance(dep.get("native"), dict)
                          else ({} if dep.get("native") else None)),
            action_conditioned=bool(e.get("action_conditioned", domain == "ewm")),
            action_dim=int(e.get("action_dim", 128)),
            seed=int(d.get("seed", 0)),
            output_dir=d.get("output_dir", "runs"),
        )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def _psnr(mse: float) -> float | None:
    if mse <= 0:
        return 100.0
    if not math.isfinite(mse):
        return None
    return 10.0 * math.log10(1.0 / mse)


def _tensor_score(ref: torch.Tensor, cur: torch.Tensor) -> dict[str, float]:
    a = ref.float().reshape(-1)
    b = cur.float().reshape(-1)
    n = min(a.numel(), b.numel())
    mse = float(torch.mean((a[:n] - b[:n]) ** 2))
    return {"mse": mse, "psnr": _psnr(mse) if _psnr(mse) is not None else float("nan")}


class WorldJob:
    def __init__(self, spec: WorldJobSpec, job_id: str | None = None,
                 adapter_factory: Callable[[WorldJobSpec], Any] | None = None) -> None:
        self.spec = spec
        self.job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.out = Path(spec.output_dir) / self.job_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.ev = EventSink(self.job_id, self.out / "events.jsonl")
        self.adapter_factory = adapter_factory or self._default_adapter
        self.kind = "ewm" if spec.action_conditioned else "world"
        self.report: dict[str, Any] = {
            "apiVersion": "mqde/v1", "kind": self.kind, "job_id": self.job_id,
            "spec": spec.__dict__, "stages": [],
        }

    @staticmethod
    def _default_adapter(spec: WorldJobSpec) -> Any:
        from mqde.models.world import load_world_adapter

        return load_world_adapter(spec.model_path, device=spec.device, action_conditioned=spec.action_conditioned, seed=spec.seed)

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

    @staticmethod
    def _peak_mem(device: str) -> tuple[int, str]:
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            return int(torch.cuda.max_memory_allocated(device)), "CUDA max_memory_allocated"
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "process peak RSS (Linux ru_maxrss)"

    def _generate(self, ad: Any, samples: list[WorldSample], inputs: list[dict[str, Any]], seed_offset: int = 0) -> list[torch.Tensor]:
        return [ad.generate(s, seed=self.spec.seed + i + seed_offset, inputs=inp) for i, (s, inp) in enumerate(zip(samples, inputs, strict=True))]

    @staticmethod
    def _score(ref: list[torch.Tensor], cur: list[torch.Tensor], samples: list[WorldSample]) -> dict[str, Any]:
        per: list[dict[str, Any]] = []
        for r, c, s in zip(ref, cur, samples, strict=True):
            d = _tensor_score(r, c)
            gt = torch.from_numpy(np.asarray(s.frames, dtype=np.float32)).permute(0, 3, 1, 2)
            g = _tensor_score(gt, c)
            per.append({**d, "psnr_vs_gt": g["psnr"], "mse_vs_gt": g["mse"], "seed_id": s.meta.get("id")})
        return {
            "psnr_vs_ref": float(np.mean([p["psnr"] for p in per])),
            "mse_vs_ref": float(np.mean([p["mse"] for p in per])),
            "psnr_vs_gt": float(np.mean([p["psnr_vs_gt"] for p in per])),
            "mse_vs_gt": float(np.mean([p["mse_vs_gt"] for p in per])),
            "per_sample": per,
        }

    def _lpips(self, ref: list[torch.Tensor], cur: list[torch.Tensor]) -> float | None:
        try:
            import lpips  # type: ignore[import-not-found]
        except Exception:
            return None
        try:
            net = lpips.LPIPS(net="alex", verbose=False).to(self.spec.device)
            vals = []
            for r, c in zip(ref, cur, strict=True):
                # lpips wants [-1, 1]
                vals.append(float(net(r.to(self.spec.device) * 2 - 1, c.to(self.spec.device) * 2 - 1).mean()))
            return float(np.mean(vals))
        except Exception:
            return None

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
            raise ValueError("quant.mixed_precision: 'bits' rules are not supported for world jobs (only skip rules)")
        if s.group_size_override is not None:
            for r in recipes:
                if r.weight.granularity == "per_group":
                    self.ev.log(f"group_size override: {r.name} {r.weight.group_size} -> {s.group_size_override}")
                    r.weight = dataclasses.replace(r.weight, group_size=s.group_size_override)
                else:
                    self.ev.log(f"group_size override ignored for {r.name}: granularity is {r.weight.granularity}, not per_group")

        def load() -> Any:
            ad = self.adapter_factory(s)
            comps = ad.qmodel.summary()
            self.report["model"] = {
                "family": getattr(ad, "family_name", s.action_conditioned and "dit_ewm" or "dit_world"),
                "weights_source": getattr(ad, "weights_source", "unknown"),
                "arch": getattr(ad, "cfg", {}).get("arch", "diffusers"),
                "frames": s.frames,
                "action_conditioned": s.action_conditioned,
                "action_dim": s.action_dim,
                "components": comps,
                "n_linears": len(ad.qmodel.units),
            }
            self.ev.log(f"world model loaded: {self.report['model']['family']} ({self.report['model']['weights_source']}), "
                        f"{len(ad.qmodel.units)} linear layers")
            return ad

        ad = self._stage("inspect", load)

        def data() -> tuple[list[WorldSample], list[WorldSample]]:
            calib, ev = load_split(s.calib_source, s.calib_samples, s.eval_source, s.eval_samples,
                                   seed=s.seed, frames=s.frames, action_conditioned=s.action_conditioned, action_dim=s.action_dim)
            if any(x.frames.shape[0] != s.frames for x in ev):
                self.ev.log(f"note: eval clips have a different frame count than requested ({s.frames})", level="warning")
            self.report["data"] = {
                "calib": {"source": s.calib_source, "n": len(calib), "frames": s.frames},
                "eval": {"source": s.eval_source, "n": len(ev), "frames": s.frames, "has_gt": True,
                         "action_conditioned": s.action_conditioned, "action_dim": s.action_dim if s.action_conditioned else 0,
                         "prompts": sorted({x.prompt for x in ev})[:8]},
            }
            return calib, ev

        calib, evals = self._stage("calib_prepare", data)
        calib_inputs = [ad.preprocess(x) for x in calib]
        eval_inputs = [ad.preprocess(x) for x in evals]
        self._save()

        ref: list[torch.Tensor] | None = None
        if not s.quantize_only:
            def baseline() -> list[torch.Tensor]:
                if torch.cuda.is_available() and str(s.device).startswith("cuda"):
                    torch.cuda.synchronize(s.device)
                    torch.cuda.reset_peak_memory_stats(s.device)
                t0 = time.perf_counter()
                gen = self._generate(ad, evals, eval_inputs)
                if torch.cuda.is_available() and str(s.device).startswith("cuda"):
                    torch.cuda.synchronize(s.device)
                lat = (time.perf_counter() - t0) / max(1, len(evals)) * 1e3
                noise = self._generate(ad, evals, eval_inputs, seed_offset=10_000)
                floor = self._score(gen, noise, evals)
                gt_psnr = float(np.mean([
                    _tensor_score(torch.from_numpy(np.asarray(x.frames, np.float32)).permute(0, 3, 1, 2), g)["psnr"]
                    for x, g in zip(evals, gen, strict=True)
                ]))
                gt_mse = float(np.mean([
                    _tensor_score(torch.from_numpy(np.asarray(x.frames, np.float32)).permute(0, 3, 1, 2), g)["mse"]
                    for x, g in zip(evals, gen, strict=True)
                ]))
                peak, note = self._peak_mem(s.device)
                self.report["baseline"] = {
                    "psnr_vs_gt": gt_psnr, "mse_vs_gt": gt_mse,
                    "noise_floor": {"psnr": floor["psnr_vs_ref"], "mse": floor["mse_vs_ref"]},
                    "latency_ms_per_video": lat, "latency_ms_per_frame": lat / max(1, s.frames),
                    "peak_mem_bytes": peak, "peak_mem_note": note,
                    "frames": s.frames, "image_size": int((ad.cfg.get("image_size", 0) if hasattr(ad, "cfg") else 0) or 0),
                    "sample_steps": s.sample_steps,
                    "action_conditioned": s.action_conditioned, "action_dim": s.action_dim if s.action_conditioned else 0,
                    "lpips_vs_gt": None,
                }
                self.ev.emit("metric", name="baseline.psnr_vs_gt", value=round(gt_psnr, 3), msg="FP vs real future")
                self.ev.emit("metric", name="baseline.noise_floor.psnr", value=round(floor["psnr_vs_ref"], 3), msg="FP seed-to-seed")
                self.ev.emit("metric", name="baseline.latency_ms_per_video", value=round(lat, 2), msg="FP sampling")
                return gen

            ref = self._stage("eval_baseline", baseline)
            self._save()

        def calibrate() -> None:
            for i, inp in enumerate(calib_inputs):
                ad.calibrate(inp, seed=s.seed + i)

        self.report["recipes"] = {}
        for recipe in recipes:
            if not recipe.weight.enabled:
                continue
            algo = recipe.default_algorithm if s.algorithm in ("", "recipe") else s.algorithm
            rep: dict[str, Any] = {
                "recipe": recipe.name, "weight": recipe.weight.to_dict(), "act": recipe.act.to_dict(),
                "maturity": recipe.maturity, "algorithm": algo,
            }

            def on_layer(e: dict[str, Any]) -> None:
                if e["kind"] == "layer":
                    self.ev.emit("trace", **e)

            engine = FakeQuantEngine(ad.qmodel, recipe, progress=on_layer, algorithm=algo)

            def usable(mods: list[str]) -> list[Any]:
                return [u for u in engine.select_units(mods) if not policy.lookup(u.name, u.block).skip]

            modalities = [m for m in dict.fromkeys(recipe.modalities + s.also_quantize) if usable([m])]
            rep["skipped"] = sorted(
                u.name for u in engine.select_units(list(dict.fromkeys(recipe.modalities + s.also_quantize)))
                if policy.lookup(u.name, u.block).skip
            )

            if s.attribution and not s.quantize_only:
                assert ref is not None

                def attribution() -> list[dict[str, Any]]:
                    rows = []
                    for mod in modalities:
                        units = usable([mod])
                        engine.apply(units, engine.capture(units, calibrate, max_rows=256) if algo != "rtn" else None)
                        cur = self._generate(ad, evals, eval_inputs)
                        m = self._score(ref, cur, evals)
                        engine.restore()
                        engine.traces.clear()
                        rows.append({"modality": mod, "units": len(units), "psnr_vs_ref": m["psnr_vs_ref"],
                                     "mse_vs_ref": m["mse_vs_ref"], "psnr_vs_gt": m["psnr_vs_gt"]})
                        self.ev.emit("metric", name=f"{recipe.name}.attribution.{mod}.psnr_vs_ref",
                                     value=round(m["psnr_vs_ref"], 3), msg=f"{recipe.name} only-{mod}")
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
                def evaluate() -> dict[str, Any]:
                    assert ref is not None
                    if torch.cuda.is_available() and str(s.device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(s.device)
                    t0 = time.perf_counter()
                    cur = self._generate(ad, evals, eval_inputs)
                    if torch.cuda.is_available() and str(s.device).startswith("cuda"):
                        torch.cuda.synchronize(s.device)
                    lat = (time.perf_counter() - t0) / max(1, len(evals)) * 1e3
                    m = self._score(ref, cur, evals)
                    peak, note = self._peak_mem(s.device)
                    m.update(latency_ms_per_video=lat, latency_ms_per_frame=lat / max(1, s.frames),
                             peak_mem_bytes=peak, peak_mem_note=note,
                             lpips_vs_ref=self._lpips(ref, cur),
                             psnr_drop_db=(self.report["baseline"]["psnr_vs_gt"] - m["psnr_vs_gt"]) if s.eval_source else None,
                             mse_ratio_vs_gt=(m["mse_vs_gt"] / self.report["baseline"]["mse_vs_gt"])
                             if self.report["baseline"]["mse_vs_gt"] > 0 else None)
                    for k in ("psnr_vs_ref", "mse_vs_ref", "psnr_vs_gt", "psnr_drop_db"):
                        self.ev.emit("metric", name=f"{recipe.name}.{k}", value=m.get(k), msg=f"{recipe.name} {k}")
                    return m

                rep["eval"] = self._stage("eval_quantized", evaluate, msg=recipe.name)
                rep["summary"] = self._summarize(rep, self.report.get("baseline"))
            if s.native_serve and not s.quantize_only:
                from mqde.backend.continuous_server import smoke_and_load

                def deploy_native() -> dict[str, Any]:
                    return smoke_and_load(ad, self.kind, s.native_serve, model_name=Path(s.model_path).name,
                                          action_conditioned=s.action_conditioned, frames=s.frames)

                rep["deploy"] = {"native": self._stage("deploy_native", deploy_native, msg=recipe.name)}
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
            "component_mean_out_sqnr_db": {k: float(np.mean(v)) for k, v in by_comp.items()},
        }
        if base is None:
            return {**out, "psnr_vs_ref": None, "mse_vs_ref": None, "psnr_vs_gt": None,
                    "psnr_drop_db": None, "mse_ratio_vs_gt": None, "lpips_vs_ref": None,
                    "psnr_over_noise_floor": None, "below_noise_floor": None, "worst_stage": None}
        e, attr = rep["eval"], rep.get("attribution") or []
        nf = base["noise_floor"]["psnr"] or 0.0
        return {
            **out,
            "psnr_vs_ref": e["psnr_vs_ref"], "mse_vs_ref": e["mse_vs_ref"],
            "psnr_vs_gt": e["psnr_vs_gt"], "psnr_drop_db": e["psnr_drop_db"],
            "mse_ratio_vs_gt": e["mse_ratio_vs_gt"], "lpips_vs_ref": e["lpips_vs_ref"],
            "psnr_over_noise_floor": (e["psnr_vs_ref"] / nf) if nf else None,
            "below_noise_floor": bool(nf and e["psnr_vs_ref"] >= nf),
            "worst_stage": min(attr, key=lambda r: r["psnr_vs_ref"])["modality"] if attr else None,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a world-model / DiT quantization evaluation job")
    ap.add_argument("job", help="job YAML")
    ap.add_argument("--device", default=None)
    ap.add_argument("--recipes", default=None, help="override recipes, comma separated")
    ap.add_argument("--algorithm", default=None)
    ap.add_argument("--eval-samples", type=int, default=None)
    ap.add_argument("--calib-samples", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--sample-steps", type=int, default=None)
    ap.add_argument("--quantize-only", action="store_true")
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--job-id", default=None, help="assigned by mqde-server")
    a = ap.parse_args()
    spec = WorldJobSpec.from_yaml(a.job)
    for k, v in {"device": a.device, "algorithm": a.algorithm, "eval_samples": a.eval_samples,
                 "calib_samples": a.calib_samples, "frames": a.frames, "sample_steps": a.sample_steps,
                 "group_size_override": a.group_size}.items():
        if v is not None:
            setattr(spec, k, v)
    if a.recipes:
        spec.recipes = a.recipes.split(",")
    if a.quantize_only:
        spec.quantize_only = True
    job = WorldJob(spec, job_id=a.job_id)
    rep = job.run()
    print(json.dumps({"job_id": job.job_id, "report": str(job.out / "report.json"),
                      "recipes": {k: v["summary"] for k, v in rep["recipes"].items()}}, indent=2, default=str))


if __name__ == "__main__":
    main()
