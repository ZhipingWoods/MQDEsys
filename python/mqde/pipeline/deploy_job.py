"""Deploy / evaluate job (B2 of 作业派生): serve a checkpoint a finished LLM job already exported -- on another backend,
another device, other evaluation settings -- without quantizing again.

The artifact is named, not given as a path: `artifact: {job, recipe, format}` -> `<runs>/<job>/export/<recipe>-<format>/`,
or `{job, format: gguf, type}` -> `<runs>/<job>/export/gguf/model-<type>.gguf`, the layout llm_job writes. The parent's
report supplies what the served numbers are compared with: its real-weight PPL (HF formats) or its llama-perplexity PPL
(GGUF). Nothing is written into the parent job's directory.

Stages: validate -> eval_prepare (tokenizer from the checkpoint, wikitext-2 test ids) -> deploy_vllm | deploy_sglang |
deploy_llamacpp -> report.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from mqde.backend.llamacpp import bench, perplexity
from mqde.data.text import eval_ids, wikitext_text
from mqde.pipeline.envinfo import device_name
from mqde.pipeline.events import EventSink
from mqde.pipeline.llm_job import LLMJob

HF_FORMATS = {"awq", "gptq", "compressed-tensors"}
JOB_ID = re.compile(r"[0-9A-Za-z_-]+")
RECIPE = re.compile(r"[0-9A-Za-z_]+")
GGUF_TYPE = re.compile(r"[0-9a-z_]+")


def load_tokenizer(path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


@dataclass
class DeployJobSpec:
    name: str
    model_path: str  # the original model (for the record and the tokenizer of a GGUF artifact)
    artifact: dict[str, Any]
    eval_seq_len: int = 512
    eval_windows: int = 40
    vllm: dict[str, Any] | None = None
    sglang: dict[str, Any] | None = None
    llamacpp: dict[str, Any] | None = None  # {gpu_layers, serve: {...}}; GGUF artifacts only
    hardware: str = ""
    device: str = "cuda:0"
    output_dir: str = "runs"
    parent_job: str | None = None
    # what LLMJob's serving stages read
    llamacpp_serve: dict[str, Any] | None = field(default=None)
    llamacpp_gpu_layers: int | None = None

    def __post_init__(self) -> None:
        if self.llamacpp is not None:
            self.llamacpp_serve = self.llamacpp.get("serve", {"concurrency": 4, "requests": 16, "prompt_len": 256, "n_predict": 64})
            self.llamacpp_gpu_layers = self.llamacpp.get("gpu_layers")

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeployJobSpec:
        d = yaml.safe_load(Path(path).read_text())
        e, dep = d.get("evaluation", {}), d.get("deploy", {})
        return cls(
            name=d.get("metadata", {}).get("name", Path(path).stem),
            model_path=d["model"]["path"],
            artifact=dict(d["artifact"]),
            eval_seq_len=int(e.get("seq_len", 512)),
            eval_windows=int(e.get("max_windows", 40)),
            vllm=dep.get("vllm"),
            sglang=dep.get("sglang"),
            llamacpp=dep.get("llamacpp"),
            hardware=str(dep.get("hardware", "")),
            device="cpu" if dep.get("device") == "cpu" else "cuda:0",
            output_dir=d.get("output_dir", "runs"),
            parent_job=d.get("metadata", {}).get("parent_job"),
        )


class DeployJob(LLMJob):
    spec: DeployJobSpec  # type: ignore[assignment]  # narrower than LLMJob's: this job has an artifact and no recipes

    def __init__(self, spec: DeployJobSpec, job_id: str | None = None) -> None:
        self.spec = spec
        self.job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + __import__("uuid").uuid4().hex[:6]
        self.out = Path(spec.output_dir) / self.job_id
        self.out.mkdir(parents=True, exist_ok=True)
        self.ev = EventSink(self.job_id, self.out / "events.jsonl")
        self.report: dict[str, Any] = {"job_id": self.job_id, "kind": "llm_deploy", "spec": spec.__dict__, "stages": []}

    def _serve_dir(self, ckpt: Path) -> Path:
        d = self.out / "serve"
        d.mkdir(exist_ok=True)
        return d

    def _gguf_dir(self) -> Path:
        return self._serve_dir(Path())

    def run(self) -> dict[str, Any]:
        self.ev.emit("job_state", state="RUNNING", msg=f"job {self.job_id}")
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

    def _resolve(self) -> dict[str, Any]:
        s = self.spec
        return resolve_artifact(s.artifact, s.output_dir, vllm=s.vllm, sglang=s.sglang, llamacpp=s.llamacpp,
                                check_backend=True)

    def _run(self) -> None:
        s = self.spec
        self.report["env"] = {"torch": torch.__version__, "device": s.device, "hardware": s.hardware,
                              "device_name": device_name(s.device)}
        art = self.report["artifact"] = self._stage("validate", self._resolve, json.dumps(s.artifact))
        # PPL depends on the context length and on which windows are scored: only the same protocol as the parent's makes
        # "served vs the parent's real weights" a like-for-like number (a real run with 256 x 4 against 512 x 40 said +26%)
        pe = art["parent_eval"]
        art["ppl_comparable"] = bool(pe and pe["windows"] == s.eval_windows and pe["seq_len"] == s.eval_seq_len)
        if not art["ppl_comparable"]:
            art["ppl_note"] = (f"this job scores {s.eval_windows} x {s.eval_seq_len} wikitext-2 windows, "
                               f"the parent {pe['windows']} x {pe['seq_len']}: "
                               "PPL is not comparable across protocols, so no 'vs parent' difference is given"
                               if pe else
                               "the parent report did not record its evaluation protocol: PPL against it is not assumed comparable")
        parent_real = art.get("parent_real_ppl") if art["ppl_comparable"] else None
        tok_dir = s.model_path if art["format"] == "gguf" else art["path"]
        ids = self._stage("eval_prepare", lambda: eval_ids(load_tokenizer(tok_dir)))
        self.report["data"] = {"eval": {"dataset": "wikitext-2 test", "windows": s.eval_windows, "seq_len": s.eval_seq_len}}
        self.report["deploy"] = {}
        ckpt = Path(art["path"])
        if s.vllm is not None:
            self.report["deploy"]["vllm"] = self._stage("deploy_vllm", lambda: self._vllm_serve(ckpt, ids, parent_real), ckpt.name)
            self._save()
        if s.sglang is not None:
            self.report["deploy"]["sglang"] = self._stage("deploy_sglang", lambda: self._sglang_serve(ckpt, ids, parent_real), ckpt.name)
            self._save()
        if s.llamacpp is not None:
            self.report["deploy"]["llamacpp"] = self._stage("deploy_llamacpp", lambda: self._gguf(ckpt, art), art["type"])
        self._stage("report", self._save)

    def _gguf(self, path: Path, art: dict[str, Any]) -> dict[str, Any]:
        s, ngl = self.spec, self.spec.llamacpp_gpu_layers
        text = self._serve_dir(path) / "wiki.test.txt"
        text.write_text(wikitext_text("test"))
        threads = torch.get_num_threads()
        p = perplexity(path, text, ctx=s.eval_seq_len, chunks=s.eval_windows, threads=threads, gpu_layers=ngl)
        b = bench(path, pp=s.eval_seq_len, tg=64, threads=threads, reps=2, gpu_layers=ngl)
        self.ev.emit("metric", name=f"llamacpp.{art['type']}.ppl", value=p["ppl"], msg=f"llama.cpp {art['type']}")
        out = {**p, **{k: v for k, v in b.items() if k.endswith("per_s")}, "backend": b.get("backend"), "gpu_layers": ngl,
               "threads": threads, "parent_ppl": art.get("parent_ppl"),
               "ppl_vs_parent_pct": (100.0 * (p["ppl"] / art["parent_ppl"] - 1.0)
                                     if art.get("parent_ppl") and art.get("ppl_comparable") else None)}
        if s.llamacpp_serve:
            out["serve"] = self._llamacpp_serve(path, art["type"], threads)
        return out


def resolve_artifact(a: dict[str, Any], output_dir: str | Path, *, vllm: Any = None, sglang: Any = None,
                     llamacpp: Any = None, check_backend: bool = False) -> dict[str, Any]:
    """Resolve `{job, recipe, format}` / `{job, format: gguf, type}` to an exported path and its parent numbers.

    `vllm`/`sglang`/`llamacpp` are only inspected when `check_backend` (DeployJob's case: it must serve the
    artifact); an evaluate-only job just needs the path, so it leaves the defaults alone.
    """
    job, recipe, fmt, ty = str(a.get("job", "")), str(a.get("recipe") or ""), str(a.get("format", "")), a.get("type")
    if not JOB_ID.fullmatch(job):
        raise ValueError(f"artifact.job: bad job id '{job}'")
    if fmt not in HF_FORMATS | {"gguf"}:
        raise ValueError(f"artifact.format '{fmt}': one of {sorted(HF_FORMATS)} or gguf")
    parent = Path(output_dir) / job
    rep_path = parent / "report.json"
    if not rep_path.is_file():
        raise ValueError(f"artifact.job: job {job} not found (no {rep_path})")
    rep = json.loads(rep_path.read_text())
    pe = ((rep.get("data") or {}).get("eval")) or {}
    parent_eval = {"windows": pe["windows"], "seq_len": pe["seq_len"]} if "windows" in pe and "seq_len" in pe else None
    out: dict[str, Any] = {"job": job, "format": fmt, "parent_baseline_ppl": (rep.get("baseline") or {}).get("ppl"),
                           "parent_model_path": (rep.get("spec") or {}).get("model_path"), "parent_eval": parent_eval}
    if fmt == "gguf":
        if not (isinstance(ty, str) and GGUF_TYPE.fullmatch(ty)):
            raise ValueError(f"artifact.type: bad GGUF type '{ty}'")
        res = ((rep.get("llamacpp") or {}).get("results") or {}).get(ty)
        path = parent / "export" / "gguf" / f"model-{ty}.gguf"
        if res is None or not path.is_file():
            raise ValueError(f"artifact: job {job} has no GGUF {ty} (llama.cpp types it produced: "
                             f"{sorted((rep.get('llamacpp') or {}).get('results') or {})})")
        out.update(type=ty, path=str(path), bytes=path.stat().st_size, parent_ppl=res.get("ppl"))
    else:
        if not RECIPE.fullmatch(recipe):
            raise ValueError(f"artifact.recipe: bad recipe name '{recipe}'")
        r = (rep.get("recipes") or {}).get(recipe)
        if r is None:
            raise ValueError(f"artifact: job {job} did not quantize {recipe} (it has {sorted(rep.get('recipes') or {})})")
        ex = (r.get("export") or {}).get(fmt)
        if not ex or ex.get("status") != "exported":
            raise ValueError(f"artifact: {recipe} of job {job} has no {fmt} export (status: {(ex or {}).get('status', 'not requested')}"
                             f"{' -- unsupported' if (ex or {}).get('status') == 'unsupported' else ''})")
        path = parent / "export" / f"{recipe.lower()}-{fmt}"
        if not (path / "config.json").is_file():
            raise ValueError(f"artifact: {path} is gone (was it cleaned up?)")
        out.update(recipe=recipe, algorithm=r.get("algorithm"), path=str(path), bytes=ex.get("bytes"),
                   parent_real_ppl=ex.get("real_ppl"), parent_fake_ppl=(r.get("summary") or {}).get("ppl"))
    if check_backend:
        if fmt == "gguf" and (vllm is not None or sglang is not None):
            raise ValueError("a GGUF artifact is served by llama.cpp, not vLLM / SGLang")
        if fmt != "gguf" and llamacpp is not None:
            raise ValueError(f"llama.cpp serves GGUF files; a {fmt} checkpoint goes to vLLM or SGLang")
        if vllm is None and sglang is None and llamacpp is None:
            raise ValueError("no serving backend: set deploy.vllm, deploy.sglang or deploy.llamacpp")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve / evaluate a checkpoint exported by a finished LLM job")
    ap.add_argument("job", help="job YAML")
    ap.add_argument("--device", default=None)
    ap.add_argument("--job-id", default=None, help="assigned by mqde-server")
    a = ap.parse_args()
    spec = DeployJobSpec.from_yaml(a.job)
    if a.device:
        spec.device = a.device
    job = DeployJob(spec, job_id=a.job_id)
    rep = job.run()
    print(json.dumps({"job_id": job.job_id, "report": str(job.out / "report.json"),
                      "deploy": {k: v.get("status") for k, v in rep["deploy"].items()}}))


if __name__ == "__main__":
    main()
