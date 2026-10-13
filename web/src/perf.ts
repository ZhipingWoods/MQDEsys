import type { ConcurrencySweep, DeployJobReport, LLMJobReport, VllmDeploy } from "./api";

/** One served model with a concurrency sweep, from an LLM job's own serve stage or from a deploy job. */
export interface ServingRun {
  job: string;
  backend: "vllm" | "sglang";
  /** what was served: `W4A16 · awq` or, for a deploy job, `<parent>/<recipe> · <format>` */
  label: string;
  sweep: ConcurrencySweep;
  peak: number;
  saturation: number | null;
}

const BACKENDS = ["vllm", "sglang"] as const;

function push(out: ServingRun[], job: string, backend: "vllm" | "sglang", label: string, d: VllmDeploy | undefined) {
  const s = d?.sweep;
  if (!s || "error" in s || !s.points?.length) return;
  out.push({ job, backend, label, sweep: s, peak: s.peak?.output_tokens_per_s ?? 0, // "not-reached" / "single-level" name the top level that was measured, not a saturation point
    saturation: s.saturation && (s.saturation.reason === "throughput" || s.saturation.reason === "ttft") ? s.saturation.concurrency : null });
}

/** Every finished sweep in the given reports (LLM jobs: one per recipe x format x backend; deploy jobs: one per backend). */
export function servingRuns(reps: (LLMJobReport | DeployJobReport)[]): ServingRun[] {
  const out: ServingRun[] = [];
  for (const r of reps) {
    if (r.kind === "llm_deploy") {
      const a = r.artifact;
      const label = a ? `${a.job}/${a.recipe ?? a.type ?? "?"} · ${a.format}` : r.spec.name;
      for (const b of BACKENDS) push(out, r.job_id, b, label, r.deploy?.[b]);
    } else {
      for (const rr of Object.values(r.recipes ?? {}))
        for (const [fmt, ex] of Object.entries(rr.export ?? {}))
          for (const b of BACKENDS) push(out, r.job_id, b, `${rr.recipe} · ${fmt}`, ex[b]);
    }
  }
  return out;
}

/** The runs whose engines were configured alike enough for throughput to be compared: a note for the rest. */
export function engineNote(runs: ServingRun[]): string | null {
  return new Set(runs.map((r) => r.backend)).size > 1
    ? "vLLM 阶段用 enforce_eager，SGLang 用 CUDA graph：两个引擎的吞吐不可直接比较，只比较同一引擎内的曲线。"
    : null;
}
