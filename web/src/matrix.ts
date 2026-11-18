import type { LLMJobReport } from "./api";
import { FP_KEY, MatrixEntry } from "./components/ScoreMatrix";

/** One matrix entry per (job, recipe), plus an "unquantized model" entry per model-and-protocol group (it only has a PPL). */
export function matrixEntries(reps: LLMJobReport[]): MatrixEntry[] {
  const out: MatrixEntry[] = [];
  const haveFp = new Set<string>();
  for (const r of reps) {
    const model = r.spec.model_path.split("/").pop() ?? "";
    const windows = r.baseline?.windows ?? r.data?.eval?.windows;
    const seq = r.baseline?.seq_len ?? r.data?.eval?.seq_len;
    if (windows == null || seq == null) continue; // how it was scored is unknown, so there is nothing it could be compared with
    // a non-default evaluation corpus (T-1304 leftover: LLM jobs can now pick their own) is its own protocol --
    // lumping a job scored on someone's uploaded text together with the default wikitext-2 test group would be
    // a false comparison, so it only joins the default group when it used the default dataset
    const evalDataset = r.data?.eval?.dataset;
    const nonDefault = evalDataset && evalDataset !== "wikitext-2 test" ? ` · ${evalDataset}` : "";
    const protocol = `${windows} 窗口 × ${seq}${nonDefault}`;
    const g = `${model}|${protocol}`;
    if (r.baseline && !haveFp.has(g)) {
      haveFp.add(g);
      out.push({ key: FP_KEY, label: "原模型（未量化）", model, protocol, metrics: { ppl: r.baseline.ppl } });
    }
    for (const rr of Object.values(r.recipes ?? {})) {
      const exports = Object.values(rr.export ?? {});
      const serves = exports.flatMap((e) => (["vllm", "sglang"] as const).flatMap((b) => (e[b] ? [{ b, d: e[b]! }] : [])));
      const withTps = serves.filter((x) => x.d.load?.output_tokens_per_s != null);
      const bestTps = withTps.length ? withTps.reduce((a, b) => (b.d.load!.output_tokens_per_s! > a.d.load!.output_tokens_per_s! ? b : a)) : null;
      const metrics: MatrixEntry["metrics"] = { ppl: rr.eval.fake_quant_wo_kv.ppl };
      if (rr.eval.fake_quant) metrics.ppl_kv = rr.eval.fake_quant.ppl;
      const real = exports.find((e) => e.real_ppl != null)?.real_ppl;
      if (real != null) metrics.real_ppl = real;
      const served = serves.find((x) => x.d.served_ppl != null)?.d.served_ppl;
      if (served != null) metrics.served_ppl = served;
      if (bestTps) metrics.tps = bestTps.d.load!.output_tokens_per_s!;
      const bytes = exports.find((e) => e.bytes != null)?.bytes;
      if (bytes != null) metrics.size_mib = bytes / 2 ** 20;
      out.push({
        key: `${r.job_id}/${rr.recipe}`,
        label: `${r.job_id.slice(-6)} · ${rr.recipe}/${rr.algorithm}`,
        model,
        protocol,
        metrics,
        ...(bestTps ? { engine: bestTps.b } : {}),
      });
    }
  }
  return out;
}
