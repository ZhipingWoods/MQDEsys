// 作业派生：把一个已提交作业的 job.yaml 读回新建作业向导的状态，用户只改想改的那一步（评估数据、设备、推理框架、
// 配方……），其余全部沿用。量化会重新运行：fake-quant 不落盘量化权重，但同样的配方、数据和种子得到同样的结果。
/* eslint-disable @typescript-eslint/no-explicit-any */

export type DeriveStep = "eval" | "device" | "deploy" | "recipe" | "all";
export const DERIVE_STEPS: { step: DeriveStep; label: string; suffix: string }[] = [
  { step: "eval", label: "换评估数据", suffix: "eval" },
  { step: "device", label: "换设备", suffix: "dev" },
  { step: "deploy", label: "换推理框架", suffix: "deploy" },
  { step: "recipe", label: "换配方", suffix: "recipe" },
  { step: "all", label: "复制全部参数", suffix: "copy" },
];

export type Rule = { match: string; bits?: number; skip?: boolean };

export interface Derived {
  domain: string;
  name: string;
  model: string;
  picked: string[];
  algorithm: string;
  groupSizeOverride: string;
  action: boolean;
  actStatic: string;
  kv: string;
  kvMethod: string;
  keepEnds: boolean;
  downBits: string;
  inheritedRules: Rule[]; // mixed-precision rules the wizard has no control for (e.g. a dynamic-quant skip list)
  nCalib: number;
  seqLen: number;
  dataset: string;
  nEval: number;
  windows: number;
  maxNewTokens: number;
  attribution: boolean;
  quantizeOnly: boolean;
  quantDevice: string;
  evalDevice: string;
  hw: string;
  deployBackend: "none" | "vllm" | "llamacpp" | "sglang";
  useVllm: boolean;
  vllmUtil: number;
  useSglang: boolean;
  sglangMem: number;
  useSweep: boolean;
  sweepLevels: string;
  sweepBudget: string;
  useLmEval: boolean;
  lmTasks: string;
  lmLimit: number;
  lmShots: number;
  gguf: string[];
  exportFmts: string[];
  sens: string;
  probe: boolean;
  probeExact: boolean;
}

const KEEP_ENDS: Rule[] = [{ match: "block:0", bits: 8 }, { match: "block:-1", bits: 8 }];
const DOWN = "mlp\\.down_proj$";

/** job.yaml (as GET /jobs/<id> returns it) -> wizard state. Missing fields fall back to `d` (the wizard defaults). */
export function deriveFromJob(job: any, step: DeriveStep, d: Derived): Derived {
  const q = job?.quant ?? {}, c = job?.calibration ?? {}, e = job?.evaluation ?? {}, dep = job?.deploy ?? {}, an = job?.analysis ?? {};
  const type = job?.model?.type ?? "vla";
  // a deploy job (B2) quantizes nothing; its artifact names the recipe that produced it
  const recipes: string[] = Array.isArray(q.recipe) ? q.recipe : q.recipe ? [q.recipe] : job?.artifact?.recipe ? [String(job.artifact.recipe).toLowerCase()] : d.picked;
  // First match wins, so the order is part of the meaning: the rules map onto the wizard's two controls only when
  // they are exactly what those controls write, in their order; anything else is carried over untouched.
  const all: Rule[] = [...(q.mixed_precision?.rules ?? [])];
  const isEnd = (r: Rule, i: number) => r.match === KEEP_ENDS[i]?.match && r.bits === 8 && !r.skip;
  const hasEnds = all.length >= 2 && isEnd(all[0], 0) && isEnd(all[1], 1);
  const rest = all.slice(hasEnds ? 2 : 0);
  const canon = rest.length === 0 || (rest.length === 1 && rest[0].match === DOWN && (rest[0].skip || rest[0].bits != null));
  const keepEnds = canon && hasEnds;
  const down = canon ? rest[0] : undefined;
  const rules: Rule[] = canon ? [] : all;
  const kv = q.kv ? (String(q.kv.format).startsWith("fp8") ? "fp8" : `int${q.kv.bits}`) : job?.quant && type === "llm" ? "" : d.kv;
  const device = dep.device === "cpu" ? "cpu" : "gpu";
  const suffix = DERIVE_STEPS.find((s) => s.step === step)?.suffix ?? "copy";
  return {
    ...d,
    domain: job?.domain ?? type,
    name: `${job?.metadata?.name ?? "job"}-${suffix}`,
    model: job?.model?.path ?? d.model,
    picked: recipes,
    algorithm: q.algorithm ?? "recipe",
    groupSizeOverride: q.group_size != null ? String(q.group_size) : "",
    action: (q.also_quantize ?? []).includes("action"),
    actStatic: q.act_static ?? "",
    kv,
    kvMethod: q.kv?.method ?? d.kvMethod,
    keepEnds,
    downBits: down ? (down.skip ? "fp" : String(down.bits ?? "")) : "",
    inheritedRules: rules,
    nCalib: c.n_samples ?? d.nCalib,
    seqLen: c.seq_len ?? e.seq_len ?? d.seqLen,
    dataset: c.dataset && type !== "llm" ? c.dataset : d.dataset,
    nEval: e.n_samples ?? d.nEval,
    windows: e.max_windows ?? d.windows,
    maxNewTokens: e.max_new_tokens ?? d.maxNewTokens,
    attribution: e.attribution ?? d.attribution,
    quantizeOnly: !!e.skip,
    quantDevice: job?.quant_device ?? device,
    evalDevice: e.device ?? device,
    hw: dep.hardware ?? d.hw,
    deployBackend: dep.vllm ? "vllm" : dep.sglang ? "sglang" : dep.llamacpp ? "llamacpp" : "none",
    useVllm: !!dep.vllm,
    vllmUtil: dep.vllm?.gpu_memory_utilization ?? d.vllmUtil,
    useSglang: !!dep.sglang,
    sglangMem: dep.sglang?.mem_fraction_static ?? d.sglangMem,
    useSweep: !!(dep.vllm?.sweep ?? dep.sglang?.sweep),
    sweepLevels: ((dep.vllm?.sweep ?? dep.sglang?.sweep)?.levels ?? d.sweepLevels.split(",")).join(","),
    sweepBudget: String((dep.vllm?.sweep ?? dep.sglang?.sweep)?.max_ttft_p99_ms ?? ""),
    useLmEval: !!dep.vllm?.lm_eval,
    lmTasks: (dep.vllm?.lm_eval?.tasks ?? d.lmTasks.split(",")).join(","),
    lmLimit: dep.vllm?.lm_eval?.limit ?? d.lmLimit,
    lmShots: dep.vllm?.lm_eval?.num_fewshot ?? d.lmShots,
    gguf: dep.llamacpp?.types ?? [],
    exportFmts: job?.export?.formats ?? [],
    sens: an.sensitivity ?? "",
    probe: !!an.logits_probe,
    probeExact: !!an.logits_probe?.exact,
  };
}
