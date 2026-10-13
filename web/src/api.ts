// Typed client for mqde-server (/api/v1). Shapes mirror the C++ to_json() functions and the
// Python JobReport; they will be generated from api/openapi.yaml once it exists (T-0007).

export type Bytes = number;

export interface QuantSpecJson {
  format: string;
  bits?: number;
  symmetric?: boolean;
  granularity?: string;
  group_size?: number;
  calib?: string;
}

export interface Recipe {
  id: string;
  name: string;
  description: string;
  maturity: "production" | "experimental";
  default_algorithm: string;
  modalities: string[];
  weight: QuantSpecJson;
  act: QuantSpecJson;
  kv: QuantSpecJson;
  summary: string;
}

export interface HardwareProfile {
  id: string;
  vendor: string;
  name: string;
  arch: string;
  mem_bytes: Bytes;
  mem_bw_gbps: number;
  peak_tflops: Record<string, number>;
  native_formats: string[];
  source?: string;
}

export interface CurvePoint {
  x: number;
  peak_bytes: Bytes;
  kv_bytes: Bytes;
  feasible: boolean;
}

export interface ComponentMemory {
  name: string;
  modality: string;
  params: number;
  original_bytes: Bytes;
  weight_bytes: Bytes;
  qmeta_bytes: Bytes;
  quantized_params: number;
  quantized: boolean;
}

export interface Estimate {
  recipe: string;
  recipe_id: string;
  hardware: string;
  context_tokens: number;
  components: ComponentMemory[];
  breakdown: { weights: Bytes; qmeta: Bytes; kv_cache: Bytes; activation: Bytes; runtime: Bytes };
  original_weights_bytes: Bytes;
  kv_bytes_per_token: Bytes;
  state_bytes_per_seq: Bytes;
  peak_bytes: Bytes;
  capacity_bytes: Bytes;
  feasible: boolean;
  max_concurrency: number;
  reasons: string[];
  perf: { flops_per_step: number; bytes_per_step: number; latency_floor_ms: number; compute_format: string };
  assumptions: string[];
  confidence: string;
  recipe_detail: Recipe;
  curves: { concurrency: CurvePoint[]; images: CurvePoint[] };
}

export interface RuntimeParams {
  concurrency: number;
  images_per_step: number;
  text_tokens: number;
  image_resolution: number;
  gpu_memory_utilization: number;
  extra_modalities: string[];
}

export interface MatrixCell {
  hardware: string;
  hardware_name: string;
  recipe: string;
  feasible: boolean;
  peak_bytes: Bytes;
  capacity_bytes: Bytes;
  max_concurrency: number;
  latency_floor_ms: number;
}

export interface Device {
  kind: "cpu" | "gpu" | "npu" | "other";
  vendor?: string;
  index: number;
  name: string;
  arch: string;
  mem_total_bytes: Bytes;
  mem_free_bytes: Bytes;
  compute_units: number;
  driver_version: string;
  profile?: string;
}

export interface BackendInfo {
  id: string;
  activation_model: "per_seq" | "batched";
  max_batched_tokens: number;
  runtime_overhead_bytes: number;
  allocator_factor: number;
}

export interface HalPlugin {
  name: string;
  vendor: string;
  version: string;
  status: "loaded" | "inactive" | "rejected" | "disabled" | "unknown";
  devices: number;
  reason: string;
  has_profile: boolean;
}

export interface ProfileResult {
  plugin: string;
  device: number;
  kind: string;
  dtype: string;
  value: number;
  unit: string;
  seconds: number;
}

export interface JobListItem {
  job_id: string;
  name?: string;
  state: string;
  stage?: string | null;
  recipes?: string[];
  model?: string;
  gpu?: number;
  kind?: "vla" | "llm" | "vlm" | "llm_deploy";
  device?: string;
  device_name?: string; // measured device from the report (GPU index is unknown after a server restart)
  parent_job?: string; // 作业派生: the job this one was derived from
  started_ts?: number;
  last_ts?: number;
  n_events: number;
}

/** Equal-width histogram over [min, max] with exact totals (see mqde.quant.engine.histogram_summary). */
export interface Histogram {
  min: number;
  max: number;
  mean: number;
  std: number;
  n: number;
  counts: number[];
}

export interface LayerTrace {
  name: string;
  component: string;
  modality: string;
  block: number;
  shape: [number, number];
  w_sqnr_db: number;
  w_rel_l2: number;
  w_cosine: number;
  out_sqnr_db: number | null;
  out_rel_l2: number | null;
  out_cosine: number | null;
  act_absmax: number | null;
  act_kurtosis: number | null;
  act_outlier_ratio: number | null;
  w_kurtosis: number | null;
  w_hist?: Histogram | null;
  act_hist?: Histogram | null;
  w_scale_summary?: { n: number; min: number; p50: number; p99: number; max: number; spread: number | null } | null;
  algorithm?: string;
  algo_info?: { bits?: number; act?: string } & Record<string, unknown>;
}

export interface Score {
  action_rel_l2: number;
  action_cos: number;
  action_max_abs: number;
  vl_cos: number;
  vl_sqnr_db: number | null;
  gt_mae: number | null;
  latency_ms_per_step_fake?: number;
  per_sample?: Record<string, number | null>[];
}

export interface RecipeReport {
  recipe: string;
  weight: QuantSpecJson;
  act: QuantSpecJson;
  maturity: string;
  algorithm?: string;
  attribution?: ({ modality: string; units: number } & Score)[];
  layers: LayerTrace[];
  eval: Score;
  summary: {
    worst_layers: { name: string; out_sqnr_db: number; act_kurtosis: number | null }[];
    component_mean_out_sqnr_db: Record<string, number>;
    worst_stage: string | null;
    action_rel_l2_over_noise_floor: number | null;
  };
}

export interface PplResult {
  ppl: number;
  tokens: number;
  windows: number;
  seq_len: number;
  seconds: number;
}

/** report.json of mqde.pipeline.llm_job (kind === "llm"). */
export interface LLMRecipeReport {
  recipe: string;
  algorithm: string;
  weight: QuantSpecJson;
  act: QuantSpecJson;
  layers: LayerTrace[];
  skipped: string[];
  quantize_seconds: number;
  kv?: QuantSpecJson & { method: string; granularity: string };
  eval: { fake_quant_wo_kv: PplResult; fake_quant?: PplResult };
  summary: { ppl: number; ppl_delta_pct: number; kv_delta_pct: number | null; worst_layers: { name: string; out_sqnr_db: number }[] };
  export?: Record<string, LLMExport>;
  sensitivity?: Sensitivity;
  logits?: LogitsResult;
}

export interface Sensitivity {
  by: "block" | "type";
  metric: string;
  metric_fp: number;
  metric_quantized: number;
  groups: { group: string; layers: number; params: number; metric: number; recovered: number; avg_bits: number }[];
  ranking: string[];
  seconds: number;
  suggestion: {
    rules: { match: string; bits?: number; skip?: boolean }[];
    groups: string[];
    predicted_recovered: number;
    predicted_metric: number;
    degradation: number;
    extra_weight_bytes: number;
    bits: number;
    note: string;
  };
}

export interface LLMExport {
  status: "exported" | "unsupported";
  problems?: { layer: string; reason: string }[];
  n_problems?: number;
  out_dir?: string;
  bytes?: number;
  quantized?: string[];
  kept_fp?: string[];
  notes?: string[];
  quantization_config?: Record<string, unknown>;
  real_check?: { codes_identical: boolean; code_mismatch: number; zero_mismatch: number; w_max_abs: number; w_rel_l2_mean: number; layers: number };
  real_ppl?: number;
  real_vs_fake_ppl_pct?: number;
  vllm?: VllmDeploy;
  sglang?: VllmDeploy; // same shape: SGLang's engine info just has no quant_kernel / max_concurrency fields
}

/** T-R12: one served model under several concurrency levels (eval/loadtest.py sweep()). */
export interface SweepPoint {
  concurrency: number; ok: number; failed: number; output_tokens_per_s?: number;
  ttft_ms?: { p50?: number; p90?: number; p99?: number }; tpot_ms?: { p50?: number; p90?: number; p99?: number };
  itl_ms?: { p50?: number; p90?: number; p99?: number };
}
export interface ConcurrencySweep {
  points: SweepPoint[]; peak: SweepPoint | null; max_ttft_p99_ms?: number;
  saturation: { concurrency: number; reason: "throughput" | "ttft" | "single-level" | "not-reached"; next_gain_pct: number | null } | null;
}

/** S11 + S12 on vLLM or SGLang: serve the exported checkpoint, load-test it, and score it with its own prompt logprobs. */
export interface VllmDeploy {
  status: "succeeded" | "degraded" | "failed" | "skipped";
  reason?: string;
  error?: string;
  diagnosis?: string | null;
  startup_s?: number;
  engine?: {
    kv_cache_tokens?: number;
    max_concurrency_at_len?: number;
    max_concurrency_len?: number;
    weights_gib?: number;
    kv_cache_gib?: number;
    quant_kernel?: string;
    init_s?: number;
    weight_load_s?: number;
    avail_after_pool_gib?: number;
    cuda_graph_s?: number;
  };
  smoke?: { ok: boolean; text?: string };
  sweep?: ConcurrencySweep | { error: string };
  load?: {
    ok: number;
    failed: number;
    spec: { concurrency: number; requests: number; prompt_len: number; n_predict: number };
    ttft_ms?: { p50?: number; p90?: number; p99?: number };
    tpot_ms?: { p50?: number; p90?: number; p99?: number };
    output_tokens_per_s?: number;
    requests_per_s?: number;
  };
  served_ppl?: number;
  served_windows?: number;
  served_vs_real_ppl_pct?: number;
  lm_eval?: LmEval;
}

/** lm-evaluation-harness run through its official CLI against the served model (ADR-0002). */
export interface LmEval {
  status: "succeeded" | "failed" | "skipped";
  reason?: string;
  error?: string;
  version?: string | null;
  limit?: number | null;
  num_fewshot?: number | null;
  tasks?: Record<
    string,
    {
      alias?: string;
      samples?: { original?: number; effective?: number };
      metrics: Record<string, { metric: string; filter: string; value: number; stderr: number | null }>;
    }
  >;
}

/** Output-distribution comparison against the unquantized model (eval/logits_probe.py). */
export interface LogitsResult {
  kl: number;
  top1_agreement: number;
  ref_top1_prob_drop: number;
  positions: number;
  windows: number;
  exact: boolean;
  k: number;
}

/** report.json of mqde.pipeline.vlm_job (kind === "vlm"). */
export interface VLMExample { question: string; answer: string; truth: string | null }
export interface VLMRecipeReport {
  recipe: string;
  algorithm: string;
  weight: QuantSpecJson;
  act: QuantSpecJson;
  layers: LayerTrace[];
  attribution?: { modality: string; units: number; kl: number; top1_agreement: number; answer_ppl: number }[];
  eval: {
    kl: number; top1_agreement: number; ref_top1_prob_drop: number; positions: number; answer_ppl: number;
    generation_exact_match: number; generation_prefix_match: number; gt_accuracy: number | null; examples: VLMExample[];
  };
  summary: {
    kl: number | null; top1_agreement: number | null; answer_ppl: number | null; answer_ppl_delta_pct: number | null;
    generation_exact_match: number | null; generation_prefix_match: number | null; gt_accuracy: number | null; gt_accuracy_delta: number | null;
    suggestion?: { rules: { match: string; skip: boolean }[]; layers: { name: string; component: string; out_sqnr_db: number; median_db: number }[]; extra_weight_bytes: number; basis: string };
    worst_stage: string | null; worst_layers: { name: string; out_sqnr_db: number; act_kurtosis: number }[];
    component_mean_out_sqnr_db: Record<string, number>;
  };
}
export interface VLMJobReport {
  job_id: string;
  kind: "vlm";
  error?: string;
  verdict?: string;
  spec: { name: string; model_path: string; recipes: string[]; algorithm: string; device: string; quantize_only?: boolean; max_new_tokens: number };
  stages: { stage: string; status: string; seconds?: number; error?: string }[];
  model?: { family: string; n_linears: number; components: Record<string, { modality: string; linears: number; quantizable: number; blocks: number }> };
  data?: { calib: { source: string; n: number }; eval: { source: string; n: number; has_gt: boolean; questions: string[] } };
  baseline?: { answer_ppl: number; n_samples: number; n_answer_tokens: number; gen_seconds_per_sample: number; gt_samples: number; gt_accuracy: number | null; examples: VLMExample[] };
  recipes?: Record<string, VLMRecipeReport>;
}

/** POST /models:gguf: a GGUF file's header (metadata + tensor table), read without its payload (T-R05). */
export interface GgufSummary {
  path: string; version: number; alignment: number; file_bytes: number; data_offset: number; architecture: string; name: string;
  block_count?: number; n_tensors: number; n_params: number; tensor_bytes: number;
  metadata: Record<string, unknown>;
  quant_types: Record<string, { tensors: number; params: number; bytes: number }>;
  blocks: { index: number; tensors: number; params: number; bytes: number; types: Record<string, number> }[];
  global: string[];
  tensors?: { name: string; type: string; shape: number[]; numel: number; nbytes: number; offset: number }[];
}

/** POST /recommend: one ranked deployment option (planner::recommend + the accuracy risk from advice.cpp). */
export interface Advice {
  recipe: string; algorithm: string; backend: string; hardware: string; maturity: string; feasible: boolean;
  deploy_verdict: string; peak_bytes: number; capacity_bytes: number; max_concurrency: number; export_format: string; reasons: string[];
  risk: "low" | "medium" | "high"; risk_source: "measured" | "prior";
  measured?: { job_id: string; kind: string; algorithm: string; metric: string; value: number };
}
export type AdviceObjective = "accuracy" | "balanced" | "throughput";

/** report.json of mqde.pipeline.deploy_job (kind === "llm_deploy"): an existing export served again, no quantization. */
export interface DeployJobReport {
  job_id: string;
  kind: "llm_deploy";
  verdict?: string;
  error?: string;
  spec: { name: string; model_path: string; artifact: { job: string; recipe?: string; format: string; type?: string }; eval_seq_len: number; eval_windows: number; device: string };
  stages: { stage: string; status: string; seconds?: number; error?: string }[];
  artifact?: {
    job: string; format: string; path: string; bytes?: number | null; recipe?: string; algorithm?: string; type?: string;
    parent_real_ppl?: number | null; parent_fake_ppl?: number | null; parent_baseline_ppl?: number | null; parent_ppl?: number | null;
    /** the parent's evaluation protocol; PPL is only comparable when this job's is the same */
    parent_eval?: { windows: number; seq_len: number } | null; ppl_comparable?: boolean; ppl_note?: string;
  };
  data?: { eval: { dataset: string; windows: number; seq_len: number } };
  deploy?: {
    vllm?: VllmDeploy;
    sglang?: VllmDeploy;
    llamacpp?: { ppl: number; parent_ppl?: number | null; ppl_vs_parent_pct?: number | null; backend?: string; gpu_layers?: number | null; threads: number; serve?: LlamaServe; [k: string]: unknown };
  };
  env?: { device_name?: string };
}

export interface LLMJobReport {
  job_id: string;
  kind: "llm";
  error?: string;
  spec: { name: string; model_path: string; recipes: string[]; algorithm: string; act_static: string | null; kv_method: string; device: string; mixed_precision: unknown };
  stages: { stage: string; status: string; seconds?: number; error?: string }[];
  model?: { path: string; blocks: number; linears: number };
  data?: { calib: { dataset: string; n: number; seq_len: number }; eval: { dataset: string; windows: number; seq_len: number } };
  baseline?: PplResult;
  recipes?: Record<string, LLMRecipeReport>;
  env?: { torch: string; torch_threads: number; device?: string; quant_device?: string; eval_device?: string; quant_device_name?: string; eval_device_name?: string; hardware?: string };
  llamacpp?: LlamaCppDeploy;
  baseline_lm_eval?: LmEval;
  logits_probe?: { windows: number; seq_len: number; k: number; exact: boolean; reference_mib: number };
}

export interface LlamaCppRun {
  type: string;
  bytes: number;
  ppl: number;
  ppl_err: number;
  tensor_types?: Record<string, number>;
  fallback?: string[];
  serve?: LlamaServe;
  [k: string]: unknown; // pp<N>_tokens_per_s / tg<N>_tokens_per_s
}

type Pcts = { p50?: number; p90?: number; p99?: number };

/** llama-server stage: startup, one chat smoke request and a closed-loop streaming load test. */
export interface LlamaServe {
  status: "succeeded" | "degraded" | "failed" | "skipped";
  reason?: string;
  error?: string;
  diagnosis?: string | null;
  startup_s?: number;
  smoke?: { ok: boolean; text?: string };
  load?: {
    concurrency: number;
    requests: number;
    ok: number;
    ttft_ms?: Pcts;
    tpot_ms?: Pcts;
    e2e_ms?: Pcts;
    output_tokens_per_s?: number;
    requests_per_s?: number;
  };
}

export interface LlamaCppDeploy {
  protocol: string;
  threads: number;
  gpu_layers?: number | null;
  backend?: string | null;
  results: Record<string, LlamaCppRun>;
}

/** tokens/s entries of a llama-bench result, e.g. {pp: 631, tg: 18.5}. */
export function llamaSpeed(r: LlamaCppRun): { pp?: number; tg?: number } {
  const pick = (k: string) => Object.entries(r).find(([n]) => n.startsWith(k) && n.endsWith("_tokens_per_s"))?.[1] as number | undefined;
  return { pp: pick("pp"), tg: pick("tg") };
}

export interface JobReport {
  job_id: string;
  kind?: string;
  verdict?: string;
  error?: string;
  spec: Record<string, unknown> & { name: string; recipes: string[]; model_path: string; hardware: string };
  env?: Record<string, string | null>;
  stages: { stage: string; status: string; seconds?: number; error?: string }[];
  preflight?: { model: Record<string, unknown>; estimates: Record<string, Record<string, Estimate>> };
  model?: { blocks: Record<string, { modality: string; linears: number; quantizable: number; blocks: number }>; n_linears: number };
  data?: { calib: { source: string; n: number }; eval: { source: string; n: number; has_gt: boolean; tasks: string[]; views: number } };
  baseline?: {
    latency_ms_per_step: number;
    peak_mem_bytes: Bytes;
    gt_mae: number | null;
    noise_floor: Score;
    context_tokens: number;
  };
  recipes?: Record<string, RecipeReport>;
}

export interface JobDetail {
  job_id: string;
  state: string;
  stage?: string | null;
  gpu?: number;
  exit_code?: number;
  report: JobReport | null;
  job?: Record<string, unknown>;
}

export interface JobEvent {
  job_id: string;
  seq: number;
  ts: number;
  type: "job_state" | "stage_started" | "stage_finished" | "log" | "metric" | "trace" | "progress";
  stage: string | null;
  msg?: string;
  state?: string;
  status?: string;
  seconds?: number;
  name?: string;
  value?: number | null;
  level?: string;
  error?: string;
  index?: number;
  total?: number;
  trace?: LayerTrace;
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/v1${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.message ?? `${res.status} ${res.statusText}`);
  return data as T;
}

// ---- inference backend environments (vLLM / SGLang / llama.cpp): install and update ----
export type BackendId = "vllm" | "sglang" | "llamacpp";
export interface BackendEnv {
  id: BackendId;
  installed: boolean;
  version: string | null;
  path: string | null;
  source: "env" | "managed" | "path" | "legacy" | null;
  managed: boolean;
  envs_dir: string;
  latest?: string | null; // present once the newest version has been looked up
  update_available?: boolean;
  check_error?: string;
  source_build?: boolean; // llama.cpp built from 3rdpart/llama.cpp (never to be replaced by a release download)
}
export interface BackendInstall {
  id: string;
  backend: BackendId;
  version: string;
  state: "running" | "succeeded" | "failed";
  exit_code: number;
  started_at: number;
  finished_at: number;
  log_tail: string[];
}

// ---- deployment pre-evaluation (llmfit-style) ----
export interface FitSystem {
  name: string; backend: string; gpu_count: number; vram_gb: number; vram_free_gb: number; ram_gb: number;
  mem_bw_gbps: number; peak_tflops_bf16: number; unified_memory: boolean; native_formats: string[];
}
export type FitLevel = "perfect" | "good" | "marginal" | "too_tight";
export interface ModelFit {
  model: string; family: string; modality: string; params_b: number; active_params_b: number; is_moe: boolean;
  best_quant: string; fitting_quants: string[]; run_mode: string; fit_level: FitLevel;
  moe_tier: 0 | 1 | 2; moe_vram_pressure: number; moe_offloaded_gb: number;
  memory_required_gb: number; memory_available_gb: number; utilization_pct: number;
  estimated_tps: number; speed_unit: string; prefill_tps: number | null; ttft_ms: number | null; context: number;
  /** estimated_tps is an unreachable roofline ceiling (launch overhead not modeled), not a prediction. */
  speed_is_upper_bound: boolean;
  estimate_confidence: string; score: number;
  runtime: string; installed: boolean; installed_providers: string[]; usable_context: number; disk_size_gb: number;
  supports_tp: number[]; verify_command: string; license: string; capability_ids: string[];
  hf_downloads: number; hf_likes: number;
  fit_label: string; run_mode_label: string; estimate_confidence_label: string;
  estimate_basis: { method: string; gpu_bandwidth_gbps: number; ddr_bandwidth_gbps: number; efficiency: number; assumed_context: number; local_calibration: number; };
  formula_tps: number;
  measured_tps: { tok_s: number; ttft_ms: number | null; sample_count: number; hardware_label: string; source: "local" | "community" } | null;
  score_components: { quality: number; speed: number; fit: number; context: number };
  notes: string[]; detail?: Estimate; hardware?: string; system?: FitSystem;
}
export interface FitQuery {
  hardware: string; gpu_count?: number; ram_gb?: number; vram_gb?: number; mem_bw_gbps?: number;
  use_case?: string; context?: number; concurrency?: number; quant?: string; sort?: string; models?: string[];
  /** "local" = scan the model library; "catalog" = HF catalog entries (evaluated without weights). */
  source?: "local" | "catalog"; search?: string; provider?: string; limit?: number;
}
export interface CatalogModel {
  name: string; provider: string; parameter_count: string; parameters_raw: number; quantization: string; format: string;
  context_length: number; architecture: string; pipeline_tag: string; license: string; release_date: string;
  capabilities: string[]; hf_downloads: number; hf_likes: number; demoted: boolean; demote_reason: string;
  /** T-1303: deterministic two-level grouping for a dropdown-style browser ("LLM" / "Qwen2.5"). */
  category: string; family_group: string;
}
export interface HfHit { name: string; provider: string; pipeline_tag: string; hf_downloads: number; hf_likes: number; parameters_raw: number }
export interface DownloadStatus {
  id: string; repo: string; group: string; dir: string; state: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  error: string; bytes_done: number; bytes_total: number; files_done: number; files_total: number;
}
export interface InstalledIndex { models: { name: string; canonical: string; source: string }[]; active_sources: string[]; elapsed_ms: number }
export interface ClaimBounds {
  conservative_tps: number; optimistic_tps: number; holds: boolean; best_quant: string; max_context_at_min_tps: number;
  memory_gb: number; efficiency_used: number; speed_unit: string; basis: "formula" | "measured";
}
// ---- model library + netron-style graph ----
export interface LibraryModel {
  name: string; path: string; family?: string; modality?: string; model_type?: string; total_params?: number;
  logical_params?: number; active_params?: number; total_bytes?: number; error?: string;
  checkpoint_quant?: { method: string; algo: string; bits: number; group_size: number; n_ignored: number };
  text?: Record<string, number>; components?: { name: string; modality: string; params: number; bytes: number; n_blocks: number }[];
}
export interface GraphTensor { name: string; full: string; dtype: string; shape: number[]; logical_shape: number[]; quantizable: boolean; aux: boolean }
export interface GraphNode {
  name: string; path: string; kind: "model" | "component" | "block" | "module"; op?: string;
  params: number; bytes: number; n_children?: number; children?: GraphNode[]; tensors?: GraphTensor[];
}
export interface TensorView {
  name: string; dtype: string; shape: number[]; numel: number; sampled?: number; unsupported?: string;
  stats?: { min: number; max: number; absmax: number; mean: number; std: number; kurtosis: number; outlier_ratio: number };
  hist?: { min: number; max: number; counts: number[] };
  quant_preview?: { recipe: string; metrics: { sqnr_db: number | null; cosine: number; rel_l2: number | null; max_abs: number } };
}

export interface RunPath { run_mode: string; best_quant: string; memory_required_gb: number; estimated_tps: number; speed_unit: string; fit_level: FitLevel; runtime: string; utilization_pct: number }
export interface FitPlan {
  model: string; rows: ModelFit[]; run_paths: RunPath[];
  minimum_hardware: ModelFit | null; recommended_hardware: ModelFit | null;
  upgrades: { hardware: string; vram_delta_gb: number; tps_delta: number; best_quant: string }[];
  kv_alternatives: { kv_quant: string; kv_bytes_per_token: number; memory_delta_gb: number }[];
}

export interface Algorithm { name: string; label: string; kind: "quantizer" | "transform" | "combo"; description: string; needs_calibration: boolean; params?: Record<string, number> }

/** The top-level problem domain a JobSpec targets (T-1301, data/domains/*.yaml). */
export interface Domain {
  id: string;
  display_name: string;
  model_families: string[];
  applicable_recipes: string[];
  classic_algorithms: { name: string; note: string }[];
  default_eval_dataset: string;
  /** "implemented": a real quantization pipeline exists. "ui_only_pending_backend": selectable, but
   * submitting a job for it is rejected by the server with a clear reason -- not silently accepted. */
  pipeline_status: "implemented" | "ui_only_pending_backend";
}

export interface ExportSuggestion { format: string; verdict: "native" | "emulated" | "unsupported"; reason: string }

export interface ConcurrencyEstimate {
  requested_users: number; max_users: number; context: number;
  weights_gb: number; kv_per_user_gb: number; overhead_gb: number;
  total_gb: number; capacity_gb: number; utilization_pct: number;
  fits: boolean; high_pressure: boolean;
  max_context_for_users: number;
  curve?: { users: number; total_gb: number; weights_gb: number; kv_gb: number; overhead_gb: number; utilization_pct: number; fits: boolean; max_context: number }[];
}

export const api = {
  health: () => req<{ status: string; version: string }>("GET", "/health"),
  config: () => req<{ allowed_roots: string[]; gpus: number[]; runs_dir: string }>("GET", "/config"),
  recipes: () => req<Recipe[]>("GET", "/recipes"),
  algorithms: () => req<Algorithm[]>("GET", "/algorithms"),
  domains: () => req<Domain[]>("GET", "/domains"),
  exportSuggestion: (recipe: string, backend: string, hardware: string) =>
    req<ExportSuggestion>(
      "GET",
      `/export-suggestion?recipe=${encodeURIComponent(recipe)}&backend=${encodeURIComponent(backend)}&hardware=${encodeURIComponent(hardware)}`,
    ),
  hardwareProfiles: () => req<HardwareProfile[]>("GET", "/hardware/profiles"),
  hardwarePlugins: () => req<HalPlugin[]>("GET", "/hardware/plugins"),
  profileDevice: (plugin: string, device: number, kind: "bandwidth" | "gemm", dtype = "fp32") =>
    req<ProfileResult>("POST", "/hardware/profile", { plugin, device, kind, dtype }),
  hardware: () => req<{ devices: Device[]; job_gpus: { pool: { gpu: number; job: string | null }[]; queued: number } }>("GET", "/hardware"),
  inspect: (path: string) => req<Record<string, any>>("POST", "/models:inspect", { path }),
  backends: () => req<BackendInfo[]>("GET", "/backends"),
  recommend: (model: string, hardware: string[], backend: string[], objective: AdviceObjective, runtime: Partial<RuntimeParams>, top_k = 8) =>
    req<{ top: Advice[]; excluded: number; considered: number; objective: AdviceObjective; measured_jobs: number }>("POST", "/recommend", { model, hardware, backend, objective, runtime, top_k }),
  estimate: (model: string, hardware: string, recipes: string[], runtime: RuntimeParams, backend = "hf_transformers") =>
    req<{ hardware: HardwareProfile; estimates: Estimate[] }>("POST", "/estimate", { model, hardware, recipes, runtime, backend }),
  matrix: (model: string, runtime: RuntimeParams) => req<{ cells: MatrixCell[] }>("POST", "/estimate:matrix", { model, runtime }),
  jobs: () => req<JobListItem[]>("GET", "/jobs"),
  job: (id: string) => req<JobDetail>("GET", `/jobs/${id}`),
  events: (id: string, after = 0) => req<JobEvent[]>("GET", `/jobs/${id}/events?after_seq=${after}`),
  submit: (job: unknown) => req<{ job_id: string }>("POST", "/jobs", job),
  cancel: (id: string) => req<{ state: string }>("POST", `/jobs/${id}:cancel`),
  models: () => req<LibraryModel[]>("GET", "/models"),
  ggufFiles: () => req<{ path: string; name: string; dir: string; bytes: number }[]>("GET", "/gguf"),
  gguf: (path: string, tensors = true) => req<GgufSummary>("POST", "/models:gguf", { path, tensors }),
  graph: (path: string, focus = "", depth = 2) => req<GraphNode>("POST", "/models:graph", { path, focus, depth }),
  tensor: (path: string, tensor: string, recipe = "") => req<TensorView>("POST", "/models:tensor", { path, tensor, recipe }),
  fit: (q: FitQuery) => req<{ system: FitSystem; models: ModelFit[]; errors: { model: string; error: string }[]; use_cases: string[] }>("POST", "/fit", q),
  plan: (model: string, q: Partial<FitQuery>) => req<FitPlan>("POST", "/fit:plan", { model, ...q }),
  claim: (model: string, hardware: string, min_tps: number) => req<ClaimBounds>("POST", "/fit:claim", { model, hardware, min_tps }),
  installed: () => req<InstalledIndex>("GET", "/installed"),
  catalog: (search = "", provider = "", capability = "", limit = 100, include_demoted = false) =>
    req<{
      updated_at: string; total: number; demoted: number; models: CatalogModel[];
      /** T-1303: background refresh status; refresh_interval_s === 0 means it is disabled for this server. */
      refresh_interval_s: number; last_refresh_ok: boolean; last_refresh_at: string;
    }>(
      "GET", `/catalog?search=${encodeURIComponent(search)}&provider=${encodeURIComponent(provider)}&capability=${encodeURIComponent(capability)}&limit=${limit}&include_demoted=${include_demoted ? 1 : 0}`),
  hfSearch: (q: string, limit = 20) => req<{ models: HfHit[] }>("GET", `/hf/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  download: (model: string, opts: { quant?: string; budget?: string; list?: boolean } = {}) =>
    req<DownloadStatus & { groups?: { group: string; bytes: number; files: number; quality: number }[]; chosen?: string }>("POST", "/download", { model, ...opts }),
  downloadStatus: (id: string) => req<DownloadStatus>("GET", `/download/${id}/status`),
  backendEnvs: (check = false) =>
    req<{ backends: BackendEnv[]; checked: boolean; installing: BackendInstall | null }>("GET", `/backend-envs${check ? "?check=1" : ""}`),
  installBackend: (id: BackendId, version?: string) => req<{ install_id: string }>("POST", `/backend-envs/${id}/install`, version ? { version } : undefined),
  backendInstall: (installId: string) => req<BackendInstall>("GET", `/backend-envs/install/${installId}`),
  cancelDownload: (id: string) => req<DownloadStatus>("POST", `/download/${id}:cancel`),
  concurrency: (model: string, hardware: string, recipe: string, users: number, context: number) =>
    req<ConcurrencyEstimate>("POST", "/fit:concurrency", { model, hardware, recipe, users, context }),
};

/** Follows a job's event stream (SSE). Reconnects resume from the last seen seq. */
export function followJob(id: string, onEvent: (e: JobEvent) => void, onEnd: () => void): () => void {
  let closed = false;
  let lastSeq = 0;
  let es: EventSource | null = null;
  const open = () => {
    es = new EventSource(`/api/v1/jobs/${id}/stream?after_seq=${lastSeq}`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data) as JobEvent;
      if (ev.seq <= lastSeq) return;
      lastSeq = ev.seq;
      onEvent(ev);
      if (ev.type === "job_state" && ev.state !== "RUNNING") {
        es?.close();
        closed = true;
        onEnd();
      }
    };
    es.onerror = () => {
      es?.close();
      if (!closed) setTimeout(() => !closed && open(), 1500);
    };
  };
  open();
  return () => {
    closed = true;
    es?.close();
  };
}

export const fmtBytes = (b: number | null | undefined): string => {
  if (b == null || !isFinite(b)) return "–";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (Math.abs(b) >= 1024 && i < u.length - 1) {
    b /= 1024;
    i++;
  }
  return `${b.toFixed(i === 0 ? 0 : 2)} ${u[i]}`;
};
export const gib = (b: number) => b / 2 ** 30;
export const fmt = (v: number | null | undefined, d = 3) => (v == null || !isFinite(v) ? "–" : v.toFixed(d));
