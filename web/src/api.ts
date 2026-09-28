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
  kind: "cpu" | "gpu";
  index: number;
  name: string;
  arch: string;
  mem_total_bytes: Bytes;
  mem_free_bytes: Bytes;
  compute_units: number;
  driver_version: string;
  profile?: string;
}

export interface JobListItem {
  job_id: string;
  name?: string;
  state: string;
  stage?: string | null;
  recipes?: string[];
  model?: string;
  gpu?: number;
  started_ts?: number;
  last_ts?: number;
  n_events: number;
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

export interface JobReport {
  job_id: string;
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
  type: "job_state" | "stage_started" | "stage_finished" | "log" | "metric" | "trace";
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
  estimate_basis: { method: string; gpu_bandwidth_gbps: number; ddr_bandwidth_gbps: number; efficiency: number; assumed_context: number; local_calibration: number; };
  formula_tps: number;
  measured_tps: { tok_s: number; ttft_ms: number | null; sample_count: number; hardware_label: string; source: "local" | "community" } | null;
  score_components: { quality: number; speed: number; fit: number; context: number };
  notes: string[]; detail?: Estimate; hardware?: string; system?: FitSystem;
}
export interface FitQuery {
  hardware: string; gpu_count?: number; ram_gb?: number; vram_gb?: number; mem_bw_gbps?: number;
  use_case?: string; context?: number; concurrency?: number; quant?: string; sort?: string; models?: string[];
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

export const api = {
  health: () => req<{ status: string; version: string }>("GET", "/health"),
  config: () => req<{ allowed_roots: string[]; gpus: number[]; runs_dir: string }>("GET", "/config"),
  recipes: () => req<Recipe[]>("GET", "/recipes"),
  hardwareProfiles: () => req<HardwareProfile[]>("GET", "/hardware/profiles"),
  hardware: () => req<{ devices: Device[]; job_gpus: { pool: { gpu: number; job: string | null }[]; queued: number } }>("GET", "/hardware"),
  inspect: (path: string) => req<Record<string, any>>("POST", "/models:inspect", { path }),
  estimate: (model: string, hardware: string, recipes: string[], runtime: RuntimeParams) =>
    req<{ hardware: HardwareProfile; estimates: Estimate[] }>("POST", "/estimate", { model, hardware, recipes, runtime }),
  matrix: (model: string, runtime: RuntimeParams) => req<{ cells: MatrixCell[] }>("POST", "/estimate:matrix", { model, runtime }),
  jobs: () => req<JobListItem[]>("GET", "/jobs"),
  job: (id: string) => req<JobDetail>("GET", `/jobs/${id}`),
  events: (id: string, after = 0) => req<JobEvent[]>("GET", `/jobs/${id}/events?after_seq=${after}`),
  submit: (job: unknown) => req<{ job_id: string }>("POST", "/jobs", job),
  cancel: (id: string) => req<{ state: string }>("POST", `/jobs/${id}:cancel`),
  models: () => req<LibraryModel[]>("GET", "/models"),
  graph: (path: string, focus = "", depth = 2) => req<GraphNode>("POST", "/models:graph", { path, focus, depth }),
  tensor: (path: string, tensor: string, recipe = "") => req<TensorView>("POST", "/models:tensor", { path, tensor, recipe }),
  fit: (q: FitQuery) => req<{ system: FitSystem; models: ModelFit[]; errors: { model: string; error: string }[]; use_cases: string[] }>("POST", "/fit", q),
  plan: (model: string, q: Omit<FitQuery, "hardware">) => req<{ model: string; rows: ModelFit[] }>("POST", "/fit:plan", { model, ...q }),
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
