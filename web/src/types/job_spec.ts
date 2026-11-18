// T-1305: the JobSpec type, hand-written to match schemas/job_spec.schema.json (keep the two in sync by
// hand until T-0007 adds codegen). Not yet wired into NewJob.tsx's submission, which still builds an
// untyped object literal for both the VLA and LLM branches -- that replacement is T-1304's job, since
// NewJob.tsx's whole structure is being rebuilt there anyway; introducing this type now and immediately
// reworking its only real call site would be thrash. Exported so T-1304 (and anything else that needs to
// describe a job) can import a real type today.

export type Domain = "llm" | "vla" | "vlm" | "world" | "ewm";

export interface MixedPrecisionRule {
  match: string;
  bits?: number;
  skip?: boolean;
}

export interface KvSpec {
  format?: string;
  bits?: number;
  symmetric?: boolean;
  method?: string;
}

export interface JobSpec {
  apiVersion?: string;
  kind?: string;
  // Optional today -- every existing caller keys off model.type instead. See T-1301's data/domains/*.yaml.
  domain?: Domain;
  metadata?: { name?: string };
  model: {
    source?: string;
    path: string;
    /** An already-quantized checkpoint to evaluate instead of quantizing model.path again (LLM only). */
    quantized_path?: string;
    quant_format?: string;
    type?: "llm" | "vlm" | "vla" | "world" | "ewm";
    modality?: string;
  };
  /** Explicit pipeline stages; absent keeps the historical all-in-one pipeline. */
  stages?: ("quantize" | "evaluate" | "export" | "deploy")[];
  quant: {
    recipe: string[];
    algorithm?: string;
    act_static?: string | null;
    also_quantize?: string[];
    kv?: KvSpec;
    mixed_precision?: { rules: MixedPrecisionRule[] };
  };
  calibration?: {
    dataset?: string;
    n_samples?: number;
    seq_len?: number;
    frames?: number;
    // T-1304: free-form calibration text instead of a known dataset. Not yet read by any pipeline stage.
    custom_samples?: string[];
  };
  evaluation?: {
    dataset?: string;
    n_samples?: number;
    seq_len?: number;
    max_windows?: number;
    max_new_tokens?: number;
    suite?: string;
    benchmarks?: string | string[];
    limit?: number;
    frames?: number;
    sample_steps?: number;
    action_conditioned?: boolean;
    action_dim?: number;
    concurrency?: number[];
    attribution?: boolean;
    // T-1304: independent of deploy.device/quant_device. Not yet read by any pipeline stage.
    device?: string;
  };
  // T-1304: the device the quantization stage itself runs on. Not yet read by any pipeline stage.
  quant_device?: string;
  deploy?: {
    hardware?: string;
    device?: string;
    // T-1304: explicit backend choice, independent of export.formats/deploy.llamacpp/deploy.vllm today.
    backend?: "hf_transformers" | "vllm" | "sglang" | "llama.cpp";
    llamacpp?: { types: string[] };
    vllm?: {
      gpu_memory_utilization?: number;
      lm_eval?: { tasks: string[]; limit?: number; num_fewshot?: number; baseline?: boolean };
      evalscope?: EvalScopeSpec;
      sweep?: Record<string, unknown>;
    };
    sglang?: {
      mem_fraction_static?: number;
      lm_eval?: { tasks: string[]; limit?: number; num_fewshot?: number; baseline?: boolean };
      evalscope?: EvalScopeSpec;
      sweep?: Record<string, unknown>;
    };
    native?: { concurrency?: number; requests?: number; seed?: number };
  };
  export?: { formats: string[] };
  analysis?: {
    sensitivity?: string;
    sensitivity_windows?: number;
    logits_probe?: { exact?: boolean };
  };
  output_dir?: string;
}

interface EvalScopeSpec {
  datasets: string[];
  limit?: number;
  generation_config?: string | Record<string, unknown>;
  eval_batch_size?: number;
  seed?: number;
  timeout?: number;
}
