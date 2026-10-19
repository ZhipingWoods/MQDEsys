#pragma once
// Resource estimator (docs/ARCHITECTURE.md §9): weights, quantization metadata, KV cache,
// linear-attention recurrent state, activations, runtime overhead, concurrency limits and a
// roofline latency floor. Pure functions, no I/O beyond the loaders below.

#include <map>
#include <string>
#include <vector>

#include "mqde/modelspec/model_spec.h"
#include "mqde/quant/quant_spec.h"

namespace mqde::estimator {

struct Recipe {
  std::string name;
  std::string description;
  std::string maturity;  // production | experimental
  std::string default_algorithm;
  std::vector<std::string> modalities;  // component modalities the weight spec applies to
  quant::QuantSpec weight, act, kv;
};

struct HardwareProfile {
  std::string id, vendor, name, arch;
  std::vector<std::string> match_names;
  uint64_t mem_bytes = 0;
  // T-A: a live reading of this device's actually-free memory right now (NVML / /proc/meminfo), filled in
  // by the server -- never by this header's functions, which stay pure -- only when `id` matches a device
  // physically present on the machine answering the request. 0 is "not probed" (the common case: the
  // requested hardware isn't the one running here, or this is an offline estimate), in which case `estimate()`
  // falls back to `mem_bytes` exactly as before this field existed. When set, it -- not the static datasheet
  // total -- is what capacity/feasibility/concurrency are computed from, because another process or job may
  // already be using part of the card.
  uint64_t live_free_bytes = 0;
  double mem_bw_gbps = 0;
  std::map<std::string, double> peak_tflops;  // fp32/bf16/fp16/fp8/int8/int4
  std::vector<std::string> native_formats;
  std::string source;  // T-1307: vendor datasheet/doc URL these numbers came from, shown in the hardware
                       // page's detail view; every data/hardware_db/**/*.yaml entry has one, so this is
                       // optional (empty string) only for an entry someone adds without it.
};

struct BackendProfile {
  std::string id = "hf_transformers";
  uint64_t runtime_overhead_bytes = 512ull << 20;
  double allocator_factor = 1.05;
  double eta_bw = 0.6;
  double mfu = 0.35;
  // "per_seq": every concurrent sequence gets its own activation memory (HF eager, one forward per sequence).
  // "batched": the engine packs all sequences into one batch of at most max_batched_tokens tokens, so the
  // language model's activation memory is charged once for min(context x concurrency, max_batched_tokens)
  // tokens (vLLM, ADR-0003).
  std::string activation_model = "per_seq";
  int64_t max_batched_tokens = 8192;  // only used by "batched"
};

// Domain registry (T-1301): the top-level problem domain a JobSpec targets, one data/domains/<id>.yaml per
// domain. `id` matches modelspec::ModelFamily.modality ("llm" | "vlm" | "vla" | "world") plus any new
// domains added purely as data (e.g. "ewm", the embodied/action-conditioned world-model variant) without
// needing a matching ModelFamily yet. This only centralizes domain *metadata* (which recipes/algorithms are
// relevant, whether a real pipeline exists); it does not replace the finer-grained per-component modality
// checks already in fit.cpp (vision/audio/tts/mtp capability flags), which are a different, unrelated axis.
struct Domain {
  std::string id;
  std::string display_name;
  std::vector<std::string> model_families;      // data/model_families/<id> entries this domain covers
  std::vector<std::string> applicable_recipes;  // data/recipes/<id> entries; empty = no restriction
  struct ClassicAlgorithm {
    std::string name, note;
  };
  std::vector<ClassicAlgorithm> classic_algorithms;  // reference list (docs/REFERENCES.md-style); an entry
                                                     // here does not imply it is implemented
  std::string default_eval_dataset;
  // "implemented": a real quantization pipeline exists for this domain (see python/mqde/pipeline/*).
  // "ui_only_pending_backend": the domain is selectable but submitting a job for it is rejected with a
  // clear reason, not silently accepted.
  std::string pipeline_status = "ui_only_pending_backend";
  // Position of this domain's tab in the new-job wizard (ascending); a domain without one sorts after every
  // numbered one, by id. The order is data so adding a domain never touches frontend code.
  int order = 1000;
  // Inference backends a quantized export of this domain can be served by (e.g. "vllm", "sglang",
  // "llamacpp"). Empty means no backend deployment is supported or implemented. Drives the frontend's
  // BackendCards gate so new domains never require a frontend code change.
  std::vector<std::string> deploy_backends;
};

Result<Recipe> load_recipe(std::string_view name_or_path);
Result<HardwareProfile> load_hardware(std::string_view id_or_path);
Result<BackendProfile> load_backend(std::string_view id_or_path);
Result<Domain> load_domain(std::string_view id_or_path);
// Every domain, in wizard order (`order` ascending, then id). A domain file that fails to load is an error,
// not skipped.
Result<std::vector<Domain>> list_domains();
Result<std::vector<Domain>> list_domains_in(const std::filesystem::path& dir);
Result<std::vector<std::string>> list_data_entries(std::string_view sub);  // recipes, hardware_db, ...
// Finds the hardware_db profile whose match_names occur in a probed device name ("NVIDIA GeForce RTX 4090").
Result<HardwareProfile> match_hardware(std::string_view device_name);
// T-A: the memory budget `estimate()` computes capacity/feasibility/concurrency from -- `hw.live_free_bytes`
// when the caller has probed one (> 0), else the static `hw.mem_bytes` datasheet total, in both cases scaled
// by `gpu_memory_utilization`. The one place this scaling happens, so `estimator.cpp` and `fit/concurrency.cpp`
// can't drift apart on what "capacity" means.
uint64_t effective_capacity_bytes(const HardwareProfile& hw, double gpu_memory_utilization);

struct RuntimeParams {
  int64_t concurrency = 1;      // independent sequences (robots / requests) served at once
  int64_t images_per_step = 1;  // camera views x frames in the context
  int64_t text_tokens = 32;     // instruction + template tokens
  int64_t context_tokens = 0;   // override total LM context (0 = derived from images/text)
  int64_t image_resolution = 448;
  double gpu_memory_utilization = 0.9;
  std::vector<std::string> extra_modalities;  // e.g. {"action"} to also quantize the action head
};

struct ComponentMemory {
  std::string name, modality;
  int64_t params = 0;
  uint64_t original_bytes = 0;
  uint64_t weight_bytes = 0;  // after quantization (quantized + kept tensors)
  uint64_t qmeta_bytes = 0;   // scales / zeros
  int64_t quantized_params = 0;
  bool quantized = false;
};

struct Estimate {
  std::string model, recipe, hardware, backend;
  RuntimeParams rt;
  std::vector<ComponentMemory> components;
  uint64_t weights_bytes = 0;  // incl. qmeta
  uint64_t qmeta_bytes = 0;
  uint64_t original_weights_bytes = 0;
  int64_t context_tokens = 0;
  uint64_t kv_bytes_per_token = 0;   // full-attention layers only
  uint64_t state_bytes_per_seq = 0;  // linear-attention recurrent + conv state
  uint64_t kv_bytes = 0;             // (kv/token * context + state) * concurrency
  uint64_t activation_bytes = 0;
  uint64_t runtime_bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t capacity_bytes = 0;  // mem_bytes * utilization
  bool feasible = false;
  int64_t max_concurrency = 0;
  std::vector<std::string> reasons;
  // Per decision step (VLA) or per forward (LLM prefill)
  double flops_per_step = 0;
  double bytes_per_step = 0;
  double latency_floor_ms = 0;
  // Pieces of the roofline model for token generation (used by estimate_perf): language-model weights are
  // re-read for every generated token, the LM head is touched once per token, attention work grows with the
  // context length.
  double flops_lang_per_token = 0;  // matmul FLOPs of the language layers for one token
  double flops_head_per_token = 0;  // LM head FLOPs for one token
  double bytes_lang = 0;            // weight bytes of the language layers touched per forward
  double bytes_head = 0;            // weight bytes of the LM head
  double attn_flops_per_ctx_token =
      0;  // 4 * full-attention layers * heads * head_dim: FLOPs per token per context position
  std::string compute_format;  // which peak was used
  std::vector<std::string> assumptions;
  std::string confidence = "low";
};

Estimate estimate(const modelspec::ModelSpec& spec, const Recipe& recipe, const HardwareProfile& hw,
                  const BackendProfile& backend, const RuntimeParams& rt);

// ---- token-generation performance (roofline per phase), T-0209 ----
struct PerfParams {
  int64_t prompt_tokens = 1024;
  int64_t output_tokens = 128;
};

struct PerfEstimate {
  double ttft_ms = 0;  // time to first token: all `concurrency` prompts prefill together
  double tpot_ms = 0;  // time per output token for every running sequence (one batched decode step)
  double tokens_per_s_per_seq = 0;  // 1000 / tpot_ms
  double tokens_per_s_total = 0;    // concurrency * per-seq rate
  double request_latency_ms = 0;    // ttft + (output_tokens - 1) * tpot
  double requests_per_s = 0;        // concurrency / request latency
  std::string prefill_bound;        // "compute" | "memory": which roof sets TTFT
  std::string decode_bound;         // "compute" | "memory": which roof sets TPOT
  double eta_bw = 0, mfu = 0;       // coefficients used (from the backend profile / calibration)
  std::string compute_format;
  std::string confidence = "low";  // copied from the Estimate
};

// Prefill: max(FLOPs / (peak x MFU), weight + KV-write bytes / (BW x eta)). Decode step at the average
// context length: max(FLOPs / (peak x MFU), weight bytes + KV / state read per sequence / (BW x eta)). The
// kernel-launch overhead is not modeled.
PerfEstimate estimate_perf(const Estimate& e, const HardwareProfile& hw, const BackendProfile& be,
                           const PerfParams& p);

// Inversions used by the feedback loop: the coefficient that would have produced a measurement. They fail,
// with a reason, when the measured phase is not bound by the roof the coefficient belongs to (a compute-bound
// decode says nothing about bandwidth).
Result<double> infer_eta_bw(const Estimate& e, const HardwareProfile& hw, const BackendProfile& be,
                            const PerfParams& p, double measured_tpot_ms);
Result<double> infer_mfu(const Estimate& e, const HardwareProfile& hw, const BackendProfile& be,
                         const PerfParams& p, double measured_ttft_ms);
// measured peak minus the modelled weights + KV + activations (times the allocator factor)
Result<double> infer_overhead_bytes(const Estimate& e, const BackendProfile& be, double measured_peak_bytes);
// Copy of `be` with the calibrated coefficients applied.
BackendProfile with_coefficients(BackendProfile be, double eta_bw, double mfu, double overhead_bytes);

struct CurvePoint {
  int64_t x = 0;
  uint64_t peak_bytes = 0;
  uint64_t kv_bytes = 0;
  bool feasible = false;
};
std::vector<CurvePoint> memory_vs_concurrency(const modelspec::ModelSpec&, const Recipe&,
                                              const HardwareProfile&, const BackendProfile&, RuntimeParams rt,
                                              const std::vector<int64_t>& xs);
std::vector<CurvePoint> memory_vs_images(const modelspec::ModelSpec&, const Recipe&, const HardwareProfile&,
                                         const BackendProfile&, RuntimeParams rt,
                                         const std::vector<int64_t>& xs);

Json to_json(const Recipe& r);
Json to_json(const HardwareProfile& h);
Json to_json(const Domain& d);
Json to_json(const Estimate& e);
Json to_json(const std::vector<CurvePoint>& c);

}  // namespace mqde::estimator
