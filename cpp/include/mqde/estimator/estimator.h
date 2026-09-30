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
  double mem_bw_gbps = 0;
  std::map<std::string, double> peak_tflops;  // fp32/bf16/fp16/fp8/int8/int4
  std::vector<std::string> native_formats;
};

struct BackendProfile {
  std::string id = "hf_transformers";
  uint64_t runtime_overhead_bytes = 512ull << 20;
  double allocator_factor = 1.05;
  double eta_bw = 0.6;
  double mfu = 0.35;
};

Result<Recipe> load_recipe(std::string_view name_or_path);
Result<HardwareProfile> load_hardware(std::string_view id_or_path);
Result<BackendProfile> load_backend(std::string_view id_or_path);
Result<std::vector<std::string>> list_data_entries(std::string_view sub);  // recipes, hardware_db, ...
// Finds the hardware_db profile whose match_names occur in a probed device name ("NVIDIA GeForce RTX 4090").
Result<HardwareProfile> match_hardware(std::string_view device_name);

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
  std::string compute_format;  // which peak was used
  std::vector<std::string> assumptions;
  std::string confidence = "low";
};

Estimate estimate(const modelspec::ModelSpec& spec, const Recipe& recipe, const HardwareProfile& hw,
                  const BackendProfile& backend, const RuntimeParams& rt);

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
Json to_json(const Estimate& e);
Json to_json(const std::vector<CurvePoint>& c);

}  // namespace mqde::estimator
