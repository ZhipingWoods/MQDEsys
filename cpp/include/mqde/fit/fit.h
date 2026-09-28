#pragma once
// Deployment pre-evaluation ("will it run, how fast, with which quant") — a C++ port of llmfit's
// fit/score engine (docs/REFERENCES.md §1), extended with our exact per-recipe memory model
// (group metadata, linear-attention state, vision/action components) from mqde::estimator.

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "mqde/estimator/estimator.h"

namespace mqde::fit {

enum class FitLevel { kPerfect, kGood, kMarginal, kTooTight };
enum class RunMode { kGpu, kTensorParallel, kMoeOffload, kCpuOffload, kCpuOnly };
enum class Confidence { kMeasuredLocal, kMeasuredCommunity, kEstimated };

std::string_view to_string(FitLevel);
std::string_view to_string(RunMode);
std::string_view to_string(Confidence);

struct UseCase {
  std::string name;
  double w_quality = 0.45, w_speed = 0.30, w_fit = 0.15, w_context = 0.10;
  double speed_target = 40;  // tok/s (LLM) or decision steps/s (VLA)
  int64_t context_target = 4096;
};

// MoE two-tier decode model (docs/REFERENCES.md A.5).
struct MoeConfig {
  double tier1_fixed_bpp = 3.2;
  std::map<std::string, double> tier1_fixed_bpp_by_arch;            // model_type prefix -> bpp
  std::map<std::string, std::pair<double, double>> tier2_by_arch;   // prefix -> (efficiency, overhead)
  std::vector<std::pair<double, double>> tier2_overhead_by_experts; // (max experts, overhead)
  double tier2_overhead_many = 0.40, tier2_overhead_unknown = 0.60;
  double vram_pressure_threshold = 0.60, vram_pressure_floor = 0.30, default_expert_ratio = 0.50;
};

// Per-token bandwidth decomposition of a MoE model (llmfit moe_bandwidth_decomposition), billions.
struct MoeDecomposition {
  double active_ffn_b = 0;  // routed experts actually read per token
  double fixed_b = 0;       // attention + shared expert + router + lm_head + embedding
};
// nullopt unless the model is MoE and hidden/layers/top-k/expert width/vocab are all known.
std::optional<MoeDecomposition> moe_decomposition(const modelspec::ModelSpec& spec);
// Tier-2 overhead by expert count (n_experts <= 0 means unknown).
double moe_tier2_overhead(int64_t n_experts, const MoeConfig& mc);
// VRAM cache-pressure factor for GPU-mode MoE. util = total weights / VRAM.
// 1.0 when util > 1.0 or util < threshold; else max(floor, 1 - (util - threshold) * inactive_ratio).
double moe_vram_pressure(double util, int64_t n_experts, int64_t experts_per_tok, const MoeConfig& mc);

struct FitConfig {
  std::map<std::string, double> memory_bpp, speed_bpp, quality_penalty, speed_k, quant_speed_mult, run_mode_factor;
  std::vector<std::string> quant_hierarchy;
  double overhead_gb = 0.5;
  int64_t default_context = 8192;
  double th_perfect = 0.60, th_good = 0.85, th_marginal = 0.98;
  double bw_efficiency = 0.55, prefill_util = 0.35, ddr_bw_gbps = 50;
  std::map<std::string, UseCase> use_cases;
  std::vector<std::pair<double, double>> quality_tiers;  // (upper bound B, score)
  // Quality adjustments (see docs/REFERENCES.md A.1).
  std::map<std::string, double> family_bonus;            // family substring -> bonus
  std::vector<std::pair<double, double>> recency_bonus;  // (age months upper bound, bonus)
  std::map<std::string, double> task_bonus;              // coding_name_match | reasoning_large | vision_multimodal
  double reasoning_large_min_b = 13;
  // K-constant speed path bonus for many-core hosts (A.4).
  int64_t many_cores_threshold = 8;
  double many_cores_speed_bonus = 1.1;
  MoeConfig moe;
};

Result<FitConfig> load_fit_config();  // data/fit/fit.yaml

// The machine a model is evaluated against (from the probe or a hardware profile).
struct System {
  std::string name;
  std::string backend = "cuda";  // cuda|rocm|metal|vulkan|ascend|cpu_x86|cpu_arm
  int gpu_count = 1;
  double vram_gb = 0;            // per GPU
  double vram_free_gb = 0;       // per GPU, 0 = unknown (use total)
  double ram_gb = 0;
  double mem_bw_gbps = 0;        // per GPU, 0 = unknown
  double peak_tflops_bf16 = 0;
  int cpu_cores = 0;             // 0 = unknown (no many-core bonus)
  bool unified_memory = false;
  std::vector<std::string> native_formats;
};

System system_from_profile(const estimator::HardwareProfile& hw, double ram_gb = 0);

struct Request {
  std::string use_case = "general";
  int64_t context = 0;           // 0 = use-case default for LLMs / derived for VLA
  int64_t concurrency = 1;
  std::string force_quant;       // empty = choose the best that fits
  std::string force_run_mode;    // empty = auto
  double measured_tps = 0;       // >0 overrides the estimate (local/community benchmark)
  Confidence measured_source = Confidence::kEstimated;
};

struct ScoreComponents {
  double quality = 0, speed = 0, fit = 0, context = 0;
};

struct ModelFit {
  std::string model, family, modality;
  double params_b = 0, active_params_b = 0;
  bool is_moe = false;
  std::string best_quant;
  std::vector<std::string> fitting_quants;  // every quant in the hierarchy that fits
  RunMode run_mode = RunMode::kGpu;
  FitLevel fit_level = FitLevel::kTooTight;
  double memory_required_gb = 0, memory_available_gb = 0, utilization_pct = 0;
  double estimated_tps = 0;   // decode tok/s (LLM) or decision steps/s (VLA)
  // Null when the hardware profile has no compute figure, or on CPU-only: llmfit reports no TTFT
  // rather than a fabricated one (docs/REFERENCES.md A.6).
  std::optional<double> prefill_tps, ttft_ms;
  std::string speed_unit = "tok/s";
  Confidence confidence = Confidence::kEstimated;
  ScoreComponents components;
  double score = 0;
  int64_t context = 0;
  std::vector<std::string> notes;
  // MoE speed path actually taken: 0 = not MoE / dense formula, 1 = Tier-1, 2 = Tier-2.
  int moe_tier = 0;
  double moe_vram_pressure = 1.0;  // cache-pressure factor applied (GPU mode), 1.0 = none
  double moe_offloaded_gb = 0;     // expert weights placed in system RAM (MoE offload mode)
  estimator::Estimate detail;  // full memory breakdown for the chosen quant
};

// Evaluates one model on one system (llmfit analyze()).
Result<ModelFit> analyze(const modelspec::ModelSpec& spec, const System& sys, const FitConfig& cfg,
                         const estimator::BackendProfile& backend, const Request& req);

// Sort key helpers (llmfit ranking: TooTight last, then by the chosen column).
void rank(std::vector<ModelFit>& fits, std::string_view sort_by = "score");

Json to_json(const ModelFit& f);
Json to_json(const System& s);

}  // namespace mqde::fit
