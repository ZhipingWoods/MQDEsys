#include "mqde/fit/fit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

#include <fmt/ranges.h>

namespace mqde::fit {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

double lookup(const std::map<std::string, double>& m, const std::string& k, double fallback) {
  auto it = m.find(k);
  return it == m.end() ? fallback : it->second;
}

std::map<std::string, double> read_map(const Json& j, const char* key) {
  std::map<std::string, double> out;
  if (auto it = j.find(key); it != j.end() && it->is_object())
    for (auto e = it->begin(); e != it->end(); ++e)
      if (e->is_number()) out[e.key()] = e->get<double>();
  return out;
}

FitLevel level_for(double ratio, const FitConfig& c) {
  if (ratio <= c.th_perfect) return FitLevel::kPerfect;
  if (ratio <= c.th_good) return FitLevel::kGood;
  if (ratio <= c.th_marginal) return FitLevel::kMarginal;
  return FitLevel::kTooTight;
}

// Strict upper bounds, as llmfit (fit.rs:1969): "< 1B -> 30, < 3B -> 45, ...", so 1.0B lands in 45.
double quality_base(double params_b, const FitConfig& c) {
  for (const auto& [ub, score] : c.quality_tiers)
    if (params_b < ub) return score;
  return c.quality_tiers.empty() ? 60.0 : c.quality_tiers.back().second;
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(::tolower(ch)); });
  return s;
}

// Family bonus: the LARGEST matching entry, not the sum — "MiniCPM-o" would otherwise collect
// several overlapping keywords. Matched against family / model_type / path (lower-cased).
double family_bonus_for(const modelspec::ModelSpec& spec, const FitConfig& c) {
  const std::string hay = lower(spec.family) + "\n" + lower(spec.model_type) + "\n" + lower(spec.path);
  double best = 0;
  for (const auto& [key, bonus] : c.family_bonus)
    if (hay.find(lower(key)) != std::string::npos) best = std::max(best, bonus);
  return best;
}

// Local checkpoints carry no release date, so age comes from the newest mtime of config.json /
// the model directory. Returns 0 when the path is unreadable (no bonus, never a penalty).
double recency_bonus_for(const modelspec::ModelSpec& spec, const FitConfig& c) {
  if (c.recency_bonus.empty() || spec.path.empty()) return 0;
  std::error_code ec;
  auto newest = std::filesystem::file_time_type::min();
  for (const char* name : {"config.json", ""}) {
    const auto p = name[0] ? std::filesystem::path(spec.path) / name : std::filesystem::path(spec.path);
    const auto t = std::filesystem::last_write_time(p, ec);
    if (!ec) newest = std::max(newest, t);
  }
  if (newest == std::filesystem::file_time_type::min()) return 0;
  const auto age = std::filesystem::file_time_type::clock::now() - newest;
  const double months = std::chrono::duration<double, std::ratio<2629746>>(age).count();  // mean Gregorian month
  if (months < 0) return 0;
  for (const auto& [ub, bonus] : c.recency_bonus)
    if (months < ub) return bonus;
  return 0;
}

// Heuristic task bonus (llmfit A.1); we have no per-model benchmark table yet, so only the
// heuristic branch exists. Keyed off the resolved use case, not the model's own modality.
double task_bonus_for(const modelspec::ModelSpec& spec, const std::string& use_case, double active_b,
                      const FitConfig& c) {
  const auto get = [&](const char* k) { return lookup(c.task_bonus, k, 0.0); };
  if (use_case == "coding") {
    const std::string hay = lower(spec.path) + "\n" + lower(spec.model_type);
    for (const char* kw : {"coder", "coding", "code", "starcoder", "deepseek-v"})
      if (hay.find(kw) != std::string::npos) return get("coding_name_match");
    return 0;
  }
  if (use_case == "reasoning") return active_b >= c.reasoning_large_min_b ? get("reasoning_large") : 0;
  if (use_case == "multimodal") return spec.vision ? get("vision_multimodal") : 0;
  return 0;
}

}  // namespace

std::string_view to_string(FitLevel f) {
  switch (f) {
    case FitLevel::kPerfect: return "perfect";
    case FitLevel::kGood: return "good";
    case FitLevel::kMarginal: return "marginal";
    case FitLevel::kTooTight: return "too_tight";
  }
  return "unknown";
}
std::string_view to_string(RunMode m) {
  switch (m) {
    case RunMode::kGpu: return "gpu";
    case RunMode::kTensorParallel: return "tensor_parallel";
    case RunMode::kMoeOffload: return "moe_offload";
    case RunMode::kCpuOffload: return "cpu_offload";
    case RunMode::kCpuOnly: return "cpu_only";
  }
  return "unknown";
}
std::string_view to_string(Confidence c) {
  switch (c) {
    case Confidence::kMeasuredLocal: return "measured_local";
    case Confidence::kMeasuredCommunity: return "measured_community";
    case Confidence::kEstimated: return "estimated";
  }
  return "unknown";
}

Result<FitConfig> load_fit_config() {
  MQDE_ASSIGN_OR_RETURN(auto path, resolve_data_file("fit", "fit"));
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(path));
  FitConfig c;
  c.memory_bpp = read_map(j, "memory_bpp");
  c.speed_bpp = read_map(j, "speed_bpp");
  c.quality_penalty = read_map(j, "quant_quality_penalty");
  c.speed_k = read_map(j, "backend_speed_k");
  c.quant_speed_mult = read_map(j, "quant_speed_multiplier");
  c.run_mode_factor = read_map(j, "run_mode_factor");
  c.quant_hierarchy = get_or<std::vector<std::string>>(j, "quant_hierarchy", {"bf16", "fp8", "w8a16", "w4a16"});
  c.overhead_gb = get_or<double>(j, "overhead_gb", c.overhead_gb);
  c.default_context = get_or<int64_t>(j, "default_context", c.default_context);
  if (auto t = j.find("fit_thresholds"); t != j.end()) {
    c.th_perfect = get_or<double>(*t, "perfect", c.th_perfect);
    c.th_good = get_or<double>(*t, "good", c.th_good);
    c.th_marginal = get_or<double>(*t, "marginal", c.th_marginal);
  }
  c.bw_efficiency = get_or<double>(j, "bandwidth_efficiency", c.bw_efficiency);
  c.prefill_util = get_or<double>(j, "prefill_compute_utilization", c.prefill_util);
  c.ddr_bw_gbps = get_or<double>(j, "default_ddr_bandwidth_gbps", c.ddr_bw_gbps);
  if (auto u = j.find("use_cases"); u != j.end() && u->is_object())
    for (auto e = u->begin(); e != u->end(); ++e) {
      UseCase uc;
      uc.name = e.key();
      auto w = get_or<std::vector<double>>(*e, "weights", {0.45, 0.30, 0.15, 0.10});
      if (w.size() != 4) return make_error(Errc::kParse, "fit.yaml: use_cases.{}.weights must have 4 entries", e.key());
      uc.w_quality = w[0], uc.w_speed = w[1], uc.w_fit = w[2], uc.w_context = w[3];
      uc.speed_target = get_or<double>(*e, "speed_target", 40);
      uc.context_target = get_or<int64_t>(*e, "context_target", 4096);
      c.use_cases[uc.name] = uc;
    }
  for (const auto& row : j.value("quality_tiers", Json::array()))
    if (row.is_array() && row.size() == 2) c.quality_tiers.emplace_back(row[0].get<double>(), row[1].get<double>());
  c.family_bonus = read_map(j, "family_quality_bonus");
  c.task_bonus = read_map(j, "task_quality_bonus");
  for (const auto& row : j.value("recency_quality_bonus", Json::array()))
    if (row.is_array() && row.size() == 2) c.recency_bonus.emplace_back(row[0].get<double>(), row[1].get<double>());
  c.reasoning_large_min_b = get_or<double>(j, "reasoning_large_min_b", c.reasoning_large_min_b);
  c.many_cores_threshold = get_or<int64_t>(j, "many_cores_threshold", c.many_cores_threshold);
  c.many_cores_speed_bonus = get_or<double>(j, "many_cores_speed_bonus", c.many_cores_speed_bonus);
  return c;
}

System system_from_profile(const estimator::HardwareProfile& hw, double ram_gb) {
  System s;
  s.name = hw.name;
  s.backend = hw.vendor == "nvidia" ? "cuda" : hw.vendor == "amd" ? "rocm" : hw.vendor == "apple" ? "metal"
              : hw.vendor == "ascend" ? "ascend" : hw.vendor == "cpu" ? "cpu_x86" : hw.vendor;
  s.vram_gb = static_cast<double>(hw.mem_bytes) / kGiB;
  s.mem_bw_gbps = hw.mem_bw_gbps;
  s.peak_tflops_bf16 = hw.peak_tflops.contains("bf16") ? hw.peak_tflops.at("bf16") : 0.0;
  s.native_formats = hw.native_formats;
  s.unified_memory = hw.vendor == "apple" || hw.name.find("Orin") != std::string::npos;
  s.ram_gb = ram_gb;
  return s;
}

Result<ModelFit> analyze(const modelspec::ModelSpec& spec, const System& sys, const FitConfig& cfg,
                         const estimator::BackendProfile& backend, const Request& req) {
  ModelFit f;
  f.model = spec.path.empty() ? spec.model_type : spec.path;
  f.family = spec.family;
  f.modality = spec.modality;
  const int64_t logical = spec.logical_params > 0 ? spec.logical_params : spec.total_params;
  f.params_b = static_cast<double>(logical) / 1e9;
  f.active_params_b = static_cast<double>(spec.active_params > 0 ? spec.active_params : logical) / 1e9;
  f.is_moe = spec.text.n_experts > 0;

  const bool vla = spec.modality == "vla";
  std::string uc_name = req.use_case;
  if (uc_name == "auto" || uc_name.empty()) uc_name = vla ? "vla" : spec.vision ? "multimodal" : "general";
  auto uc_it = cfg.use_cases.find(uc_name);
  if (uc_it == cfg.use_cases.end()) return make_error(Errc::kInvalidArgument, "unknown use case '{}'", uc_name);
  const UseCase& uc = uc_it->second;

  estimator::RuntimeParams rt;
  rt.concurrency = std::max<int64_t>(1, req.concurrency);
  if (!vla) rt.context_tokens = req.context > 0 ? req.context : std::min(cfg.default_context, std::max<int64_t>(uc.context_target, 1));
  else if (req.context > 0) rt.context_tokens = req.context;

  const double vram_total = sys.vram_gb * std::max(1, sys.gpu_count);
  const double vram_avail = (sys.vram_free_gb > 0 ? sys.vram_free_gb : sys.vram_gb) * std::max(1, sys.gpu_count);
  const bool has_gpu = sys.vram_gb > 0;
  estimator::HardwareProfile hw;
  hw.id = sys.name;
  hw.name = sys.name;
  hw.mem_bytes = static_cast<uint64_t>((has_gpu ? vram_avail : sys.ram_gb) * kGiB);
  hw.mem_bw_gbps = sys.mem_bw_gbps;
  hw.native_formats = sys.native_formats;
  if (sys.peak_tflops_bf16 > 0) hw.peak_tflops["bf16"] = sys.peak_tflops_bf16;
  rt.gpu_memory_utilization = 1.0;  // llmfit grades against the raw pool; thresholds provide the headroom

  // Walk the quant hierarchy (llmfit choose_quant): the first one whose ratio is not TooTight wins.
  std::vector<std::string> candidates = cfg.quant_hierarchy;
  if (!req.force_quant.empty()) candidates = {req.force_quant};
  std::optional<estimator::Estimate> chosen;
  std::string chosen_q;
  estimator::Estimate last;
  std::string last_q;
  for (const auto& q : candidates) {
    auto recipe = estimator::load_recipe(q);
    if (!recipe) return std::unexpected(recipe.error().with_context("fit"));
    auto e = estimator::estimate(spec, *recipe, hw, backend, rt);
    const double req_gb = static_cast<double>(e.peak_bytes) / kGiB + cfg.overhead_gb;
    const double ratio = req_gb / std::max(1e-9, has_gpu ? vram_avail : sys.ram_gb);
    if (ratio <= cfg.th_marginal) {
      f.fitting_quants.push_back(recipe->name);
      if (!chosen) {
        chosen = e;
        chosen_q = recipe->name;
      }
    }
    last = e;
    last_q = recipe->name;
  }

  // Run mode (llmfit analyze_inner): GPU if it fits in VRAM; MoE offload if experts can live in RAM;
  // CPU offload if RAM holds it; else CPU only (and TooTight).
  const estimator::Estimate& e = chosen ? *chosen : last;
  f.best_quant = chosen ? chosen_q : last_q;
  const double req_gb = static_cast<double>(e.peak_bytes) / kGiB + cfg.overhead_gb;
  f.memory_required_gb = req_gb;
  double pool = has_gpu ? vram_avail : sys.ram_gb;
  if (!has_gpu) f.run_mode = RunMode::kCpuOnly;
  else if (chosen) f.run_mode = sys.gpu_count > 1 ? RunMode::kTensorParallel : RunMode::kGpu;
  else if (f.is_moe && sys.ram_gb > 0) {
    const double expert_gb = static_cast<double>(spec.expert_params) * lookup(cfg.memory_bpp, [&] {
      std::string k = f.best_quant; std::transform(k.begin(), k.end(), k.begin(), ::tolower); return k; }(), 2.0) / kGiB;
    if (req_gb - expert_gb <= vram_avail * cfg.th_marginal && expert_gb <= sys.ram_gb * cfg.th_marginal) {
      f.run_mode = RunMode::kMoeOffload;
      pool = vram_avail + sys.ram_gb;
    } else f.run_mode = RunMode::kCpuOffload, pool = vram_avail + sys.ram_gb;
  } else if (sys.ram_gb > 0 && !sys.unified_memory) {
    f.run_mode = RunMode::kCpuOffload;
    pool = vram_avail + sys.ram_gb;
  }
  if (!req.force_run_mode.empty()) f.notes.push_back("run mode forced: " + req.force_run_mode);
  f.memory_available_gb = pool;
  const double ratio = req_gb / std::max(1e-9, pool);
  f.utilization_pct = ratio * 100.0;
  f.fit_level = level_for(ratio, cfg);
  if ((f.run_mode == RunMode::kCpuOnly || f.run_mode == RunMode::kCpuOffload) && f.fit_level == FitLevel::kPerfect)
    f.fit_level = FitLevel::kGood;  // llmfit cap_for_run_mode
  if (!chosen && has_gpu && f.run_mode == RunMode::kGpu) f.fit_level = FitLevel::kTooTight;

  // Speed. LLM: decode tok/s from bandwidth (llmfit estimate_tps). VLA: decision steps/s from the
  // estimator's per-step roofline (bytes + FLOPs of vision + LM prefill + DiT denoising).
  std::string qkey = f.best_quant;
  std::transform(qkey.begin(), qkey.end(), qkey.begin(), ::tolower);
  const double mode_factor = lookup(cfg.run_mode_factor, std::string(to_string(f.run_mode)), 1.0);
  const double bpp = lookup(cfg.speed_bpp, qkey, 2.0);
  if (vla) {
    f.speed_unit = "steps/s";
    const double t_mem = sys.mem_bw_gbps > 0 ? e.bytes_per_step / (sys.mem_bw_gbps * 1e9 * cfg.bw_efficiency) : 0;
    const double t_cmp = sys.peak_tflops_bf16 > 0 ? e.flops_per_step / (sys.peak_tflops_bf16 * 1e12 * cfg.prefill_util) : 0;
    const double t = std::max(t_mem, t_cmp);
    f.estimated_tps = t > 0 ? mode_factor / t : 0;
    // A VLA step latency is only meaningful when both bandwidth and compute are known.
    if (t_mem > 0 && t_cmp > 0 && f.run_mode != RunMode::kCpuOnly) f.ttft_ms = t * 1e3 / std::max(mode_factor, 1e-9);
  } else {
    if (sys.mem_bw_gbps > 0) {
      f.estimated_tps = sys.mem_bw_gbps / std::max(1e-9, f.active_params_b * bpp) * cfg.bw_efficiency * mode_factor;
      if (f.run_mode == RunMode::kCpuOffload || f.run_mode == RunMode::kMoeOffload) {
        // Offloaded bytes stream from DDR (llmfit moe offload path).
        const double gb_per_tok = f.active_params_b * bpp;
        const double ddr = sys.ram_gb > 0 ? cfg.ddr_bw_gbps : 1e9;
        const double t = gb_per_tok / ddr * 0.5 + gb_per_tok / (sys.mem_bw_gbps * cfg.bw_efficiency) * 0.5;
        f.estimated_tps = 1.0 / std::max(1e-9, t) * mode_factor;
      }
    } else {
      const double k = lookup(cfg.speed_k, sys.backend, 70);
      f.estimated_tps = k / std::max(1e-9, f.active_params_b) * lookup(cfg.quant_speed_mult, qkey, 1.0) * mode_factor;
      if (sys.cpu_cores >= cfg.many_cores_threshold) f.estimated_tps *= cfg.many_cores_speed_bonus;
    }
    const int64_t ctx = e.context_tokens > 0 ? e.context_tokens : cfg.default_context;
    // llmfit A.6: null unless compute is known and we are not CPU-only.
    if (sys.peak_tflops_bf16 > 0 && f.run_mode != RunMode::kCpuOnly) {
      const double ptps = sys.peak_tflops_bf16 * 1e12 * cfg.prefill_util / (2.0 * f.active_params_b * 1e9);
      f.prefill_tps = ptps;
      f.ttft_ms = static_cast<double>(ctx) / ptps * 1e3;
    }
  }
  if (req.measured_tps > 0) {
    f.estimated_tps = req.measured_tps;
    f.confidence = req.measured_source;
  }
  f.context = e.context_tokens;

  // Scores (0-100) and weighted total (llmfit scoring.rs).
  f.components.speed = std::min(100.0, f.estimated_tps / uc.speed_target * 100.0);
  const double z = std::max(0.0, (ratio - 0.70) / 0.20);
  f.components.fit = f.fit_level == FitLevel::kTooTight ? 0.0 : 100.0 * std::exp(-0.5 * z * z);
  const int64_t ctx_have = vla ? uc.context_target : std::max<int64_t>(spec.text.max_position_embeddings, e.context_tokens);
  f.components.context = ctx_have >= uc.context_target ? 100 : ctx_have >= uc.context_target / 2 ? 70 : 30;
  f.components.quality = std::clamp(quality_base(f.active_params_b, cfg) + lookup(cfg.quality_penalty, qkey, 0) +
                                        family_bonus_for(spec, cfg) + recency_bonus_for(spec, cfg) +
                                        task_bonus_for(spec, uc_name, f.active_params_b, cfg),
                                    0.0, 100.0);
  f.score = std::round((uc.w_quality * f.components.quality + uc.w_speed * f.components.speed +
                        uc.w_fit * f.components.fit + uc.w_context * f.components.context) * 10.0) / 10.0;

  // Notes (llmfit "notes").
  if (!chosen) f.notes.push_back(fmt::format("no quant in the hierarchy fits in {:.1f} GiB VRAM", vram_avail));
  if (f.fitting_quants.size() > 1)
    f.notes.push_back(fmt::format("also fits: {}", fmt::join(f.fitting_quants.begin() + 1, f.fitting_quants.end(), ", ")));
  if (spec.checkpoint_quant)
    f.notes.push_back(fmt::format("checkpoint is already quantized ({} {}); estimates re-quantize from logical params",
                                  spec.checkpoint_quant->method, spec.checkpoint_quant->algo));
  if (f.is_moe)
    f.notes.push_back(fmt::format("MoE: {} experts, {} active; speed uses {:.2f}B active params", spec.text.n_experts,
                                  spec.text.experts_per_tok, f.active_params_b));
  for (const auto& r : e.reasons)
    if (r.find("native kernel") != std::string::npos || r.find("natively") != std::string::npos) f.notes.push_back(r);
  (void)vram_total;
  f.detail = e;
  return f;
}

void rank(std::vector<ModelFit>& fits, std::string_view sort_by) {
  auto key = [&](const ModelFit& f) -> double {
    if (sort_by == "tps") return f.estimated_tps;
    if (sort_by == "params") return f.params_b;
    if (sort_by == "mem") return -f.memory_required_gb;
    return f.score;
  };
  std::stable_sort(fits.begin(), fits.end(), [&](const ModelFit& a, const ModelFit& b) {
    const bool ta = a.fit_level == FitLevel::kTooTight, tb = b.fit_level == FitLevel::kTooTight;
    if (ta != tb) return tb;
    return key(a) > key(b);
  });
}

Json to_json(const ModelFit& f) {
  return {{"model", f.model},
          {"family", f.family},
          {"modality", f.modality},
          {"params_b", f.params_b},
          {"active_params_b", f.active_params_b},
          {"is_moe", f.is_moe},
          {"best_quant", f.best_quant},
          {"fitting_quants", f.fitting_quants},
          {"run_mode", to_string(f.run_mode)},
          {"fit_level", to_string(f.fit_level)},
          {"memory_required_gb", f.memory_required_gb},
          {"memory_available_gb", f.memory_available_gb},
          {"utilization_pct", f.utilization_pct},
          {"estimated_tps", f.estimated_tps},
          {"speed_unit", f.speed_unit},
          {"prefill_tps", f.prefill_tps ? Json(*f.prefill_tps) : Json(nullptr)},
          {"ttft_ms", f.ttft_ms ? Json(*f.ttft_ms) : Json(nullptr)},
          {"context", f.context},
          {"estimate_confidence", to_string(f.confidence)},
          {"score", f.score},
          {"score_components",
           {{"quality", f.components.quality}, {"speed", f.components.speed}, {"fit", f.components.fit}, {"context", f.components.context}}},
          {"notes", f.notes},
          {"detail", estimator::to_json(f.detail)}};
}

Json to_json(const System& s) {
  return {{"name", s.name},         {"backend", s.backend},       {"gpu_count", s.gpu_count},
          {"vram_gb", s.vram_gb},   {"vram_free_gb", s.vram_free_gb}, {"ram_gb", s.ram_gb},
          {"mem_bw_gbps", s.mem_bw_gbps}, {"peak_tflops_bf16", s.peak_tflops_bf16},
          {"unified_memory", s.unified_memory}, {"native_formats", s.native_formats}};
}

}  // namespace mqde::fit
