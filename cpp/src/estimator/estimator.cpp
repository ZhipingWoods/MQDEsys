#include "mqde/estimator/estimator.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace mqde::estimator {
namespace {

using modelspec::ModelSpec;
using quant::NumFormat;
using quant::QuantSpec;

double elem_bytes(const QuantSpec& s, double fallback) {
  return s.enabled() ? static_cast<double>(s.bits) / 8.0 : fallback;
}

std::string format_name(const QuantSpec& s) {
  if (!s.enabled()) return "bf16";
  if (s.fmt == NumFormat::kFp8E4M3) return "fp8_e4m3";
  if (s.fmt == NumFormat::kFp8E5M2) return "fp8_e5m2";
  return fmt::format("int{}", s.bits);
}

// Bytes of scale/zero metadata for one [rows, cols] weight.
double qmeta_bytes(const QuantSpec& s, int64_t rows, int64_t cols) {
  double groups = 1;
  if (s.gran == quant::Granularity::kPerChannel) groups = static_cast<double>(rows);
  if (s.gran == quant::Granularity::kPerGroup)
    groups =
        static_cast<double>(rows) * std::ceil(static_cast<double>(cols) / static_cast<double>(s.group_size));
  const double scale_b = s.fmt == NumFormat::kInt ? 2.0 : 4.0;  // fp16 scales for INT, fp32 for FP8
  const double zero_b = (s.fmt == NumFormat::kInt && !s.symmetric) ? static_cast<double>(s.bits) / 8.0 : 0.0;
  return groups * (scale_b + zero_b);
}

bool is_embedding(const std::string& name) {
  return name.find("embed_tokens") != std::string::npos ||
         name.find("position_embedding") != std::string::npos ||
         name.find("future_tokens") != std::string::npos || name.find("patch_embedding") != std::string::npos;
}

struct Shape {
  int64_t vis_tokens = 0;  // patch tokens entering the vision encoder
  int64_t ctx = 0;         // language-model context tokens
  int64_t action_tokens = 0;
  int64_t action_steps = 1;
};

Shape runtime_shape(const ModelSpec& spec, const RuntimeParams& rt) {
  Shape sh;
  int64_t per_image = 0;
  if (spec.vision && spec.vision->patch_size > 0) {
    const int64_t side = rt.image_resolution / spec.vision->patch_size;
    sh.vis_tokens = side * side * rt.images_per_step;
    per_image = spec.vision->tokens_per_image > 0 ? spec.vision->tokens_per_image : side * side / 4;
  }
  sh.ctx = rt.context_tokens > 0 ? rt.context_tokens : per_image * rt.images_per_step + rt.text_tokens;
  if (spec.action) {
    sh.action_tokens = spec.action->query_tokens + spec.action->horizon;
    sh.action_steps = std::max<int64_t>(1, spec.action->inference_steps);
  }
  return sh;
}

// Tokens each component processes per decision step (drives FLOPs / weight re-reads).
double component_tokens(const std::string& comp, const Shape& sh) {
  if (comp == "vision") return static_cast<double>(sh.vis_tokens);
  if (comp == "merger") return static_cast<double>(sh.vis_tokens) / 4.0;
  if (comp == "language") return static_cast<double>(sh.ctx);
  if (comp == "action_head") return static_cast<double>(sh.action_tokens * sh.action_steps);
  if (comp == "lm_head") return 1.0;
  return 0.0;
}

}  // namespace

Estimate estimate(const ModelSpec& spec, const Recipe& recipe, const HardwareProfile& hw,
                  const BackendProfile& be, const RuntimeParams& rt) {
  Estimate e;
  e.model = spec.path.empty() ? spec.model_type : spec.path;
  e.recipe = recipe.name;
  e.hardware = hw.id;
  e.backend = be.id;
  e.rt = rt;
  const Shape sh = runtime_shape(spec, rt);
  e.context_tokens = sh.ctx;

  std::set<std::string> quant_modalities(recipe.modalities.begin(), recipe.modalities.end());
  quant_modalities.insert(rt.extra_modalities.begin(), rt.extra_modalities.end());

  // ---- weights ----
  std::vector<ComponentMemory> comps;
  auto comp_of = [&](const std::string& name, const std::string& modality) -> ComponentMemory& {
    for (auto& c : comps)
      if (c.name == name) return c;
    comps.push_back({.name = name, .modality = modality});
    return comps.back();
  };
  double flops = 0, bytes_read = 0, qmeta_total = 0;
  for (size_t i = 0; i < spec.tensors.size(); ++i) {
    const auto& t = spec.tensors[i];
    const auto& u = spec.units[i];
    auto& c = comp_of(u.component, u.modality);
    // Estimates are "re-quantize from the logical 16-bit model": packed checkpoint tensors are expanded to
    // their logical size and existing scale/zero tensors are dropped (the recipe adds its own qmeta).
    if (u.aux) continue;
    const int64_t numel = u.logical_numel > 0 ? u.logical_numel : t.numel;
    const bool packed = numel != t.numel || t.dtype.starts_with("F8");
    const double orig = packed ? static_cast<double>(numel) * 2.0 : static_cast<double>(t.nbytes);
    const auto& shape = u.logical_shape.empty() ? t.shape : u.logical_shape;
    c.params += numel;
    c.original_bytes += static_cast<uint64_t>(orig);
    const bool quantize = u.quantizable && recipe.weight.enabled() && quant_modalities.contains(u.modality);
    double bytes = orig;
    if (quantize) {
      const double qm = qmeta_bytes(recipe.weight, shape[0], shape[1]);
      bytes = std::ceil(static_cast<double>(numel) * elem_bytes(recipe.weight, 2.0));
      c.qmeta_bytes += static_cast<uint64_t>(qm);
      qmeta_total += qm;
      bytes += qm;
      c.quantized_params += numel;
      c.quantized = true;
    }
    c.weight_bytes += static_cast<uint64_t>(bytes);
    // Matmul work: 2-D weights (embeddings are lookups); category-specific 3-D weights use one slice.
    const double tokens = component_tokens(u.component, sh);
    // MoE routed experts: only experts_per_tok / n_experts of them are touched per token.
    const double moe_frac =
        (t.name.find(".experts.") != std::string::npos && spec.text.n_experts > 0 &&
         spec.text.experts_per_tok > 0)
            ? static_cast<double>(spec.text.experts_per_tok) / static_cast<double>(spec.text.n_experts)
            : 1.0;
    if (t.name.ends_with(".weight") && !is_embedding(t.name) && shape.size() >= 2) {
      const double eff_params =
          shape.size() == 3 ? static_cast<double>(numel / shape[0]) : static_cast<double>(numel);
      flops += 2.0 * eff_params * tokens * moe_frac;
    }
    if (tokens > 0) {
      // Category-specific (per-embodiment) 3-D weights: only one slice is touched per step.
      const double touched = (shape.size() == 3 ? bytes / static_cast<double>(shape[0]) : bytes) * moe_frac;
      const double rereads = u.component == "action_head" ? static_cast<double>(sh.action_steps) : 1.0;
      bytes_read += touched * rereads;
    }
  }
  for (const auto& c : comps) {
    e.weights_bytes += c.weight_bytes;
    e.original_weights_bytes += c.original_bytes;
  }
  e.qmeta_bytes = static_cast<uint64_t>(qmeta_total);
  e.components = comps;

  // ---- KV cache + linear-attention state (per sequence) ----
  const auto& tx = spec.text;
  const double kv_b = elem_bytes(recipe.kv, 2.0);
  const double kv_tok_full =
      static_cast<double>(tx.n_full_attention) * 2.0 * tx.n_kv_heads * tx.head_dim * kv_b;
  const double kv_tok_slide =
      static_cast<double>(tx.n_sliding_attention) * 2.0 * tx.n_kv_heads * tx.head_dim * kv_b;
  const int64_t slide_ctx = tx.sliding_window > 0 ? std::min(sh.ctx, tx.sliding_window) : sh.ctx;
  e.kv_bytes_per_token = static_cast<uint64_t>(kv_tok_full + kv_tok_slide);
  if (tx.n_linear_attention > 0) {
    const double state_b = tx.linear_state_dtype == "float32" ? 4.0 : 2.0;
    const double conv_dim = 2.0 * tx.linear_num_key_heads * tx.linear_key_head_dim +
                            static_cast<double>(tx.linear_num_value_heads) * tx.linear_value_head_dim;
    const double per_layer =
        static_cast<double>(tx.linear_num_value_heads) * tx.linear_key_head_dim * tx.linear_value_head_dim *
            state_b +
        conv_dim * static_cast<double>(std::max<int64_t>(tx.linear_conv_kernel - 1, 0)) * 2.0;
    e.state_bytes_per_seq = static_cast<uint64_t>(per_layer * static_cast<double>(tx.n_linear_attention));
  }
  const double kv_seq = kv_tok_full * static_cast<double>(sh.ctx) +
                        kv_tok_slide * static_cast<double>(slide_ctx) +
                        static_cast<double>(e.state_bytes_per_seq);
  e.kv_bytes = static_cast<uint64_t>(kv_seq * static_cast<double>(rt.concurrency));
  bytes_read += kv_tok_full * static_cast<double>(sh.ctx) * static_cast<double>(rt.concurrency);

  // ---- activations: stages run sequentially, so the peak is the largest stage ----
  const double act_b = 2.0;
  double act_vision = 0, act_lang = 0, act_action = 0;
  if (spec.vision)
    act_vision =
        static_cast<double>(sh.vis_tokens) * (spec.vision->intermediate + 4.0 * spec.vision->hidden) * act_b;
  act_lang =
      static_cast<double>(sh.ctx) *
          (2.0 * tx.intermediate + 6.0 * tx.hidden + static_cast<double>(tx.n_layers + 1) * tx.hidden) *
          act_b +
      static_cast<double>(tx.vocab) * 4.0;  // hidden-state outputs retained + last-token logits (fp32)
  if (spec.action) {
    const double width = 768.0;  // DiT inner dim of the MiniCPM action head; refined by trace data later
    act_action = (static_cast<double>(sh.action_tokens) * (4.0 * width + 4.0 * width) +
                  static_cast<double>(sh.ctx) * 2.0 * width) *
                 4.0;
  }
  e.activation_bytes = static_cast<uint64_t>(std::max({act_vision, act_lang, act_action}) *
                                             static_cast<double>(rt.concurrency));
  e.runtime_bytes = be.runtime_overhead_bytes;

  const double f = be.allocator_factor;
  e.peak_bytes =
      static_cast<uint64_t>((static_cast<double>(e.weights_bytes) + static_cast<double>(e.kv_bytes) +
                             static_cast<double>(e.activation_bytes)) *
                                f +
                            static_cast<double>(e.runtime_bytes));
  e.capacity_bytes = static_cast<uint64_t>(static_cast<double>(hw.mem_bytes) * rt.gpu_memory_utilization);
  e.feasible = e.peak_bytes <= e.capacity_bytes;

  const double per_seq = (kv_seq + std::max({act_vision, act_lang, act_action})) * f;
  const double budget = static_cast<double>(e.capacity_bytes) - static_cast<double>(e.runtime_bytes) -
                        static_cast<double>(e.weights_bytes) * f;
  e.max_concurrency = (budget > 0 && per_seq > 0) ? static_cast<int64_t>(std::floor(budget / per_seq)) : 0;

  // ---- roofline floor per decision step ----
  e.compute_format = recipe.act.enabled() ? format_name(recipe.act) : "bf16";
  const std::string peak_key = e.compute_format.starts_with("fp8") ? "fp8" : e.compute_format;
  const double peak = hw.peak_tflops.contains(peak_key) ? hw.peak_tflops.at(peak_key)
                      : hw.peak_tflops.contains("bf16") ? hw.peak_tflops.at("bf16")
                                                        : 0.0;
  e.flops_per_step = flops * static_cast<double>(rt.concurrency);
  e.bytes_per_step = bytes_read;
  const double t_compute = peak > 0 ? e.flops_per_step / (peak * 1e12 * be.mfu) : 0.0;
  const double t_memory = hw.mem_bw_gbps > 0 ? e.bytes_per_step / (hw.mem_bw_gbps * 1e9 * be.eta_bw) : 0.0;
  e.latency_floor_ms = std::max(t_compute, t_memory) * 1e3;

  // ---- verdict ----
  if (!e.feasible)
    e.reasons.push_back(
        fmt::format("peak {} exceeds usable capacity {} ({}% of {})", format_bytes(double(e.peak_bytes)),
                    format_bytes(double(e.capacity_bytes)), static_cast<int>(rt.gpu_memory_utilization * 100),
                    format_bytes(double(hw.mem_bytes))));
  const auto wfmt = format_name(recipe.weight);
  if (recipe.weight.enabled() &&
      std::find(hw.native_formats.begin(), hw.native_formats.end(), wfmt) == hw.native_formats.end())
    e.reasons.push_back(
        fmt::format("{} weights have no native kernel on {} ({}); memory savings need a custom kernel, "
                    "accuracy can still be evaluated with fake quantization",
                    wfmt, hw.name, hw.arch));
  if (recipe.act.enabled()) {
    const auto afmt = format_name(recipe.act);
    if (std::find(hw.native_formats.begin(), hw.native_formats.end(), afmt) == hw.native_formats.end())
      e.reasons.push_back(
          fmt::format("{} activations are not natively supported on {} ({})", afmt, hw.name, hw.arch));
  }
  if (recipe.maturity == "experimental")
    e.reasons.push_back(fmt::format("{} is an experimental recipe", recipe.name));
  for (const auto& c : e.components)
    if (!c.quantized && c.modality != "other" && recipe.weight.enabled() &&
        !quant_modalities.contains(c.modality) && c.params > 0)
      e.reasons.push_back(fmt::format("component '{}' ({}) kept in original precision", c.name, c.modality));

  e.assumptions = {
      fmt::format("context = {} tokens ({} image(s) x {} tokens + {} text tokens)", sh.ctx,
                  rt.images_per_step, spec.vision ? spec.vision->tokens_per_image : 0, rt.text_tokens),
      "KV cache counts full-attention layers only; linear-attention layers hold a constant recurrent + conv "
      "state",
      "activation peak = largest of vision / language / action stages (they run sequentially); SDPA/flash "
      "attention assumed",
      fmt::format("runtime overhead {} and allocator factor {:.2f} from backend '{}'",
                  format_bytes(double(be.runtime_overhead_bytes)), be.allocator_factor, be.id),
      fmt::format("latency floor = max(FLOPs / ({} peak x MFU {:.2f}), bytes / (BW x eta {:.2f})); "
                  "kernel-launch overhead not modeled",
                  e.compute_format, be.mfu, be.eta_bw),
  };
  e.confidence = "low";  // becomes medium/high once measured feedback (S15) exists for (hw, backend, recipe)
  return e;
}

std::vector<CurvePoint> memory_vs_concurrency(const ModelSpec& s, const Recipe& r, const HardwareProfile& h,
                                              const BackendProfile& b, RuntimeParams rt,
                                              const std::vector<int64_t>& xs) {
  std::vector<CurvePoint> out;
  for (auto x : xs) {
    rt.concurrency = x;
    auto e = estimate(s, r, h, b, rt);
    out.push_back({x, e.peak_bytes, e.kv_bytes, e.feasible});
  }
  return out;
}

std::vector<CurvePoint> memory_vs_images(const ModelSpec& s, const Recipe& r, const HardwareProfile& h,
                                         const BackendProfile& b, RuntimeParams rt,
                                         const std::vector<int64_t>& xs) {
  std::vector<CurvePoint> out;
  for (auto x : xs) {
    rt.images_per_step = x;
    rt.context_tokens = 0;
    auto e = estimate(s, r, h, b, rt);
    out.push_back({x, e.peak_bytes, e.kv_bytes, e.feasible});
  }
  return out;
}

Json to_json(const Estimate& e) {
  Json comps = Json::array();
  for (const auto& c : e.components)
    comps.push_back({{"name", c.name},
                     {"modality", c.modality},
                     {"params", c.params},
                     {"original_bytes", c.original_bytes},
                     {"weight_bytes", c.weight_bytes},
                     {"qmeta_bytes", c.qmeta_bytes},
                     {"quantized_params", c.quantized_params},
                     {"quantized", c.quantized}});
  return {{"apiVersion", "mqde/v1"},
          {"kind", "EstimateReport"},
          {"model", e.model},
          {"recipe", e.recipe},
          {"hardware", e.hardware},
          {"backend", e.backend},
          {"runtime",
           {{"concurrency", e.rt.concurrency},
            {"images_per_step", e.rt.images_per_step},
            {"text_tokens", e.rt.text_tokens},
            {"image_resolution", e.rt.image_resolution},
            {"gpu_memory_utilization", e.rt.gpu_memory_utilization}}},
          {"context_tokens", e.context_tokens},
          {"components", comps},
          {"breakdown",
           {{"weights", e.weights_bytes - e.qmeta_bytes},
            {"qmeta", e.qmeta_bytes},
            {"kv_cache", e.kv_bytes},
            {"activation", e.activation_bytes},
            {"runtime", e.runtime_bytes}}},
          {"original_weights_bytes", e.original_weights_bytes},
          {"kv_bytes_per_token", e.kv_bytes_per_token},
          {"state_bytes_per_seq", e.state_bytes_per_seq},
          {"peak_bytes", e.peak_bytes},
          {"capacity_bytes", e.capacity_bytes},
          {"feasible", e.feasible},
          {"max_concurrency", e.max_concurrency},
          {"reasons", e.reasons},
          {"perf",
           {{"flops_per_step", e.flops_per_step},
            {"bytes_per_step", e.bytes_per_step},
            {"latency_floor_ms", e.latency_floor_ms},
            {"compute_format", e.compute_format}}},
          {"assumptions", e.assumptions},
          {"confidence", e.confidence}};
}

Json to_json(const std::vector<CurvePoint>& c) {
  Json arr = Json::array();
  for (const auto& p : c)
    arr.push_back(
        {{"x", p.x}, {"peak_bytes", p.peak_bytes}, {"kv_bytes", p.kv_bytes}, {"feasible", p.feasible}});
  return arr;
}

}  // namespace mqde::estimator
