#include <algorithm>

#include <gtest/gtest.h>

#include "mqde/fit/fit.h"

using namespace mqde;
using namespace mqde::fit;

namespace {

// Dense LLM of `params_b` billion params in BF16, llama-like geometry (32 layers, 8 kv heads, 128 dim).
modelspec::ModelSpec dense_llm(double params_b) {
  Json cfg = {{"model_type", "llama"},
              {"num_hidden_layers", 32},
              {"hidden_size", 4096},
              {"num_attention_heads", 32},
              {"num_key_value_heads", 8},
              {"head_dim", 128},
              {"intermediate_size", 14336},
              {"vocab_size", 128256},
              {"max_position_embeddings", 131072}};
  const int64_t rows = 4096;
  const int64_t cols = static_cast<int64_t>(params_b * 1e9 / rows);
  std::vector<modelspec::TensorInfo> t = {
      {"model.layers.0.mlp.up_proj.weight", "BF16", {rows, cols}, rows * cols, static_cast<uint64_t>(rows * cols * 2), "m"}};
  auto fam = modelspec::find_model_family("llama");
  EXPECT_TRUE(fam);
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  EXPECT_TRUE(spec);
  spec->path = "test/dense";
  return *spec;
}

System gpu24() {
  System s;
  s.name = "Test 24G";
  s.backend = "cuda";
  s.vram_gb = 24;
  s.ram_gb = 64;
  s.mem_bw_gbps = 1008;
  s.peak_tflops_bf16 = 165;
  s.native_formats = {"bf16", "fp8_e4m3", "int8", "int4"};
  return s;
}

}  // namespace

TEST(fit, config_loads_llmfit_constants) {
  auto c = load_fit_config();
  ASSERT_TRUE(c) << c.error().to_string();
  EXPECT_DOUBLE_EQ(c->th_perfect, 0.60);
  EXPECT_DOUBLE_EQ(c->th_good, 0.85);
  EXPECT_DOUBLE_EQ(c->th_marginal, 0.98);
  EXPECT_DOUBLE_EQ(c->bw_efficiency, 0.55);
  EXPECT_DOUBLE_EQ(c->run_mode_factor.at("cpu_only"), 0.3);
  ASSERT_TRUE(c->use_cases.contains("vla"));
  EXPECT_EQ(c->quant_hierarchy.front(), "bf16");
}

TEST(fit, small_model_is_perfect_on_bf16) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  auto f = analyze(dense_llm(1.0), gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(f) << f.error().to_string();
  EXPECT_EQ(f->best_quant, "BF16");
  EXPECT_EQ(f->fit_level, FitLevel::kPerfect);
  EXPECT_EQ(f->run_mode, RunMode::kGpu);
  // llmfit speed: 1008 / (1.0 * 2.0) * 0.55 ~= 277 tok/s
  EXPECT_NEAR(f->estimated_tps, 1008.0 / 2.0 * 0.55, 1.0);
}

TEST(fit, walks_hierarchy_to_smaller_quant) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  // 20B: BF16 (~40 GB) does not fit in 24 GB; FP8 (~20 GB, ratio ~0.9) is the best quant that fits (Marginal).
  auto f = analyze(dense_llm(20.0), gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(f);
  EXPECT_EQ(f->best_quant, "FP8");
  EXPECT_EQ(f->fit_level, FitLevel::kMarginal);
  EXPECT_NEAR(f->estimated_tps, 1008.0 / (20.0 * 1.0) * 0.55, 0.5);
  // 40B: only 4-bit and below fit; llmfit validation point 4090 + 27B Q4 ~ 40 tok/s.
  auto g = analyze(dense_llm(27.0), gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(g);
  EXPECT_EQ(g->best_quant, "W4A16");
  EXPECT_NEAR(g->estimated_tps, 40.0, 3.0);
}

TEST(fit, too_big_goes_to_offload_and_ranks_last) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  // 200B: even W2 (~50 GB) exceeds 24 GB VRAM, but VRAM + 64 GB RAM holds it -> CPU offload.
  auto big = analyze(dense_llm(200.0), gpu24(), *c, estimator::BackendProfile{}, Request{});
  auto small = analyze(dense_llm(1.0), gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(big && small);
  EXPECT_EQ(big->run_mode, RunMode::kCpuOffload);
  EXPECT_LT(big->estimated_tps, small->estimated_tps);
  std::vector<ModelFit> v = {*big, *small};
  rank(v);
  EXPECT_EQ(v.front().params_b, small->params_b);
}

TEST(fit, cpu_only_caps_perfect_at_good) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  System cpu;
  cpu.name = "cpu";
  cpu.backend = "cpu_x86";
  cpu.ram_gb = 256;
  auto f = analyze(dense_llm(1.0), cpu, *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(f);
  EXPECT_EQ(f->run_mode, RunMode::kCpuOnly);
  EXPECT_EQ(f->fit_level, FitLevel::kGood);
  // fallback K: 70 / 1.0 * 0.6 (bf16) * 0.3 (cpu_only)
  EXPECT_NEAR(f->estimated_tps, 70.0 * 0.6 * 0.3, 0.01);
}

TEST(fit, measured_speed_overrides_estimate) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  Request r;
  MeasuredTps m;
  m.tok_s = 123.0;
  m.sample_count = 3;
  m.hardware_label = "this machine";
  r.measured = m;
  auto f = analyze(dense_llm(1.0), gpu24(), *c, estimator::BackendProfile{}, r);
  ASSERT_TRUE(f);
  EXPECT_DOUBLE_EQ(f->estimated_tps, 123.0);
  EXPECT_EQ(f->confidence, Confidence::kMeasuredLocal);
  EXPECT_GT(f->formula_tps, 0.0);
  EXPECT_NE(f->formula_tps, 123.0) << "the formula estimate is kept alongside the measurement";
}

// ---- T-R17: llmfit score calibration (docs/REFERENCES.md A.1-A.6) ----

TEST(fit, quality_tiers_match_llmfit_strict_bounds) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  // (active params B, expected base) — strict "<" upper bounds.
  const std::vector<std::pair<double, double>> cases = {
      // dense_llm() truncates columns, so "N B" is slightly under N·1e9; use N+0.01 to land just above a bound.
      {0.9, 30}, {1.01, 45}, {3.01, 60}, {7.01, 75}, {10.01, 82}, {20.01, 89}, {40.01, 95}, {70.0, 95}};
  const double family = c->family_bonus.at("llama");  // dense_llm() is model_type=llama
  for (const auto& [b, want] : cases) {
    auto f = analyze(dense_llm(b), gpu24(), *c, estimator::BackendProfile{}, Request{});
    ASSERT_TRUE(f) << b;
    // test/dense path does not exist on disk -> no recency bonus; general use case -> no task bonus.
    std::string q = f->best_quant;
    std::transform(q.begin(), q.end(), q.begin(), ::tolower);
    const double pen = c->quality_penalty.contains(q) ? c->quality_penalty.at(q) : 0;
    EXPECT_DOUBLE_EQ(f->components.quality, want + pen + family) << b << "B " << f->best_quant;
  }
}

TEST(fit, w2_penalty_is_llmfit_q2k) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  EXPECT_DOUBLE_EQ(c->quality_penalty.at("w2a16"), -12);
}

TEST(fit, embedding_use_case_exists) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  ASSERT_TRUE(c->use_cases.contains("embedding"));
  const auto& e = c->use_cases.at("embedding");
  EXPECT_DOUBLE_EQ(e.w_quality, 0.30);
  EXPECT_DOUBLE_EQ(e.w_speed, 0.40);
  EXPECT_DOUBLE_EQ(e.speed_target, 200);
  EXPECT_EQ(e.context_target, 512);
}

TEST(fit, ttft_is_null_without_compute_figure) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  auto sys = gpu24();
  sys.peak_tflops_bf16 = 0;
  auto f = analyze(dense_llm(1.0), sys, *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(f);
  EXPECT_FALSE(f->ttft_ms.has_value());
  EXPECT_FALSE(f->prefill_tps.has_value());
  EXPECT_TRUE(to_json(*f)["ttft_ms"].is_null());
  // With compute known it is present.
  auto g = analyze(dense_llm(1.0), gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(g && g->ttft_ms.has_value());
}

TEST(fit, many_cores_bonus_on_k_constant_path) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  System cpu;
  cpu.name = "cpu";
  cpu.backend = "cpu_x86";
  cpu.ram_gb = 256;
  cpu.cpu_cores = 4;
  auto few = analyze(dense_llm(1.0), cpu, *c, estimator::BackendProfile{}, Request{});
  cpu.cpu_cores = 16;
  auto many = analyze(dense_llm(1.0), cpu, *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(few && many);
  EXPECT_NEAR(many->estimated_tps / few->estimated_tps, 1.1, 1e-9);
}

// ---- T-R18: MoE two-tier decode model (docs/REFERENCES.md A.5) ----

namespace {
// dense_llm geometry turned into a Mixtral-like MoE: 8 experts, top-2, expert FFN 14336.
modelspec::ModelSpec moe_llm(double params_b, bool with_expert_width = true, const char* model_type = "mixtral") {
  auto s = dense_llm(params_b);
  s.model_type = model_type;
  s.text.n_experts = 8;
  s.text.experts_per_tok = 2;
  s.text.moe_intermediate = with_expert_width ? 14336 : 0;
  return s;
}
}  // namespace

TEST(fit, moe_decomposition_matches_llmfit_formula) {
  auto s = moe_llm(4.0);
  auto d = moe_decomposition(s);
  ASSERT_TRUE(d);
  const double H = 4096, L = 32, hd = 128, V = 128256;
  EXPECT_NEAR(d->active_ffn_b, L * 2 * 3 * H * 14336 / 1e9, 1e-9);
  const double attn = L * (2 * 32 * hd * H + 2 * 8 * hd * H);
  EXPECT_NEAR(d->fixed_b, (attn + L * 8 * H + 2 * V * H) / 1e9, 1e-9);
  s.text.moe_intermediate = 0;
  EXPECT_FALSE(moe_decomposition(s)) << "missing expert width -> Tier-2";
  EXPECT_FALSE(moe_decomposition(dense_llm(4.0))) << "dense model has no decomposition";
}

TEST(fit, moe_tier1_speed_formula) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  auto s = moe_llm(4.0);  // BF16 ~8 GB on 24 GB -> util ~0.33, below the pressure threshold
  auto f = analyze(s, gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(f) << f.error().to_string();
  ASSERT_EQ(f->run_mode, RunMode::kGpu);
  EXPECT_EQ(f->moe_tier, 1);
  EXPECT_DOUBLE_EQ(f->moe_vram_pressure, 1.0);
  auto d = moe_decomposition(s);
  const double want = 1008.0 / (d->active_ffn_b * 2.0 + d->fixed_b * 3.2);  // BF16 memory bpp 2.0, K=3.2
  EXPECT_NEAR(f->estimated_tps, want, 1e-6);
}

TEST(fit, moe_tier2_overhead_table) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  const auto& mc = c->moe;
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(8, mc), 0.90);
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(16, mc), 0.85);
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(32, mc), 0.80);
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(60, mc), 0.70);
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(64, mc), 0.70);
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(128, mc), 0.40);
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(256, mc), 0.40);
  EXPECT_DOUBLE_EQ(moe_tier2_overhead(0, mc), 0.60) << "unknown";
}

TEST(fit, moe_tier2_speed_formula_and_arch_calibration) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  // Generic arch: default efficiency 0.55, overhead by expert count (8 -> 0.90).
  auto f = analyze(moe_llm(4.0, false), gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(f);
  EXPECT_EQ(f->moe_tier, 2);
  EXPECT_NEAR(f->estimated_tps, 1008.0 / (f->active_params_b * 2.0) * 0.55 * 0.90, 1e-6);
  // deepseek_v3: per-arch calibration (0.62, 0.70) replaces both factors.
  auto g = analyze(moe_llm(4.0, false, "deepseek_v3"), gpu24(), *c, estimator::BackendProfile{}, Request{});
  ASSERT_TRUE(g);
  EXPECT_EQ(g->moe_tier, 2);
  EXPECT_NEAR(g->estimated_tps, 1008.0 / (g->active_params_b * 2.0) * 0.62 * 0.70, 1e-6);
}

TEST(fit, moe_vram_pressure_boundaries) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  const auto& mc = c->moe;
  const double inactive = 1.0 - 8.0 / 256.0;  // 256-expert top-8 MoE geometry
  EXPECT_DOUBLE_EQ(moe_vram_pressure(0.59, 256, 8, mc), 1.0) << "below threshold";
  EXPECT_DOUBLE_EQ(moe_vram_pressure(0.60, 256, 8, mc), 1.0) << "at threshold";
  EXPECT_NEAR(moe_vram_pressure(0.80, 256, 8, mc), 1.0 - 0.20 * inactive, 1e-12);
  EXPECT_NEAR(moe_vram_pressure(1.00, 256, 8, mc), 1.0 - 0.40 * inactive, 1e-12);
  EXPECT_DOUBLE_EQ(moe_vram_pressure(1.20, 256, 8, mc), 1.0) << "util > 1 cannot be GPU mode";
  EXPECT_NEAR(moe_vram_pressure(0.80, 0, 0, mc), 1.0 - 0.20 * 0.50, 1e-12) << "unknown experts -> default ratio";
  // With the default threshold (0.60) the factor bottoms out at 1 - 0.4 * inactive >= 0.6, so the
  // 0.30 floor is only reachable with a lower threshold.
  MoeConfig low = mc;
  low.vram_pressure_threshold = 0.0;
  EXPECT_DOUBLE_EQ(moe_vram_pressure(0.95, 256, 8, low), 0.30);
}

TEST(fit, moe_pressure_applied_in_gpu_mode) {
  auto c = load_fit_config();
  ASSERT_TRUE(c);
  auto f = analyze(moe_llm(8.0), gpu24(), *c, estimator::BackendProfile{}, Request{});  // BF16 ~16 GB -> util ~0.67
  ASSERT_TRUE(f);
  ASSERT_EQ(f->run_mode, RunMode::kGpu);
  ASSERT_EQ(f->best_quant, "BF16");
  const double util = f->params_b * 2.0 / 24.0;
  EXPECT_NEAR(f->moe_vram_pressure, 1.0 - (util - 0.60) * 0.75, 1e-9);
  EXPECT_LT(f->moe_vram_pressure, 1.0);
}
