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
  r.measured_tps = 123.0;
  r.measured_source = Confidence::kMeasuredLocal;
  auto f = analyze(dense_llm(1.0), gpu24(), *c, estimator::BackendProfile{}, r);
  ASSERT_TRUE(f);
  EXPECT_DOUBLE_EQ(f->estimated_tps, 123.0);
  EXPECT_EQ(f->confidence, Confidence::kMeasuredLocal);
}
