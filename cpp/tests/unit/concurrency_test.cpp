#include <gtest/gtest.h>

#include "mqde/estimator/estimator.h"
#include "mqde/fit/concurrency.h"
#include "mqde/modelspec/model_spec.h"

using namespace mqde;
using namespace mqde::fit;

namespace {

// 24 GB GPU hardware profile matching RTX 4090.
estimator::HardwareProfile gpu24() {
  estimator::HardwareProfile h;
  h.id = "nvidia/rtx4090";
  h.mem_bytes = 24ull << 30;
  h.mem_bw_gbps = 1008;
  h.peak_tflops["bf16"] = 165;
  return h;
}

estimator::BackendProfile hf_backend() {
  return {};  // defaults: runtime_overhead=512MB, allocator_factor=1.05
}

// Dense LLM spec built via build_model_spec (matching fit_test.cpp::dense_llm).
modelspec::ModelSpec tiny_dense(double params_b) {
  nlohmann::json cfg = {
      {"model_type", "llama"},      {"num_hidden_layers", 32},  {"hidden_size", 4096},
      {"num_attention_heads", 32},  {"num_key_value_heads", 8}, {"head_dim", 128},
      {"intermediate_size", 14336}, {"vocab_size", 128256},     {"max_position_embeddings", 131072}};
  const int64_t rows = 4096;
  const int64_t cols = static_cast<int64_t>(params_b * 1e9 / rows);
  std::vector<modelspec::TensorInfo> t = {{"model.layers.0.mlp.up_proj.weight",
                                           "BF16",
                                           {rows, cols},
                                           rows * cols,
                                           static_cast<uint64_t>(rows * cols * 2),
                                           "m"}};
  auto fam = modelspec::find_model_family("llama");
  if (!fam) throw std::runtime_error("llama family not found");
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  if (!spec) throw std::runtime_error("build_model_spec failed: " + std::string(spec.error().message));
  spec->path = "test/tiny";
  return *spec;
}

}  // namespace

TEST(concurrency, single_user_fits_small_model) {
  auto s = tiny_dense(7.0);
  auto hw = gpu24();
  auto be = hf_backend();
  auto c = estimate_concurrency(s, "bf16", hw, be, 1, 4096);
  EXPECT_GE(c.max_users, 1) << "a 7B BF16 model should fit at least 1 user on 24 GB";
  EXPECT_TRUE(c.fits);
  EXPECT_GT(c.weights_gb, 0);
  EXPECT_GE(c.kv_per_user_gb, 0);
  EXPECT_GT(c.capacity_gb, 0);
  EXPECT_NEAR(c.utilization_pct, c.total_gb / c.capacity_gb * 100.0, 1e-6);
}

TEST(concurrency, max_users_plus_one_does_not_fit) {
  auto s = tiny_dense(7.0);
  auto hw = gpu24();
  auto be = hf_backend();
  auto c = estimate_concurrency(s, "bf16", hw, be, 1, 4096);
  const int64_t max_u = c.max_users;
  ASSERT_GT(max_u, 0);

  // At max_users it should fit.
  auto at_max = estimate_concurrency(s, "bf16", hw, be, max_u, 4096);
  EXPECT_TRUE(at_max.fits) << "max_users=" << max_u << " should still fit";

  // At max_users + 1 it should not fit (or kv_per_user_gb is 0 meaning no KV cost).
  if (c.kv_per_user_gb > 0) {
    auto one_over = estimate_concurrency(s, "bf16", hw, be, max_u + 1, 4096);
    EXPECT_FALSE(one_over.fits) << "max_users+1=" << (max_u + 1) << " should not fit";
  }
}

TEST(concurrency, high_pressure_flag) {
  auto s = tiny_dense(7.0);
  auto hw = gpu24();
  auto be = hf_backend();
  // Request many users to force high utilization.
  auto c = estimate_concurrency(s, "bf16", hw, be, 1000, 4096);
  if (c.utilization_pct > 90.0) {
    EXPECT_TRUE(c.high_pressure);
  } else {
    EXPECT_FALSE(c.high_pressure);
  }
}

TEST(concurrency, max_context_for_users_is_power_of_two) {
  auto s = tiny_dense(7.0);
  auto hw = gpu24();
  auto be = hf_backend();
  auto c = estimate_concurrency(s, "bf16", hw, be, 1, 4096);
  const int64_t mctx = c.max_context_for_users;
  if (mctx > 0) {
    // Must be a power of two.
    EXPECT_EQ(mctx & (mctx - 1), 0) << "max_context_for_users must be a power of two, got " << mctx;
    // At this context the model should fit.
    auto at_ctx = estimate_concurrency(s, "bf16", hw, be, 1, mctx);
    EXPECT_TRUE(at_ctx.fits);
  }
}

TEST(concurrency, weights_gb_consistent_with_estimator) {
  // The weights_gb from estimate_concurrency should be close to the weights_bytes from estimate().
  auto s = tiny_dense(7.0);
  auto hw = gpu24();
  auto be = hf_backend();
  estimator::RuntimeParams rt;
  rt.concurrency = 1;
  rt.context_tokens = 4096;
  auto recipe = estimator::load_recipe("bf16");
  ASSERT_TRUE(recipe);
  auto e = estimator::estimate(s, *recipe, hw, be, rt);

  auto c = estimate_concurrency(s, "bf16", hw, be, 1, 4096);
  const double e_weights_gb = static_cast<double>(e.weights_bytes) / (1ull << 30);
  EXPECT_NEAR(c.weights_gb, e_weights_gb, 0.01)
      << "weights_gb should match estimator::estimate().weights_bytes";
}

TEST(concurrency, to_json_fields_present) {
  auto s = tiny_dense(7.0);
  auto c = estimate_concurrency(s, "bf16", gpu24(), hf_backend(), 2, 4096);
  auto j = to_json(c);
  EXPECT_TRUE(j.contains("requested_users"));
  EXPECT_TRUE(j.contains("max_users"));
  EXPECT_TRUE(j.contains("context"));
  EXPECT_TRUE(j.contains("weights_gb"));
  EXPECT_TRUE(j.contains("kv_per_user_gb"));
  EXPECT_TRUE(j.contains("overhead_gb"));
  EXPECT_TRUE(j.contains("total_gb"));
  EXPECT_TRUE(j.contains("capacity_gb"));
  EXPECT_TRUE(j.contains("utilization_pct"));
  EXPECT_TRUE(j.contains("fits"));
  EXPECT_TRUE(j.contains("high_pressure"));
  EXPECT_TRUE(j.contains("max_context_for_users"));
  EXPECT_EQ(j["requested_users"], 2);
}
