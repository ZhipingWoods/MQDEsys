#include <gtest/gtest.h>

#include "mqde/estimator/estimator.h"

using namespace mqde;
using namespace mqde::estimator;

namespace {

modelspec::ModelSpec tiny_spec() {
  // Single language weight [256, 1024] + the MiniCPM-VLA text geometry.
  Json cfg = {
      {"model_type", "minicpm_vla"},
      {"action_horizon", 30},
      {"num_inference_timesteps", 4},
      {"action_dim", 80},
      {"vlm_config",
       {{"text_config",
         {{"num_hidden_layers", 24},
          {"hidden_size", 1024},
          {"num_attention_heads", 8},
          {"num_key_value_heads", 2},
          {"head_dim", 256},
          {"intermediate_size", 3584},
          {"vocab_size", 1000},
          {"linear_num_key_heads", 16},
          {"linear_num_value_heads", 16},
          {"linear_key_head_dim", 128},
          {"linear_value_head_dim", 128},
          {"linear_conv_kernel_dim", 4},
          {"mamba_ssm_dtype", "float32"}}},
        {"vision_config",
         {{"num_hidden_layers", 1}, {"hidden_size", 64}, {"intermediate_size", 128}, {"patch_size", 14}}}}}};
  std::vector<std::string> types;
  for (int i = 0; i < 24; ++i) types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  cfg["vlm_config"]["text_config"]["layer_types"] = types;
  std::vector<modelspec::TensorInfo> t = {{"vlm.model.language_model.layers.0.mlp.up_proj.weight",
                                           "BF16",
                                           {256, 1024},
                                           256 * 1024,
                                           256 * 1024 * 2,
                                           "m"}};
  auto fam = modelspec::find_model_family("minicpm_vla");
  EXPECT_TRUE(fam);
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  EXPECT_TRUE(spec);
  return *spec;
}

HardwareProfile gpu(uint64_t mem) {
  HardwareProfile h;
  h.id = "test/gpu";
  h.name = "Test GPU";
  h.arch = "sm_89";
  h.mem_bytes = mem;
  h.mem_bw_gbps = 1000;
  h.peak_tflops = {{"bf16", 100.0}, {"int8", 200.0}};
  h.native_formats = {"bf16", "int8", "int4"};
  return h;
}

}  // namespace

TEST(estimator_memory, kv_and_linear_state_hand_computed) {
  auto spec = tiny_spec();
  auto r = load_recipe("bf16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.context_tokens = 100;
  auto e = estimate(spec, *r, gpu(1ull << 34), BackendProfile{}, rt);
  // 6 full-attention layers x K,V x 2 kv heads x 256 dim x 2 bytes
  EXPECT_EQ(e.kv_bytes_per_token, 6u * 2 * 2 * 256 * 2);
  // 18 linear layers x (16 heads x 128 x 128 x fp32 + conv_dim 6144 x (4-1) x 2 bytes)
  EXPECT_EQ(e.state_bytes_per_seq, 18u * (16 * 128 * 128 * 4 + 6144 * 3 * 2));
  EXPECT_EQ(e.kv_bytes, 12288u * 100 + e.state_bytes_per_seq);
  EXPECT_EQ(e.weights_bytes, 256u * 1024 * 2) << "bf16 recipe keeps weights";
}

TEST(estimator_memory, w4a16_weights_and_qmeta) {
  auto spec = tiny_spec();
  auto r = load_recipe("w4a16");  // INT4 asym g128: 0.5 B/elem + per group (2 B scale + 0.5 B zero)
  ASSERT_TRUE(r);
  auto e = estimate(spec, *r, gpu(1ull << 34), BackendProfile{}, RuntimeParams{});
  const uint64_t groups = 256 * (1024 / 128);
  EXPECT_EQ(e.qmeta_bytes, static_cast<uint64_t>(groups * 2.5));
  EXPECT_EQ(e.weights_bytes, 256u * 1024 / 2 + e.qmeta_bytes);
}

TEST(estimator_memory, concurrency_scales_kv_and_feasibility) {
  auto spec = tiny_spec();
  auto r = load_recipe("bf16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.context_tokens = 1000;
  BackendProfile be;
  be.runtime_overhead_bytes = 0;
  be.allocator_factor = 1.0;
  auto e1 = estimate(spec, *r, gpu(1ull << 34), be, rt);
  rt.concurrency = 8;
  auto e8 = estimate(spec, *r, gpu(1ull << 34), be, rt);
  EXPECT_EQ(e8.kv_bytes, 8 * e1.kv_bytes);
  EXPECT_TRUE(e8.feasible);
  EXPECT_GE(e1.max_concurrency, 8);
  auto small = estimate(spec, *r, gpu(1ull << 20), be, rt);
  EXPECT_FALSE(small.feasible);
  EXPECT_FALSE(small.reasons.empty());
  EXPECT_EQ(small.max_concurrency, 0);
}

TEST(estimator_memory, context_from_images) {
  auto spec = tiny_spec();
  auto r = load_recipe("bf16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.images_per_step = 3;
  rt.text_tokens = 20;
  auto e = estimate(spec, *r, gpu(1ull << 34), BackendProfile{}, rt);
  EXPECT_EQ(e.context_tokens, 3 * 64 + 20);
}

TEST(estimator_recipes, all_builtin_recipes_load) {
  auto names = list_data_entries("recipes");
  ASSERT_TRUE(names);
  EXPECT_GE(names->size(), 6u);
  for (const auto& n : *names) EXPECT_TRUE(load_recipe(n)) << n;
  auto hws = list_data_entries("hardware_db");
  ASSERT_TRUE(hws);
  for (const auto& n : *hws) EXPECT_TRUE(load_hardware(n)) << n;
  auto m = match_hardware("NVIDIA GeForce RTX 4090");
  ASSERT_TRUE(m);
  EXPECT_EQ(m->id, "nvidia/rtx4090");
}

TEST(estimator_recipes, w2_warns_on_missing_native_kernel) {
  auto spec = tiny_spec();
  auto r = load_recipe("w2a16");
  ASSERT_TRUE(r);
  auto e = estimate(spec, *r, gpu(1ull << 34), BackendProfile{}, RuntimeParams{});
  bool found = false;
  for (const auto& s : e.reasons) found |= s.find("int2") != std::string::npos;
  EXPECT_TRUE(found);
}
