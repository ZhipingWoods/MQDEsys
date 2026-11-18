#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

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

// T-A: a profile may carry a live-probed free-memory reading (populated by the server, never by the
// estimator itself, which stays a pure function); when present it, not the static datasheet mem_bytes,
// is what capacity/feasibility/concurrency are computed from -- someone else's process already using
// half the card must shrink what this estimate thinks is available.
TEST(estimator_memory, live_free_bytes_overrides_the_static_datasheet_total_when_present) {
  auto spec = tiny_spec();
  auto r = load_recipe("bf16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.context_tokens = 1000;
  BackendProfile be;
  be.runtime_overhead_bytes = 0;
  be.allocator_factor = 1.0;

  HardwareProfile hw = gpu(1ull << 34);  // 16 GiB datasheet total
  const auto e_static = estimate(spec, *r, hw, be, rt);
  EXPECT_EQ(e_static.capacity_bytes,
           static_cast<uint64_t>(static_cast<double>(hw.mem_bytes) * rt.gpu_memory_utilization))
      << "no live reading: capacity comes from the static total, unchanged from before this field existed";

  hw.live_free_bytes = 1ull << 30;  // another process (or job) left only 1 GiB actually free right now
  const auto e_live = estimate(spec, *r, hw, be, rt);
  EXPECT_EQ(e_live.capacity_bytes,
           static_cast<uint64_t>(static_cast<double>(hw.live_free_bytes) * rt.gpu_memory_utilization))
      << "a live reading present: capacity comes from what is actually free, not the datasheet total";
  EXPECT_LT(e_live.capacity_bytes, e_static.capacity_bytes);
  // with only 1 GiB really free, max_concurrency must reflect that squeeze, not the datasheet-sized answer
  EXPECT_LT(e_live.max_concurrency, e_static.max_concurrency);

  hw.live_free_bytes = 0;  // 0 means "not probed" (the field's documented default), not "zero free bytes"
  const auto e_back_to_static = estimate(spec, *r, hw, be, rt);
  EXPECT_EQ(e_back_to_static.capacity_bytes, e_static.capacity_bytes);
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
  auto bes = list_data_entries("backends");
  ASSERT_TRUE(bes);
  for (const auto& n : *bes) EXPECT_TRUE(load_backend(n)) << n;
  auto m = match_hardware("NVIDIA GeForce RTX 4090");
  ASSERT_TRUE(m);
  EXPECT_EQ(m->id, "nvidia/rtx4090");
}

// T-1307: hardware page detail view links out to the vendor doc every real entry's "source" field names.
TEST(estimator_recipes, hardware_profile_source_url_round_trips_through_to_json) {
  auto hw = load_hardware("nvidia/rtx4090");
  ASSERT_TRUE(hw);
  EXPECT_EQ(hw->source.rfind("https://", 0), 0u) << hw->source;
  EXPECT_EQ(to_json(*hw).at("source").get<std::string>(), hw->source);
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

// ---- ADR-0003: batched-engine activation model
// -------------------------------------------------------------------

namespace {

// Qwen2.5-0.5B geometry (the model vLLM reported its KV capacity for): 24 layers, hidden 896, 14 heads / 2 KV
// heads.
modelspec::ModelSpec qwen05b() {
  Json cfg = {{"model_type", "qwen2"},      {"num_hidden_layers", 24},  {"hidden_size", 896},
              {"num_attention_heads", 14},  {"num_key_value_heads", 2}, {"head_dim", 64},
              {"intermediate_size", 4864},  {"vocab_size", 151936},     {"max_position_embeddings", 32768},
              {"tie_word_embeddings", true}};
  std::vector<modelspec::TensorInfo> t;
  const auto add = [&t](const std::string& n, int64_t r, int64_t c) {
    t.push_back({n, "BF16", {r, c}, r * c, static_cast<uint64_t>(r * c * 2), "m"});
  };
  add("model.embed_tokens.weight", 151936, 896);
  for (int l = 0; l < 24; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    add(p + "self_attn.q_proj.weight", 896, 896);
    add(p + "self_attn.k_proj.weight", 128, 896);
    add(p + "self_attn.v_proj.weight", 128, 896);
    add(p + "self_attn.o_proj.weight", 896, 896);
    add(p + "mlp.gate_proj.weight", 4864, 896);
    add(p + "mlp.up_proj.weight", 4864, 896);
    add(p + "mlp.down_proj.weight", 896, 4864);
  }
  auto fam = modelspec::find_model_family("qwen2");
  EXPECT_TRUE(fam);
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  EXPECT_TRUE(spec) << (spec ? "" : std::string(spec.error().message));
  return *spec;
}

BackendProfile batched(int64_t max_tokens = 8192) {
  BackendProfile b;
  b.activation_model = "batched";
  b.max_batched_tokens = max_tokens;
  b.runtime_overhead_bytes = 0;
  b.allocator_factor = 1.0;
  return b;
}

}  // namespace

TEST(estimator_batched, defaults_keep_the_per_sequence_model) {
  const BackendProfile b;
  EXPECT_EQ(b.activation_model, "per_seq");
  auto spec = qwen05b();
  auto r = load_recipe("bf16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.context_tokens = 1024;
  rt.concurrency = 4;
  BackendProfile explicit_per_seq = b;
  explicit_per_seq.activation_model = "per_seq";
  const auto a = estimate(spec, *r, gpu(1ull << 34), b, rt);
  const auto c = estimate(spec, *r, gpu(1ull << 34), explicit_per_seq, rt);
  EXPECT_EQ(a.activation_bytes, c.activation_bytes);
  EXPECT_EQ(a.max_concurrency, c.max_concurrency);
  rt.concurrency = 8;
  EXPECT_EQ(estimate(spec, *r, gpu(1ull << 34), b, rt).activation_bytes,
            2 * estimate(spec, *r, gpu(1ull << 34), b,
                         [&] {
                           auto x = rt;
                           x.concurrency = 4;
                           return x;
                         }())
                    .activation_bytes)
      << "per_seq activations grow linearly with concurrency";
}

TEST(estimator_batched, weights_and_kv_are_unchanged_only_activations_and_concurrency_move) {
  auto spec = qwen05b();
  auto r = load_recipe("w4a16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.context_tokens = 1024;
  rt.concurrency = 16;
  const auto hf = estimate(spec, *r, gpu(24ull << 30), BackendProfile{}, rt);
  const auto vl = estimate(spec, *r, gpu(24ull << 30), batched(), rt);
  EXPECT_EQ(hf.weights_bytes, vl.weights_bytes);
  EXPECT_EQ(hf.qmeta_bytes, vl.qmeta_bytes);
  EXPECT_EQ(hf.kv_bytes, vl.kv_bytes);
  EXPECT_LT(vl.activation_bytes, hf.activation_bytes);
  EXPECT_GT(vl.max_concurrency, hf.max_concurrency);
}

TEST(estimator_batched, activation_is_charged_once_and_capped_by_max_batched_tokens) {
  auto spec = qwen05b();
  auto r = load_recipe("bf16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.context_tokens = 1024;
  const auto act = [&](int64_t conc, int64_t cap) {
    rt.concurrency = conc;
    return estimate(spec, *r, gpu(1ull << 40), batched(cap), rt).activation_bytes;
  };
  // below the cap the cost per sequence is constant (tokens x per-token bytes + logits), so it is linear in
  // concurrency
  EXPECT_EQ(act(4, 1 << 30), 4 * act(1, 1 << 30));
  // at the cap it stops growing apart from the per-sequence logits row
  const double logits = 151936.0 * 4.0;
  EXPECT_NEAR(static_cast<double>(act(64, 8192) - act(8, 8192)), 56.0 * logits, 1.0);
  // a larger cap costs more, a smaller one less (monotone in max_batched_tokens)
  EXPECT_LT(act(64, 4096), act(64, 8192));
  EXPECT_LT(act(64, 8192), act(64, 16384));
  // one token of context and concurrency 1: tokens x (2 x intermediate + 6 x hidden) x 2 bytes + logits, by
  // hand
  rt.context_tokens = 1;
  EXPECT_EQ(act(1, 8192), static_cast<uint64_t>((2.0 * 4864 + 6.0 * 896) * 2.0 + logits));
}

TEST(estimator_batched, max_concurrency_is_the_largest_count_that_fits) {
  auto spec = qwen05b();
  auto r = load_recipe("w4a16");
  ASSERT_TRUE(r);
  RuntimeParams rt;
  rt.context_tokens = 1024;
  const auto hw = gpu(24ull << 30);
  const auto base = estimate(spec, *r, hw, batched(), rt);
  const int64_t n = base.max_concurrency;
  ASSERT_GT(n, 10);
  rt.concurrency = n;
  EXPECT_TRUE(estimate(spec, *r, hw, batched(), rt).feasible) << "n sequences fit";
  rt.concurrency = n + 1;
  EXPECT_FALSE(estimate(spec, *r, hw, batched(), rt).feasible) << "n + 1 sequences do not fit";
  const auto tiny = estimate(spec, *r, gpu(1ull << 20), batched(), rt);
  EXPECT_EQ(tiny.max_concurrency, 0);
}

TEST(estimator_batched, regression_against_what_vllm_reported) {
  // vLLM 0.30 on an RTX 4090, Qwen2.5-0.5B AWQ, gpu_memory_utilization 0.3: it logged a capacity of 547.03
  // sequences at 1024 tokens and 17.09 at 32768 tokens (docs/PROGRESS.md 2026-10-08). The estimator with the
  // repository's own vllm backend profile and no fitted coefficient must land within [0.5, 1.5] of that; the
  // per-sequence model, which is what these profiles used before, was 7 to 8.5 times too low.
  auto spec = qwen05b();
  auto r = load_recipe("w4a16");
  auto hw = load_hardware("nvidia/rtx4090");
  auto be = load_backend("vllm");
  ASSERT_TRUE(r && hw && be);
  EXPECT_EQ(be->activation_model, "batched");
  RuntimeParams rt;
  rt.gpu_memory_utilization = 0.3;
  struct Case {
    int64_t ctx;
    double measured;
  };
  for (const auto& c : {Case{1024, 547.03}, Case{32768, 17.09}}) {
    rt.context_tokens = c.ctx;
    const auto e = estimate(spec, *r, *hw, *be, rt);
    const double ratio = static_cast<double>(e.max_concurrency) / c.measured;
    EXPECT_GE(ratio, 0.5) << "ctx " << c.ctx << ": estimated " << e.max_concurrency;
    EXPECT_LE(ratio, 1.5) << "ctx " << c.ctx << ": estimated " << e.max_concurrency;
    auto old = *be;
    old.activation_model = "per_seq";
    EXPECT_LT(static_cast<double>(estimate(spec, *r, *hw, old, rt).max_concurrency) / c.measured, 0.2)
        << "the old model really was far off";
  }
}

TEST(estimator_batched, backend_profile_parsing) {
  auto vllm = load_backend("vllm");
  ASSERT_TRUE(vllm);
  EXPECT_EQ(vllm->max_batched_tokens, 8192);
  auto hf = load_backend("hf_transformers");
  ASSERT_TRUE(hf);
  EXPECT_EQ(hf->activation_model, "per_seq") << "profiles that do not mention it keep the old behaviour";
  const auto dir = std::filesystem::temp_directory_path() / "mqde_bad_backend.yaml";
  std::ofstream(dir) << "id: x\nactivation_model: weird\n";
  EXPECT_FALSE(load_backend(dir.string()));
  std::ofstream(dir) << "id: x\nactivation_model: batched\nmax_batched_tokens: 0\n";
  EXPECT_FALSE(load_backend(dir.string()));
}
