// T-0209: roofline performance model (TTFT / TPOT / throughput) and the inversions the feedback loop uses.
#include <gtest/gtest.h>

#include <cmath>

#include "mqde/estimator/estimator.h"

using namespace mqde;
using namespace mqde::estimator;

namespace {

// A dense LLM small enough to compute by hand: 4 layers, hidden 256, 4 heads x 64, intermediate 512, vocab
// 1000, no GQA.
modelspec::ModelSpec toy() {
  Json cfg = {{"model_type", "llama"},    {"num_hidden_layers", 4},   {"hidden_size", 256},
              {"num_attention_heads", 4}, {"num_key_value_heads", 4}, {"head_dim", 64},
              {"intermediate_size", 512}, {"vocab_size", 1000},       {"max_position_embeddings", 4096}};
  std::vector<modelspec::TensorInfo> t;
  auto add = [&t](const std::string& n, int64_t r, int64_t c) {
    t.push_back({n, "BF16", {r, c}, r * c, static_cast<uint64_t>(r * c * 2), "m"});
  };
  add("model.embed_tokens.weight", 1000, 256);
  for (int l = 0; l < 4; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    add(p + "self_attn.q_proj.weight", 256, 256);
    add(p + "self_attn.k_proj.weight", 256, 256);
    add(p + "self_attn.v_proj.weight", 256, 256);
    add(p + "self_attn.o_proj.weight", 256, 256);
    add(p + "mlp.gate_proj.weight", 512, 256);
    add(p + "mlp.up_proj.weight", 512, 256);
    add(p + "mlp.down_proj.weight", 256, 512);
  }
  add("lm_head.weight", 1000, 256);
  auto fam = modelspec::find_model_family("llama");
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  EXPECT_TRUE(spec) << (spec ? "" : std::string(spec.error().message));
  return *spec;
}

HardwareProfile gpu(double bw_gbps, double bf16_tflops) {
  HardwareProfile h;
  h.id = "test/gpu";
  h.name = "T";
  h.arch = "x";
  h.mem_bytes = 1ull << 36;
  h.mem_bw_gbps = bw_gbps;
  h.peak_tflops = {{"bf16", bf16_tflops}};
  h.native_formats = {"bf16"};
  return h;
}

BackendProfile be(double eta, double mfu) {
  BackendProfile b;
  b.eta_bw = eta;
  b.mfu = mfu;
  b.runtime_overhead_bytes = 0;
  b.allocator_factor = 1.0;
  return b;
}

Estimate est(int64_t conc = 1, int64_t ctx = 1024) {
  auto r = load_recipe("bf16");
  RuntimeParams rt;
  rt.concurrency = conc;
  rt.context_tokens = ctx;
  return estimate(toy(), *r, gpu(1000, 100), be(0.6, 0.35), rt);
}

}  // namespace

TEST(estimator_perf, the_model_pieces_match_a_hand_count_of_the_toy_network) {
  const auto e = est();
  // per layer: q,k,v,o = 4 * 256*256 ; gate,up,down = 3 * 512*256 ; 4 layers; FLOPs per token = 2 x params
  const double layer_params = 4.0 * 256 * 256 + 3.0 * 512 * 256;
  EXPECT_DOUBLE_EQ(e.flops_lang_per_token, 2.0 * 4 * layer_params);
  EXPECT_DOUBLE_EQ(e.flops_head_per_token, 2.0 * 1000 * 256);
  EXPECT_DOUBLE_EQ(e.bytes_lang, 2.0 * 4 * layer_params) << "bf16: two bytes per weight";
  EXPECT_DOUBLE_EQ(e.bytes_head, 2.0 * 1000 * 256);
  EXPECT_DOUBLE_EQ(e.attn_flops_per_ctx_token, 4.0 * 4 * 4 * 64);
}

TEST(estimator_perf, tpot_is_memory_bound_at_batch_one_and_equals_bytes_over_effective_bandwidth) {
  const auto e = est(1, 1024);
  PerfParams p{1024, 128};
  const auto r = estimate_perf(e, gpu(1000, 100), be(0.6, 0.35), p);
  EXPECT_EQ(r.decode_bound, "memory");
  const double ctx = 1024 + 64.0;
  const double bytes = e.bytes_lang + e.bytes_head + static_cast<double>(e.kv_bytes_per_token) * ctx +
                       static_cast<double>(e.state_bytes_per_seq);
  EXPECT_NEAR(r.tpot_ms, bytes / (1000e9 * 0.6) * 1e3, 1e-9);
  EXPECT_NEAR(r.tokens_per_s_per_seq, 1000.0 / r.tpot_ms, 1e-9);
}

TEST(estimator_perf,
     ttft_grows_faster_than_linearly_with_the_prompt_once_attention_matters_and_is_compute_bound) {
  const auto e = est(1, 4096);
  const auto hw = gpu(1000, 100);
  const auto short_p = estimate_perf(e, hw, be(0.6, 0.35), {256, 16});
  const auto long_p = estimate_perf(e, hw, be(0.6, 0.35), {4096, 16});
  EXPECT_EQ(long_p.prefill_bound, "compute");
  EXPECT_GT(long_p.ttft_ms / short_p.ttft_ms, 16.0) << "16x the tokens, plus the quadratic attention term";
  // exact value for the long prompt
  const double P = 4096;
  const double flops =
      e.flops_lang_per_token * P + e.attn_flops_per_ctx_token * P * P / 2 + e.flops_head_per_token;
  EXPECT_NEAR(long_p.ttft_ms, flops / (100e12 * 0.35) * 1e3, 1e-9);
}

TEST(estimator_perf, decode_turns_compute_bound_at_a_large_enough_batch_and_throughput_stops_scaling) {
  PerfParams p{128, 128};
  const auto hw = gpu(1000, 4);  // compute-poor, and this toy is KV-heavy (about 0.8 MB per sequence), so the
                                 // crossover needs a weak device
  double prev_total = 0;
  std::string first_bound, last_bound;
  for (int64_t c : {1, 4, 16, 64, 256, 1024, 4096}) {
    const auto r = estimate_perf(est(c, 256), hw, be(0.6, 0.35), p);
    if (c == 1) first_bound = r.decode_bound;
    last_bound = r.decode_bound;
    EXPECT_GE(r.tokens_per_s_total, prev_total * 0.999) << "more sequences never reduce total throughput";
    prev_total = r.tokens_per_s_total;
  }
  EXPECT_EQ(first_bound, "memory");
  EXPECT_EQ(last_bound, "compute");
  const auto a = estimate_perf(est(2048, 256), hw, be(0.6, 0.35), p),
             b = estimate_perf(est(4096, 256), hw, be(0.6, 0.35), p);
  EXPECT_NEAR(b.tokens_per_s_total / a.tokens_per_s_total, 1.0, 0.05)
      << "past the roof, doubling the batch adds almost nothing";
}

TEST(estimator_perf, a_faster_memory_or_a_better_efficiency_lowers_tpot_proportionally) {
  const auto e = est();
  PerfParams p{1024, 128};
  const auto base = estimate_perf(e, gpu(1000, 100), be(0.5, 0.35), p);
  EXPECT_NEAR(estimate_perf(e, gpu(2000, 100), be(0.5, 0.35), p).tpot_ms, base.tpot_ms / 2, 1e-9);
  EXPECT_NEAR(estimate_perf(e, gpu(1000, 100), be(1.0, 0.35), p).tpot_ms, base.tpot_ms / 2, 1e-9);
}

TEST(estimator_perf, request_latency_and_request_rate_follow_from_ttft_and_tpot) {
  const auto e = est(8, 512);
  PerfParams p{512, 101};
  const auto r = estimate_perf(e, gpu(1000, 100), be(0.6, 0.35), p);
  EXPECT_NEAR(r.request_latency_ms, r.ttft_ms + 100 * r.tpot_ms, 1e-9);
  EXPECT_NEAR(r.requests_per_s, 8 * 1000.0 / r.request_latency_ms, 1e-9);
  EXPECT_NEAR(r.tokens_per_s_total, 8 * r.tokens_per_s_per_seq, 1e-9);
}

TEST(estimator_perf, inferring_a_coefficient_from_a_measurement_inverts_the_model) {
  const auto e = est(1, 1024);
  const auto hw = gpu(1000, 100);
  PerfParams p{1024, 128};
  for (double eta : {0.2, 0.45, 0.8}) {
    const auto r = estimate_perf(e, hw, be(eta, 0.35), p);
    auto got = infer_eta_bw(e, hw, be(0.99, 0.35), p, r.tpot_ms);
    ASSERT_TRUE(got) << got.error().message;
    EXPECT_NEAR(*got, eta, 1e-9) << "the backend's own eta is irrelevant for a memory-bound decode";
  }
  for (double mfu : {0.1, 0.3, 0.6}) {
    const auto e2 = est(1, 4096);
    PerfParams p2{4096, 16};
    const auto r = estimate_perf(e2, hw, be(0.6, mfu), p2);
    ASSERT_EQ(r.prefill_bound, "compute");
    auto got = infer_mfu(e2, hw, be(0.6, 0.99), p2, r.ttft_ms);
    ASSERT_TRUE(got) << got.error().message;
    EXPECT_NEAR(*got, mfu, 1e-9);
  }
}

TEST(estimator_perf, inference_refuses_when_the_measured_phase_is_bound_by_the_other_roof) {
  const auto hw = gpu(1000, 100);
  // decode at a huge batch on a weak device is compute-bound: bandwidth cannot be read from it
  const auto weak = gpu(1000, 1);
  const auto big = est(2048, 256);
  const auto r = estimate_perf(big, weak, be(0.6, 0.35), {128, 64});
  ASSERT_EQ(r.decode_bound, "compute");
  auto eta = infer_eta_bw(big, weak, be(0.6, 0.35), {128, 64}, r.tpot_ms);
  ASSERT_FALSE(eta);
  EXPECT_NE(eta.error().message.find("compute-bound"), std::string::npos) << eta.error().message;
  // prefill of a short prompt is memory-bound: MFU cannot be read from it
  const auto e = est(1, 16);
  const auto pr = estimate_perf(e, hw, be(0.6, 0.35), {16, 8});
  ASSERT_EQ(pr.prefill_bound, "memory");
  auto mfu = infer_mfu(e, hw, be(0.6, 0.35), {16, 8}, pr.ttft_ms);
  ASSERT_FALSE(mfu);
  EXPECT_NE(mfu.error().message.find("memory-bound"), std::string::npos);
  EXPECT_FALSE(infer_eta_bw(e, hw, be(0.6, 0.35), {16, 8}, -1));
  EXPECT_FALSE(infer_eta_bw(e, hw, be(0.6, 0.35), {16, 8}, std::nan("")));
  EXPECT_FALSE(infer_mfu(e, hw, be(0.6, 0.35), {16, 8}, 0));
}

TEST(estimator_perf, overhead_is_the_measured_peak_minus_what_the_model_accounts_for) {
  const auto e = est(2, 1024);
  const auto b = be(0.6, 0.35);
  const double modelled = static_cast<double>(e.weights_bytes + e.kv_bytes + e.activation_bytes);
  auto o = infer_overhead_bytes(e, b, modelled + 400e6);
  ASSERT_TRUE(o);
  EXPECT_NEAR(*o, 400e6, 1.0);
  auto too_low = infer_overhead_bytes(e, b, modelled * 0.5);
  ASSERT_FALSE(too_low);
  EXPECT_NE(too_low.error().message.find("below the modelled"), std::string::npos);
  EXPECT_FALSE(infer_overhead_bytes(e, b, 0));
  auto b2 = with_coefficients(b, 0.7, 0.4, 123456.0);
  EXPECT_DOUBLE_EQ(b2.eta_bw, 0.7);
  EXPECT_DOUBLE_EQ(b2.mfu, 0.4);
  EXPECT_EQ(b2.runtime_overhead_bytes, 123456u);
  EXPECT_EQ(with_coefficients(b, 0.7, 0.4, -5).runtime_overhead_bytes, 0u) << "never negative";
}

TEST(estimator_perf, a_tied_embedding_model_pays_for_the_lm_head_on_every_token_and_an_untied_one_does_too) {
  // same toy but tied: no lm_head tensor, the embedding table is the head
  Json cfg = {{"model_type", "llama"},      {"num_hidden_layers", 1},   {"hidden_size", 256},
              {"num_attention_heads", 4},   {"num_key_value_heads", 4}, {"head_dim", 64},
              {"intermediate_size", 512},   {"vocab_size", 1000},       {"max_position_embeddings", 4096},
              {"tie_word_embeddings", true}};
  std::vector<modelspec::TensorInfo> t;
  auto add = [&t](const std::string& n, int64_t r, int64_t c) {
    t.push_back({n, "BF16", {r, c}, r * c, static_cast<uint64_t>(r * c * 2), "m"});
  };
  add("model.embed_tokens.weight", 1000, 256);
  add("model.layers.0.self_attn.q_proj.weight", 256, 256);
  add("model.layers.0.mlp.up_proj.weight", 512, 256);
  auto fam = modelspec::find_model_family("llama");
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  ASSERT_TRUE(spec);
  auto r = load_recipe("bf16");
  RuntimeParams rt;
  rt.context_tokens = 128;
  const auto e = estimate(*spec, *r, gpu(1000, 100), be(0.6, 0.35), rt);
  EXPECT_DOUBLE_EQ(e.flops_head_per_token, 2.0 * 1000 * 256);
  EXPECT_DOUBLE_EQ(e.bytes_head, 2.0 * 1000 * 256);
  EXPECT_DOUBLE_EQ(e.bytes_lang, 2.0 * (256 * 256 + 512 * 256))
      << "the embedding lookup is not counted as layer weights";
}
