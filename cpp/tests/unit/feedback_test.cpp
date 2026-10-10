// T-0905: the feedback loop (S15) that turns a real measurement into a CalibrationStore update.
//
// There is no real multi-model hardware data available in this environment, so the acceptance criterion
// ("after 3 different models, the 4th model's throughput prediction is within 25%") is verified against
// SYNTHETIC ground truth: a chosen (eta_bw, mfu) pair plays the role of "what the real GPU does",
// "measurements" are computed from it with estimate_perf, and the test checks that feeding 3 of them through
// calibrate() produces a backend profile whose prediction for a 4th, different model is close to what that
// same ground truth would have given it directly. This proves the calibration machinery converges correctly;
// it does not prove any particular real GPU's eta_bw or mfu.
#include <gtest/gtest.h>
#include <unistd.h>

#include <cmath>
#include <filesystem>

#include "mqde/feedback/feedback.h"

using namespace mqde;
using namespace mqde::estimator;
using namespace mqde::feedback;

namespace {

std::unique_ptr<storage::Db> fresh_db(const char* name) {
  auto d = std::filesystem::temp_directory_path() /
           (std::string("mqde_feedback_") + name + "_" + std::to_string(::getpid()));
  std::filesystem::remove_all(d);
  auto db = storage::Db::open(d / "m.db");
  EXPECT_TRUE(db) << (db ? "" : db.error().message);
  return std::move(*db);
}

// A dense llama-family model of roughly `params_b` billion parameters, same construction the other estimator
// tests use (one real weight tensor sized to the target parameter count; estimate() only needs the aggregate
// byte counts).
modelspec::ModelSpec dense(double params_b, int64_t hidden = 4096) {
  Json cfg = {{"model_type", "llama"},
              {"num_hidden_layers", 32},
              {"hidden_size", hidden},
              {"num_attention_heads", 32},
              {"num_key_value_heads", 8},
              {"head_dim", hidden / 32},
              {"intermediate_size", hidden * 7 / 2},
              {"vocab_size", 128256},
              {"max_position_embeddings", 131072}};
  const int64_t cols = static_cast<int64_t>(params_b * 1e9 / hidden);
  std::vector<modelspec::TensorInfo> t = {{"model.layers.0.mlp.up_proj.weight",
                                           "BF16",
                                           {hidden, cols},
                                           hidden * cols,
                                           static_cast<uint64_t>(hidden * cols * 2),
                                           "m"}};
  auto fam = modelspec::find_model_family("llama");
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  EXPECT_TRUE(spec) << (spec ? "" : std::string(spec.error().message));
  spec->path = "test/dense";
  return *spec;
}

HardwareProfile gpu() {
  HardwareProfile h;
  h.id = "test/gpu";
  h.name = "T";
  h.arch = "x";
  h.mem_bytes = 1ull << 40;
  h.mem_bw_gbps = 1000;
  h.peak_tflops = {{"bf16", 150}};
  h.native_formats = {"bf16"};
  return h;
}

BackendProfile defaults_be() {
  BackendProfile b;
  b.eta_bw = 0.6;
  b.mfu = 0.35;
  b.runtime_overhead_bytes = 500ull << 20;
  b.allocator_factor = 1.0;
  return b;
}

storage::Calibration defaults_cal(const BackendProfile& b) {
  storage::Calibration c;
  c.eta_bw = b.eta_bw;
  c.mfu = b.mfu;
  c.overhead_bytes = static_cast<double>(b.runtime_overhead_bytes);
  return c;
}

Estimate est_of(const modelspec::ModelSpec& spec, const HardwareProfile& hw, const BackendProfile& be,
                int64_t ctx, int64_t conc = 1) {
  auto r = load_recipe("bf16");
  RuntimeParams rt;
  rt.context_tokens = ctx;
  rt.concurrency = conc;
  return estimate(spec, *r, hw, be, rt);
}

const storage::CalibrationKey kKey{"test/gpu", "test-backend", "BF16"};

}  // namespace

TEST(feedback_calibrate, an_observation_with_nothing_measured_changes_nothing) {
  auto db = fresh_db("empty_obs");
  storage::CalibrationStore store(*db);
  const auto be = defaults_be();
  const auto e = est_of(dense(3), gpu(), be, 4096);
  auto r = calibrate(store, kKey, defaults_cal(be), e, gpu(), be, {4096, 128}, {});
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_TRUE(r->applied.empty());
  EXPECT_EQ(r->calibration.samples, 0);
  EXPECT_DOUBLE_EQ(r->calibration.eta_bw, be.eta_bw);
}

TEST(feedback_calibrate, a_memory_bound_decode_measurement_updates_eta_bw_only) {
  auto db = fresh_db("eta_only");
  storage::CalibrationStore store(*db);
  const auto be = defaults_be();
  const auto hw = gpu();
  const PerfParams p{1024, 128};
  const auto e = est_of(dense(3), hw, be, 1024, 1);
  const auto truth = estimate_perf(e, hw,
                                   BackendProfile{.runtime_overhead_bytes = be.runtime_overhead_bytes,
                                                  .allocator_factor = 1.0,
                                                  .eta_bw = 0.5,
                                                  .mfu = 0.35},
                                   p);
  ASSERT_EQ(truth.decode_bound, "memory")
      << "the test needs a memory-bound decode to exercise eta_bw inference";
  auto r =
      calibrate(store, kKey, defaults_cal(be), e, hw, be, p,
                Observation{.peak_bytes = std::nullopt, .ttft_ms = std::nullopt, .tpot_ms = truth.tpot_ms});
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_EQ(r->applied, (std::vector<std::string>{"eta_bw"}));
  EXPECT_TRUE(r->skipped.empty()) << (r->skipped.empty() ? std::string() : r->skipped[0]);
  EXPECT_NEAR(r->calibration.eta_bw, 0.7 * be.eta_bw + 0.3 * 0.5,
              1e-9);  // CalibrationStore's own default alpha = 0.3
  EXPECT_DOUBLE_EQ(r->calibration.mfu, be.mfu) << "mfu was not observed, so it keeps the default";
}

TEST(feedback_calibrate,
     a_compute_bound_measurement_on_the_wrong_roof_is_skipped_with_a_reason_not_an_error) {
  auto db = fresh_db("wrong_roof");
  storage::CalibrationStore store(*db);
  const auto be = defaults_be();
  const auto hw = gpu();
  // a short prompt's prefill is memory-bound (confirmed in estimator_perf_test.cpp); asking infer_mfu of it
  // must be refused
  const PerfParams p{16, 8};
  const auto e = est_of(dense(3), hw, be, 16, 1);
  const auto pr = estimate_perf(e, hw, be, p);
  ASSERT_EQ(pr.prefill_bound, "memory");
  auto r = calibrate(store, kKey, defaults_cal(be), e, hw, be, p,
                     Observation{.peak_bytes = std::nullopt, .ttft_ms = pr.ttft_ms, .tpot_ms = std::nullopt});
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_TRUE(r->applied.empty());
  ASSERT_EQ(r->skipped.size(), 1u);
  EXPECT_NE(r->skipped[0].find("mfu:"), std::string::npos);
  EXPECT_NE(r->skipped[0].find("memory-bound"), std::string::npos) << r->skipped[0];
  EXPECT_EQ(r->calibration.samples, 0) << "a skipped observation is not a sample";
}

TEST(feedback_calibrate, a_compute_bound_decode_measurement_skips_eta_bw_with_a_reason) {
  auto db = fresh_db("wrong_roof_eta");
  storage::CalibrationStore store(*db);
  const auto be = defaults_be();
  // A device with very low peak_tflops relative to its bandwidth, at high concurrency, makes decode
  // compute-bound (same construction as estimator_perf_test.cpp's
  // inference_refuses_when_the_measured_phase_is_bound test).
  HardwareProfile weak = gpu();
  weak.peak_tflops = {{"bf16", 1.0}};
  const PerfParams p{128, 64};
  const auto e = est_of(dense(20.0), weak, be, 128, 2048);
  const auto pr = estimate_perf(e, weak, be, p);
  ASSERT_EQ(pr.decode_bound, "compute");
  auto r = calibrate(store, kKey, defaults_cal(be), e, weak, be, p,
                     Observation{.peak_bytes = std::nullopt, .ttft_ms = std::nullopt, .tpot_ms = pr.tpot_ms});
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_TRUE(r->applied.empty());
  ASSERT_EQ(r->skipped.size(), 1u);
  EXPECT_NE(r->skipped[0].find("eta_bw:"), std::string::npos) << r->skipped[0];
  EXPECT_NE(r->skipped[0].find("compute-bound"), std::string::npos) << r->skipped[0];
  EXPECT_EQ(r->calibration.samples, 0);
}

TEST(feedback_calibrate, overhead_bytes_is_inferred_from_a_measured_peak) {
  auto db = fresh_db("overhead");
  storage::CalibrationStore store(*db);
  const auto be = defaults_be();
  const auto hw = gpu();
  const auto e = est_of(dense(3), hw, be, 2048, 2);
  const double modelled = static_cast<double>(e.weights_bytes + e.kv_bytes + e.activation_bytes);
  const double true_overhead = 700e6;
  auto r = calibrate(
      store, kKey, defaults_cal(be), e, hw, be, {2048, 64},
      Observation{.peak_bytes = modelled + true_overhead, .ttft_ms = std::nullopt, .tpot_ms = std::nullopt});
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_EQ(r->applied, (std::vector<std::string>{"overhead_bytes"}));
  EXPECT_NEAR(r->calibration.overhead_bytes,
              0.7 * static_cast<double>(be.runtime_overhead_bytes) + 0.3 * true_overhead, 1.0);
}

TEST(
    feedback_calibrate,
    three_models_then_a_fourth_the_calibrated_throughput_prediction_beats_25_percent_and_the_uncalibrated_one_does_not) {
  auto db = fresh_db("converge");
  storage::CalibrationStore store(*db);
  const auto be = defaults_be();  // eta 0.6, mfu 0.35: the defaults, deliberately off from "truth" below
  const auto hw = gpu();
  const double eta_true = 0.40,
               mfu_true =
                   0.18;  // deliberately far from the defaults (0.6, 0.35), like a real first guess would be
  const BackendProfile truth{.id = "truth",
                             .runtime_overhead_bytes = be.runtime_overhead_bytes,
                             .allocator_factor = 1.0,
                             .eta_bw = eta_true,
                             .mfu = mfu_true};
  const PerfParams p{2048, 256};  // a context long enough that prefill is compute-bound and decode stays
                                  // memory-bound (checked below)

  storage::Calibration cal = defaults_cal(be);
  for (double params_b :
       {3.0, 8.0, 13.0}) {  // 3 "training" models, different sizes, same (hw, backend, recipe) key
    const auto spec = dense(params_b);
    const auto e = est_of(spec, hw, be, p.prompt_tokens, 1);
    const auto measured = estimate_perf(e, hw, truth, p);
    ASSERT_EQ(measured.decode_bound, "memory") << params_b;
    ASSERT_EQ(measured.prefill_bound, "compute") << params_b;
    auto r = calibrate(
        store, kKey, defaults_cal(be), e, hw, be, p,
        Observation{.peak_bytes = std::nullopt, .ttft_ms = measured.ttft_ms, .tpot_ms = measured.tpot_ms});
    ASSERT_TRUE(r) << r.error().message;
    EXPECT_EQ(r->applied.size(), 2u) << "both eta_bw and mfu should be inferable at this context/concurrency";
    cal = r->calibration;
  }
  EXPECT_LT(std::abs(cal.eta_bw - eta_true), std::abs(be.eta_bw - eta_true))
      << "3 observations moved eta_bw toward the truth";
  EXPECT_LT(std::abs(cal.mfu - mfu_true), std::abs(be.mfu - mfu_true))
      << "3 observations moved mfu toward the truth";

  // 4th, different model: compare the calibrated backend's prediction against what the true coefficients
  // would give it
  const auto spec4 = dense(20.0, 5120);
  const auto e4 = est_of(spec4, hw, be, p.prompt_tokens, 1);
  const auto want = estimate_perf(e4, hw, truth, p);
  const auto calibrated_be = backend_with_calibration(be, cal);
  const auto got_calibrated = estimate_perf(e4, hw, calibrated_be, p);
  const auto got_uncalibrated = estimate_perf(e4, hw, be, p);
  const double err_calibrated =
      std::abs(got_calibrated.tokens_per_s_total - want.tokens_per_s_total) / want.tokens_per_s_total;
  const double err_uncalibrated =
      std::abs(got_uncalibrated.tokens_per_s_total - want.tokens_per_s_total) / want.tokens_per_s_total;
  EXPECT_LE(err_calibrated, 0.25) << "calibrated relative error " << err_calibrated;
  EXPECT_GT(err_uncalibrated, 0.25) << "the uncalibrated defaults should NOT already satisfy this bound "
                                    << "(otherwise the test proves nothing): relative error "
                                    << err_uncalibrated;
  EXPECT_LT(err_calibrated, err_uncalibrated)
      << "calibration must actually help, not just coincidentally pass";
}

TEST(feedback_calibrate, propagates_a_store_error_instead_of_silently_losing_the_measurement) {
  // version-pin conflict: pre-seed the row at a version the store's internal transaction won't match after a
  // concurrent external write, forcing CalibrationStore::observe's transaction to retry/fail is out of scope
  // here; instead exercise the simpler, real error path: an out-of-range inferred value is refused by
  // CalibrationStore. infer_mfu(e, hw, be, p, measured_ttft) = flops / (peak * mfu_unit * measured_ttft), so
  // an implausibly fast reported ttft (not one this model could really produce) inverts to an mfu far past 1.
  auto db = fresh_db("bad_infer");
  storage::CalibrationStore store(*db);
  const BackendProfile be = defaults_be();
  const auto hw = gpu();
  const auto e = est_of(dense(50.0), hw, be, 4096, 1);
  const auto p = PerfParams{4096, 16};
  const auto real = estimate_perf(e, hw, be, p);
  ASSERT_EQ(real.prefill_bound, "compute");
  // 20x faster than the default-mfu prediction infers mfu ~= be.mfu * 20 (~7): past (0, 1.5], but still
  // slower than the memory floor, so infer_mfu rejects it for being out of range, not for looking
  // memory-bound.
  const double implausibly_fast_ttft_ms = real.ttft_ms / 20.0;
  auto bad_mfu = infer_mfu(e, hw, be, p, implausibly_fast_ttft_ms);
  ASSERT_TRUE(bad_mfu) << bad_mfu.error().message;
  ASSERT_GT(*bad_mfu, 1.5)
      << "the fixture does not actually produce an out-of-range value; adjust the factor above";
  auto r = calibrate(
      store, kKey, defaults_cal(be), e, hw, be, p,
      Observation{.peak_bytes = std::nullopt, .ttft_ms = implausibly_fast_ttft_ms, .tpot_ms = std::nullopt});
  ASSERT_FALSE(r) << "an inferred mfu far outside (0, 1.5] must surface as an error, not vanish";
  EXPECT_NE(r.error().message.find("mfu"), std::string::npos) << r.error().message;
}

TEST(feedback_backend_with_calibration, copies_exactly_the_three_coefficients_calibration_tracks) {
  BackendProfile be = defaults_be();
  be.id = "vllm";
  storage::Calibration cal;
  cal.eta_bw = 0.71;
  cal.mfu = 0.42;
  cal.overhead_bytes = 123456789.0;
  const auto got = backend_with_calibration(be, cal);
  EXPECT_EQ(got.id, "vllm") << "everything else about the backend profile is untouched";
  EXPECT_DOUBLE_EQ(got.allocator_factor, be.allocator_factor);
  EXPECT_DOUBLE_EQ(got.eta_bw, 0.71);
  EXPECT_DOUBLE_EQ(got.mfu, 0.42);
  EXPECT_EQ(got.runtime_overhead_bytes, 123456789u);
}
