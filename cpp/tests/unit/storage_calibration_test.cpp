// T-0209: CalibrationStore: initial values, EMA update formula, confidence, validation and concurrent
// updates.
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <thread>

#include "mqde/storage/calibration.h"

namespace fs = std::filesystem;
using namespace mqde::storage;

namespace {
std::unique_ptr<Db> fresh(const char* n) {
  auto d = fs::temp_directory_path() / (std::string("mqde_cal_") + n + "_" + std::to_string(::getpid()));
  fs::remove_all(d);
  auto db = Db::open(d / "m.db");
  EXPECT_TRUE(db);
  return std::move(*db);
}
const CalibrationKey K{"nvidia/rtx4090", "vllm", "w4a16"};
Calibration defaults() {
  Calibration c;
  c.eta_bw = 0.6;
  c.mfu = 0.35;
  c.overhead_bytes = 402653184;
  return c;
}
}  // namespace

TEST(calibration_store, without_history_the_defaults_come_back_with_low_confidence) {
  auto db = fresh("empty");
  CalibrationStore s(*db);
  auto c = s.get(K, defaults());
  ASSERT_TRUE(c);
  EXPECT_EQ(c->samples, 0);
  EXPECT_EQ(c->confidence, "low");
  EXPECT_DOUBLE_EQ(c->eta_bw, 0.6);
  EXPECT_DOUBLE_EQ(c->mfu, 0.35);
  EXPECT_DOUBLE_EQ(c->overhead_bytes, 402653184);
}

TEST(calibration_store, three_observations_give_high_confidence_and_the_exact_ema_values) {
  auto db = fresh("ema");
  CalibrationStore s(*db, 0.3);
  double eta = 0.6, mfu = 0.35, oh = 402653184;
  const double obs_eta[] = {0.40, 0.50, 0.45};
  const double obs_mfu[] = {0.20, 0.25, 0.30};
  const double obs_oh[] = {500e6, 450e6, 480e6};
  const std::string want_conf[] = {"medium", "medium", "high"};
  for (int i = 0; i < 3; ++i) {
    eta = 0.7 * eta + 0.3 * obs_eta[i];
    mfu = 0.7 * mfu + 0.3 * obs_mfu[i];
    oh = 0.7 * oh + 0.3 * obs_oh[i];
    auto c = s.observe(K, defaults(), obs_eta[i], obs_mfu[i], obs_oh[i]);
    ASSERT_TRUE(c) << c.error().message;
    EXPECT_EQ(c->samples, i + 1);
    EXPECT_EQ(c->confidence, want_conf[i]) << "after " << i + 1 << " samples";
    EXPECT_NEAR(c->eta_bw, eta, 1e-12);
    EXPECT_NEAR(c->mfu, mfu, 1e-12);
    EXPECT_NEAR(c->overhead_bytes, oh, 1e-3);
  }
  auto stored = s.get(K, defaults());
  ASSERT_TRUE(stored);
  EXPECT_EQ(stored->confidence, "high");
  EXPECT_NEAR(stored->eta_bw, eta, 1e-12) << "what get() returns is what observe() returned";
}

TEST(calibration_store, repeated_identical_measurements_converge_to_the_measurement) {
  auto db = fresh("conv");
  CalibrationStore s(*db);
  Calibration last;
  for (int i = 0; i < 40; ++i) last = *s.observe(K, defaults(), 0.8, std::nullopt, std::nullopt);
  EXPECT_NEAR(last.eta_bw, 0.8, 1e-5);
  EXPECT_DOUBLE_EQ(last.mfu, 0.35) << "a coefficient that was never observed keeps its default";
  EXPECT_EQ(last.samples, 40);
}

TEST(calibration_store, keys_are_independent) {
  auto db = fresh("keys");
  CalibrationStore s(*db);
  ASSERT_TRUE(s.observe(K, defaults(), 0.9, 0.9, 1e6));
  const CalibrationKey other_hw{"nvidia/a100-80g", "vllm", "w4a16"},
      other_backend{"nvidia/rtx4090", "hf_transformers", "w4a16"},
      other_recipe{"nvidia/rtx4090", "vllm", "fp8"};
  for (const auto& k : {other_hw, other_backend, other_recipe}) {
    auto c = s.get(k, defaults());
    ASSERT_TRUE(c);
    EXPECT_EQ(c->samples, 0) << k.hardware << "/" << k.backend << "/" << k.recipe;
    EXPECT_DOUBLE_EQ(c->eta_bw, 0.6);
  }
}

TEST(calibration_store, bad_observations_are_rejected_and_change_nothing) {
  auto db = fresh("bad");
  CalibrationStore s(*db);
  EXPECT_FALSE(s.observe(K, defaults(), std::nullopt, std::nullopt, std::nullopt));
  EXPECT_FALSE(s.observe(K, defaults(), 0.0, std::nullopt, std::nullopt));
  EXPECT_FALSE(s.observe(K, defaults(), -0.1, std::nullopt, std::nullopt));
  EXPECT_FALSE(s.observe(K, defaults(), 7.0, std::nullopt, std::nullopt));
  EXPECT_FALSE(s.observe(K, defaults(), std::nan(""), std::nullopt, std::nullopt));
  EXPECT_FALSE(s.observe(K, defaults(), std::nullopt, INFINITY, std::nullopt));
  EXPECT_FALSE(s.observe(K, defaults(), std::nullopt, std::nullopt, -1.0));
  auto c = s.get(K, defaults());
  EXPECT_EQ(c->samples, 0);
  EXPECT_DOUBLE_EQ(c->eta_bw, 0.6);
}

TEST(calibration_store, a_partly_bad_observation_does_not_apply_its_good_part) {
  auto db = fresh("partial");
  CalibrationStore s(*db);
  EXPECT_FALSE(s.observe(K, defaults(), 0.9, -1.0, std::nullopt));
  EXPECT_DOUBLE_EQ(s.get(K, defaults())->eta_bw, 0.6) << "eta 0.9 was not applied because mfu -1 was refused";
}

TEST(calibration_store, reset_goes_back_to_the_defaults) {
  auto db = fresh("reset");
  CalibrationStore s(*db);
  s.observe(K, defaults(), 0.9, 0.9, 1.0);
  ASSERT_TRUE(s.reset(K));
  auto c = s.get(K, defaults());
  EXPECT_EQ(c->samples, 0);
  EXPECT_DOUBLE_EQ(c->eta_bw, 0.6);
}

TEST(calibration_store, concurrent_feedback_jobs_do_not_lose_updates_even_on_separate_connections) {
  const auto d = fs::temp_directory_path() / ("mqde_cal_conc_" + std::to_string(::getpid()));
  fs::remove_all(d);
  {
    ASSERT_TRUE(Db::open(d / "m.db"));
  }
  std::atomic<int> failures{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 6; ++t)
    ts.emplace_back([&] {
      auto db = Db::open(d / "m.db");
      if (!db) {
        ++failures;
        return;
      }
      CalibrationStore s(**db);
      for (int i = 0; i < 20; ++i)
        if (!s.observe(K, defaults(), 0.5, std::nullopt, std::nullopt)) ++failures;
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(failures.load(), 0);
  auto db = Db::open(d / "m.db");
  CalibrationStore s(**db);
  EXPECT_EQ(s.get(K, defaults())->samples, 120)
      << "every one of the 120 read-modify-write cycles was counted";
}
