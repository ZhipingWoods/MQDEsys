// Deployment advice: accuracy risk (measured beats prior) and objective-dependent ranking.
#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "advice.h"

using namespace mqde;
using namespace mqde::server;
namespace fs = std::filesystem;

namespace {
estimator::Recipe recipe(int wbits, int abits) {
  estimator::Recipe r;
  if (wbits < 16) {
    r.weight.fmt = quant::NumFormat::kInt;
    r.weight.bits = wbits;
  }
  if (abits < 16) {
    r.act.fmt = quant::NumFormat::kInt;
    r.act.bits = abits;
  }
  return r;
}
Json rec(const std::string& name, bool feasible, const std::string& verdict, int64_t conc,
         const std::string& risk) {
  return {{"recipe", name},
          {"feasible", feasible},
          {"deploy_verdict", verdict},
          {"max_concurrency", conc},
          {"risk", risk}};
}
std::vector<std::string> names(const std::vector<Json>& v) {
  std::vector<std::string> out;
  for (const auto& r : v) out.push_back(r["recipe"]);
  return out;
}
std::vector<Json> sample() {
  return {rec("W2A16", true, "native", 64, "high"), rec("W4A16", true, "native", 40, "medium"),
          rec("W8A16", true, "native", 20, "low"), rec("BF16", false, "native", 0, "low"),
          rec("W4A8", true, "emulated", 48, "medium")};
}
}  // namespace

TEST(advice_prior, bit_widths_map_to_risk) {
  EXPECT_EQ(prior_risk(recipe(16, 16)), Risk::kLow);
  EXPECT_EQ(prior_risk(recipe(8, 16)), Risk::kLow);
  EXPECT_EQ(prior_risk(recipe(8, 8)), Risk::kLow);
  EXPECT_EQ(prior_risk(recipe(4, 16)), Risk::kMedium);
  EXPECT_EQ(prior_risk(recipe(4, 8)), Risk::kMedium);
  EXPECT_EQ(prior_risk(recipe(2, 16)), Risk::kHigh);
  EXPECT_EQ(prior_risk(recipe(3, 16)), Risk::kHigh);
  EXPECT_EQ(prior_risk(recipe(8, 4)), Risk::kHigh) << "4-bit activations";
}

TEST(advice_measured, each_kind_reads_its_own_accuracy_number) {
  auto l = measured_from_summary("llm", {{"ppl_delta_pct", 1.2}});
  ASSERT_TRUE(l);
  EXPECT_EQ(l->risk, Risk::kLow);
  EXPECT_EQ(measured_from_summary("llm", {{"ppl_delta_pct", 8.0}})->risk, Risk::kMedium);
  EXPECT_EQ(measured_from_summary("llm", {{"ppl_delta_pct", 300.0}})->risk, Risk::kHigh);
  // the real MiniCPM-V-4.6 numbers: W4A16 100%, W4A8 96.1%, W2A16 0%
  EXPECT_EQ(measured_from_summary("vlm", {{"top1_agreement", 1.0}})->risk, Risk::kLow);
  EXPECT_EQ(measured_from_summary("vlm", {{"top1_agreement", 0.961}})->risk, Risk::kMedium);
  EXPECT_EQ(measured_from_summary("vlm", {{"top1_agreement", 0.0}})->risk, Risk::kHigh);
  EXPECT_EQ(measured_from_summary("vla", {{"action_rel_l2_over_noise_floor", 1.4}})->risk, Risk::kLow);
  EXPECT_EQ(measured_from_summary("vla", {{"action_rel_l2_over_noise_floor", 4.0}})->risk, Risk::kMedium);
  EXPECT_EQ(measured_from_summary("vla", {{"action_rel_l2_over_noise_floor", 9.0}})->risk, Risk::kHigh);
}

TEST(advice_measured, a_quantize_only_summary_is_not_a_measurement) {
  EXPECT_FALSE(measured_from_summary("llm", {{"ppl_delta_pct", nullptr}}));
  EXPECT_FALSE(measured_from_summary("vlm", {{"kl", nullptr}, {"top1_agreement", nullptr}}));
  EXPECT_FALSE(measured_from_summary("vla", Json::object()));
}

TEST(advice_rank, throughput_keeps_the_planners_order_most_sequences_first) {
  auto v = sample();
  rank(v, "throughput");
  EXPECT_EQ(names(v), (std::vector<std::string>{"W2A16", "W4A16", "W8A16", "W4A8", "BF16"}));
}

TEST(advice_rank, balanced_drops_high_risk_below_every_usable_recipe) {
  auto v = sample();
  rank(v, "balanced");
  EXPECT_EQ(names(v), (std::vector<std::string>{"W4A16", "W8A16", "W4A8", "W2A16", "BF16"}));
}

TEST(advice_rank, accuracy_puts_the_lowest_risk_first_and_still_never_an_infeasible_one) {
  auto v = sample();
  rank(v, "accuracy");
  EXPECT_EQ(names(v), (std::vector<std::string>{"W8A16", "W4A16", "W4A8", "W2A16", "BF16"}));
}

TEST(advice_rank, balanced_breaks_a_concurrency_tie_by_risk) {
  std::vector<Json> v = {rec("A", true, "native", 10, "medium"), rec("B", true, "native", 10, "low")};
  rank(v, "balanced");
  EXPECT_EQ(names(v), (std::vector<std::string>{"B", "A"}));
}

TEST(advice_objective, only_the_three_objectives_are_valid) {
  EXPECT_TRUE(valid_objective("accuracy") && valid_objective("balanced") && valid_objective("throughput"));
  EXPECT_FALSE(valid_objective("fast") || valid_objective(""));
}

TEST(advice_runs, the_newest_succeeded_report_for_the_same_model_wins) {
  const auto runs = fs::temp_directory_path() / ("mqde_advice_" + std::to_string(::getpid()));
  fs::remove_all(runs);
  const auto model = runs / "models" / "m";
  fs::create_directories(model);
  auto write = [&](const std::string& id, const std::string& verdict, const std::string& path, double top1) {
    fs::create_directories(runs / id);
    std::ofstream(runs / id / "report.json")
        << Json{{"job_id", "copied-from-elsewhere"},
                {"kind", "vlm"},
                {"verdict", verdict},
                {"spec", {{"model_path", path}}},
                {"recipes", {{"W4A16", {{"algorithm", "gptq"}, {"summary", {{"top1_agreement", top1}}}}}}}}
               .dump();
  };
  write("20261001-000000-aaaaaa", "succeeded", model.string(), 0.5);
  write("20261002-000000-bbbbbb", "succeeded", (model / ".." / "m").string(),
        0.99);                                                     // same dir, other spelling
  write("20261003-000000-cccccc", "failed", model.string(), 0.1);  // failed: ignored
  write("20261004-000000-dddddd", "succeeded", (runs / "models" / "other").string(), 0.0);  // other model
  const auto m = measured_for(runs, model.string());
  ASSERT_EQ(m.count("W4A16"), 1u);
  EXPECT_EQ(m.at("W4A16").job_id, "20261002-000000-bbbbbb");
  EXPECT_EQ(m.at("W4A16").algorithm, "gptq");
  EXPECT_EQ(m.at("W4A16").risk, Risk::kLow);
  EXPECT_TRUE(measured_for(runs, (runs / "models" / "never").string()).empty());
  fs::remove_all(runs);
}

TEST(advice_runs, an_llm_report_has_no_verdict_and_counts_once_its_report_stage_succeeded) {
  EXPECT_TRUE(finished_ok({{"stages",
                            {{{"stage", "quantize"}, {"status", "succeeded"}},
                             {{"stage", "report"}, {"status", "succeeded"}}}}}));
  EXPECT_FALSE(finished_ok({{"stages",
                             {{{"stage", "eval_fake_wo_kv"}, {"status", "succeeded"}},
                              {{"stage", "export"}, {"status", "failed"}}}}}))
      << "an llm job that died in export";
  EXPECT_FALSE(
      finished_ok({{"error", "boom"}, {"stages", {{{"stage", "report"}, {"status", "succeeded"}}}}}));
  EXPECT_FALSE(finished_ok({{"stages", {{{"stage", "quantize"}, {"status", "succeeded"}}}}}))
      << "still running";
  EXPECT_FALSE(finished_ok(
      {{"stages",
        {{{"stage", "quantize"}, {"status", "succeeded"}}, {{"stage", "report"}, {"status", "failed"}}}}}));
  EXPECT_TRUE(finished_ok({{"verdict", "succeeded"}}));
  EXPECT_FALSE(
      finished_ok({{"verdict", "failed"}, {"stages", {{{"stage", "report"}, {"status", "succeeded"}}}}}));
}

TEST(advice_runs, an_old_vla_report_whose_kind_is_JobReport_is_read_as_vla) {
  const auto runs = fs::temp_directory_path() / ("mqde_advice_vla_" + std::to_string(::getpid()));
  fs::remove_all(runs);
  fs::create_directories(runs / "j1");
  std::ofstream(runs / "j1" / "report.json")
      << Json{{"job_id", "j1"},
              {"kind", "JobReport"},
              {"verdict", "succeeded"},
              {"spec", {{"model_path", runs.string()}}},
              {"recipes",
               {{"W8A16", {{"algorithm", "rtn"}, {"summary", {{"action_rel_l2_over_noise_floor", 1.1}}}}}}}}
             .dump();
  const auto m = measured_for(runs, runs.string());
  ASSERT_EQ(m.count("W8A16"), 1u);
  EXPECT_EQ(m.at("W8A16").kind, "vla");
  EXPECT_EQ(m.at("W8A16").risk, Risk::kLow);
  fs::remove_all(runs);
}

TEST(advice_rank, balanced_prefers_the_more_accurate_option_when_it_serves_almost_as_many_sequences) {
  // the real Qwen2.5-0.5B case on a 24 GB card: memory is not the limit, W4A16 (+12.7% PPL) 433 vs FP8 429
  std::vector<Json> v = {rec("W4A16", true, "native", 433, "medium"), rec("FP8", true, "native", 429, "low"),
                         rec("BF16", true, "native", 422, "low")};
  rank(v, "balanced");
  EXPECT_EQ(names(v), (std::vector<std::string>{"FP8", "BF16", "W4A16"}));
}

TEST(advice_rank, balanced_still_takes_the_compressed_recipe_when_it_serves_clearly_more) {
  std::vector<Json> v = {rec("W8A16", true, "native", 20, "low"), rec("W4A16", true, "native", 40, "medium"),
                         rec("BF16", true, "native", 8, "low")};
  rank(v, "balanced");
  EXPECT_EQ(names(v), (std::vector<std::string>{"W4A16", "W8A16", "BF16"}));
}

TEST(advice_rank, the_band_is_measured_from_the_best_usable_option_not_a_high_risk_one) {
  // W2 serves 100 but is high risk: the band is 90% of 40 (W4A16), so W8A16 at 37 is in it and wins on risk
  std::vector<Json> v = {rec("W2A16", true, "native", 100, "high"),
                         rec("W4A16", true, "native", 40, "medium"), rec("W8A16", true, "native", 37, "low")};
  rank(v, "balanced");
  EXPECT_EQ(names(v), (std::vector<std::string>{"W8A16", "W4A16", "W2A16"}));
}

TEST(advice_runs, the_newest_job_wins_whatever_order_the_filesystem_lists_them_in) {
  const auto runs = fs::temp_directory_path() / ("mqde_advice_order_" + std::to_string(::getpid()));
  fs::remove_all(runs);
  for (int i = 0; i < 30; ++i) {  // ext4 lists in hash order: 30 entries are never accidentally sorted
    const auto id = fmt::format("202610{:02d}-000000-{:06d}", i + 1, i);
    fs::create_directories(runs / id);
    std::ofstream(runs / id / "report.json")
        << Json{{"kind", "vlm"},
                {"verdict", "succeeded"},
                {"spec", {{"model_path", runs.string()}}},
                {"recipes", {{"W4A16", {{"summary", {{"top1_agreement", i == 29 ? 1.0 : 0.0}}}}}}}}
               .dump();
  }
  const auto m = measured_for(runs, runs.string());
  EXPECT_EQ(m.at("W4A16").job_id, "20261030-000000-000029");
  EXPECT_EQ(m.at("W4A16").risk, Risk::kLow);
  fs::remove_all(runs);
}
