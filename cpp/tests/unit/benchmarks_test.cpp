#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "mqde/fit/benchmarks.h"

using namespace mqde;
using namespace mqde::fit;
namespace fs = std::filesystem;

namespace {

const HardwareId kHere{"NVIDIA GeForce RTX 4090", "AMD EPYC 7763 64-Core Processor"};

// A scratch benchmarks root, removed at the end of each test.
struct TempRoot {
  fs::path dir;
  TempRoot() {
    dir = fs::temp_directory_path() /
          fmt::format("mqde-bench-{}", ::testing::UnitTest::GetInstance()->random_seed());
    dir /= ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::remove_all(dir);
    fs::create_directories(dir);
  }
  ~TempRoot() { fs::remove_all(dir); }
  void put(const std::string& rel, const Json& j) const {
    const fs::path p = dir / rel;
    fs::create_directories(p.parent_path());
    std::ofstream(p) << j.dump();
  }
};

Json submission(const std::string& gpu, const std::string& cpu, Json results) {
  return {{"schemaVersion", 1}, {"hardware", {{"hardwareName", gpu}, {"cpu", cpu}}}, {"results", results}};
}

}  // namespace

TEST(benchmarks, canonical_slug_matches_llmfit) {
  EXPECT_EQ(canonical_slug("openbmb/MiniCPM-RobotManip"), "minicpmrobotmanip");
  EXPECT_EQ(canonical_slug("/cache/wuzhiping/hub/MiniCPM-RobotManip"), "minicpmrobotmanip");
  EXPECT_EQ(canonical_slug("/cache/wuzhiping/hub/MiniCPM-RobotManip/"), "minicpmrobotmanip");
  EXPECT_EQ(canonical_slug("Qwen3.8-27B_FP8"), "qwen3827bfp8");
}

TEST(benchmarks, hardware_slug_is_directory_safe) {
  EXPECT_EQ(hardware_slug("NVIDIA GeForce RTX 4090"), "nvidia-geforce-rtx-4090");
  EXPECT_EQ(hardware_slug(""), "cpu-only");
}

TEST(benchmarks, plausibility_bounds) {
  EXPECT_TRUE(plausible_tps(3.5));
  EXPECT_FALSE(plausible_tps(0.0));
  EXPECT_FALSE(plausible_tps(-1.0));
  EXPECT_FALSE(plausible_tps(10'001.0));
  EXPECT_FALSE(plausible_tps(std::nan("")));
}

TEST(benchmarks, median_merge_and_sample_counts) {
  auto idx = BenchmarkIndex::from_rows(
      {{"m", "bf16", 10.0, 100.0, 3}, {"m", "bf16", 30.0, std::nullopt, 2}, {"m", "bf16", 20.0, 300.0, 1}},
      {});
  auto m = idx.lookup("org/m", "BF16");
  ASSERT_TRUE(m);
  EXPECT_DOUBLE_EQ(m->tok_s, 20.0);  // median of 10, 20, 30
  EXPECT_EQ(m->sample_count, 6);     // 3 + 2 + 1
  ASSERT_TRUE(m->ttft_ms);
  EXPECT_DOUBLE_EQ(*m->ttft_ms, 200.0);  // median of the runs that reported it
  EXPECT_EQ(m->source, MeasuredSource::kLocal);
  EXPECT_EQ(m->hardware_label, "this machine");

  auto even =
      BenchmarkIndex::from_rows({{"m", "", 10.0, std::nullopt, 1}, {"m", "", 20.0, std::nullopt, 1}}, {});
  EXPECT_DOUBLE_EQ(even.lookup("m")->tok_s, 15.0);
}

TEST(benchmarks, quant_narrows_but_unrecorded_quant_matches_any) {
  auto idx =
      BenchmarkIndex::from_rows({{"m", "bf16", 10.0, std::nullopt, 1}, {"m", "", 40.0, std::nullopt, 1}}, {});
  EXPECT_DOUBLE_EQ(idx.lookup("m", "bf16")->tok_s, 25.0);   // bf16 row + quant-less row
  EXPECT_DOUBLE_EQ(idx.lookup("m", "w4a16")->tok_s, 40.0);  // only the quant-less row
}

TEST(benchmarks, local_wins_over_community) {
  auto idx = BenchmarkIndex::from_rows({{"m", "", 5.0, std::nullopt, 1}}, {{"m", "", 50.0, std::nullopt, 1}});
  EXPECT_EQ(idx.lookup("m")->source, MeasuredSource::kLocal);
  auto only_community = BenchmarkIndex::from_rows({}, {{"m", "", 50.0, std::nullopt, 1}});
  auto c = only_community.lookup("m");
  ASSERT_TRUE(c);
  EXPECT_EQ(c->source, MeasuredSource::kCommunity);
  EXPECT_EQ(c->hardware_label, "identical hardware");
}

TEST(benchmarks, no_match_returns_nothing) {
  auto idx = BenchmarkIndex::from_rows({{"m", "bf16", 10.0, std::nullopt, 1}}, {});
  EXPECT_FALSE(idx.lookup("other"));
  EXPECT_FALSE(idx.lookup("m", "w4a16"));
}

TEST(benchmarks, identical_hardware_only) {
  TempRoot t;
  const Json r = Json::array({{{"model", "M"}, {"avgTps", 12.0}, {"numRuns", 2}}});
  t.put("local/a/1.json", submission(kHere.gpu_name, kHere.cpu_name, r));
  t.put("local/a/2.json",
        submission("nvidia geforce rtx 4090", "amd epyc 7763 64-core processor", r));  // case-insensitive
  t.put("local/b/3.json", submission("NVIDIA GeForce RTX 3090", kHere.cpu_name, r));   // other GPU
  t.put("local/b/4.json", submission(kHere.gpu_name, "Intel Xeon", r));                // other CPU
  auto idx = BenchmarkIndex::load(kHere, t.dir);
  EXPECT_EQ(idx.local_runs(), 2);
  auto m = idx.lookup("M");
  ASSERT_TRUE(m);
  EXPECT_EQ(m->sample_count, 4);
}

TEST(benchmarks, community_requires_source_and_bad_files_are_skipped) {
  TempRoot t;
  const Json r = Json::array({{{"model", "M"}, {"avgTps", 12.0}}});
  t.put("community/x/no-source.json", submission(kHere.gpu_name, kHere.cpu_name, r));
  Json ok = submission(kHere.gpu_name, kHere.cpu_name, r);
  ok["source"] = "llmfit community data (MIT)";
  t.put("community/x/ok.json", ok);
  {
    std::ofstream(t.dir / "community/x/broken.json") << "{ not json";
  }
  auto idx = BenchmarkIndex::load(kHere, t.dir);
  EXPECT_EQ(idx.community_runs(), 1);
  EXPECT_EQ(idx.skipped(), 2);
}

TEST(benchmarks, implausible_results_are_dropped) {
  TempRoot t;
  t.put("local/a/1.json", submission(kHere.gpu_name, kHere.cpu_name,
                                     Json::array({{{"model", "M"}, {"avgTps", 0.0}},
                                                  {{"model", "M"}, {"avgTps", 99999.0}},
                                                  {{"model", "M"}, {"avgTps", 7.0}}})));
  auto idx = BenchmarkIndex::load(kHere, t.dir);
  EXPECT_EQ(idx.local_runs(), 1);
  EXPECT_DOUBLE_EQ(idx.lookup("M")->tok_s, 7.0);
}

TEST(benchmarks, write_then_load_round_trip) {
  TempRoot t;
  Submission s;
  s.hw = kHere;
  s.tool_version = "0.1.0";
  s.results.push_back({"/cache/hub/MiniCPM-RobotManip", "bf16", "hf_transformers", 3.6, 3.4, 3.8, 280.0, 5,
                       "steps/s", 3'900'000'000, 612});
  const Json j = to_json(s);
  EXPECT_EQ(j["results"][0]["peakMemBytes"], 3'900'000'000);
  EXPECT_EQ(j["results"][0]["contextTokens"], 612);
  auto path = write_local_submission(s, t.dir);
  ASSERT_TRUE(path) << path.error().to_string();
  EXPECT_EQ(path->parent_path().filename(), "nvidia-geforce-rtx-4090");
  auto idx = BenchmarkIndex::load(kHere, t.dir);
  auto m = idx.lookup("openbmb/MiniCPM-RobotManip", "BF16");
  ASSERT_TRUE(m);
  EXPECT_DOUBLE_EQ(m->tok_s, 3.6);
  EXPECT_EQ(m->sample_count, 5);
  EXPECT_DOUBLE_EQ(*m->ttft_ms, 280.0);

  Submission bad = s;
  bad.results[0].avg_tps = 0;
  EXPECT_FALSE(write_local_submission(bad, t.dir));
}
