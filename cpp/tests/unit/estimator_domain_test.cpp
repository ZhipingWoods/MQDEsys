// T-1301: the domain registry (data/domains/*.yaml) that replaces ad-hoc "modality string == vla" checks
// with one data-driven lookup. Loads the real data/domains/ files shipped with the repo (same pattern as
// estimator_test.cpp's recipe/hardware tests), not synthetic fixtures, so a broken real file is caught here.
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "mqde/estimator/estimator.h"

using namespace mqde;
using namespace mqde::estimator;

TEST(estimator_domain, lists_every_shipped_domain_file) {
  auto ids = list_data_entries("domains");
  ASSERT_TRUE(ids) << ids.error().message;
  std::vector<std::string> v = *ids;
  for (const char* want : {"llm", "vla", "vlm", "world", "ewm"})
    EXPECT_NE(std::find(v.begin(), v.end(), want), v.end()) << "missing domain '" << want << "'";
}

TEST(estimator_domain, llm_and_vla_are_marked_implemented_because_their_pipelines_are_real) {
  auto llm = load_domain("llm");
  ASSERT_TRUE(llm) << llm.error().message;
  EXPECT_EQ(llm->id, "llm");
  EXPECT_EQ(llm->pipeline_status, "implemented");
  EXPECT_EQ(llm->model_families, (std::vector<std::string>{"generic_decoder"}));
  ASSERT_FALSE(llm->classic_algorithms.empty());

  auto vla = load_domain("vla");
  ASSERT_TRUE(vla) << vla.error().message;
  EXPECT_EQ(vla->pipeline_status, "implemented");
  EXPECT_EQ(vla->model_families, (std::vector<std::string>{"minicpm_vla"}));
}

TEST(estimator_domain, domains_with_no_real_pipeline_say_so_instead_of_pretending) {
  for (const char* id : {"vlm", "world", "ewm"}) {
    auto d = load_domain(id);
    ASSERT_TRUE(d) << id << ": " << d.error().message;
    EXPECT_EQ(d->pipeline_status, "ui_only_pending_backend") << id;
  }
}

TEST(estimator_domain, world_and_ewm_are_distinct_domains_not_aliases) {
  auto world = load_domain("world");
  auto ewm = load_domain("ewm");
  ASSERT_TRUE(world) << world.error().message;
  ASSERT_TRUE(ewm) << ewm.error().message;
  EXPECT_NE(world->display_name, ewm->display_name);
}

TEST(estimator_domain, an_unknown_domain_id_is_a_clear_not_found_error) {
  auto d = load_domain("does-not-exist");
  ASSERT_FALSE(d);
}

TEST(estimator_domain, rejects_a_domain_file_with_a_bad_pipeline_status) {
  // same technique estimator_test.cpp uses for malformed recipe/hardware fixtures: write a temp file under
  // the real data dir's domains/ subdirectory... instead, since load_domain accepts a path too, use that
  // directly rather than touching the shared data directory.
  const auto dir = std::filesystem::temp_directory_path() / ("mqde_domain_bad_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  const auto path = dir / "bad.yaml";
  {
    std::ofstream f(path);
    f << "id: bad\npipeline_status: not_a_real_value\n";
  }
  auto d = load_domain(path.string());
  ASSERT_FALSE(d);
  EXPECT_NE(d.error().message.find("pipeline_status"), std::string::npos) << d.error().message;
  std::filesystem::remove_all(dir);
}

TEST(estimator_domain, to_json_round_trips_the_classic_algorithms_list) {
  auto llm = load_domain("llm");
  ASSERT_TRUE(llm) << llm.error().message;
  const Json j = to_json(*llm);
  EXPECT_EQ(j.at("id"), "llm");
  ASSERT_TRUE(j.at("classic_algorithms").is_array());
  EXPECT_GE(j.at("classic_algorithms").size(), 3u);
  EXPECT_EQ(j.at("classic_algorithms")[0].at("name"), "rtn");
}
