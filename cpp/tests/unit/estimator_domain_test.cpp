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
  EXPECT_EQ(llm->model_families, (std::vector<std::string>{"decoder_llm", "generic_decoder"}));
  ASSERT_FALSE(llm->classic_algorithms.empty());

  auto vla = load_domain("vla");
  ASSERT_TRUE(vla) << vla.error().message;
  EXPECT_EQ(vla->pipeline_status, "implemented");
  EXPECT_EQ(vla->model_families, (std::vector<std::string>{"minicpm_vla"}));
}

TEST(estimator_domain, vlm_is_implemented_and_knows_the_minicpmv_family) {
  auto d = load_domain("vlm");
  ASSERT_TRUE(d) << d.error().message;
  EXPECT_EQ(d->pipeline_status, "implemented");
  EXPECT_NE(std::find(d->model_families.begin(), d->model_families.end(), "minicpmv"),
            d->model_families.end());
}

TEST(estimator_domain, world_and_ewm_are_implemented_and_name_their_dit_families) {
  auto world = load_domain("world");
  ASSERT_TRUE(world) << world.error().message;
  EXPECT_EQ(world->pipeline_status, "implemented");
  EXPECT_EQ(world->model_families, (std::vector<std::string>{"dit_world"}));
  EXPECT_EQ(world->default_eval_dataset, "synthetic-video");
  EXPECT_TRUE(world->deploy_backends.empty());

  auto ewm = load_domain("ewm");
  ASSERT_TRUE(ewm) << ewm.error().message;
  EXPECT_EQ(ewm->pipeline_status, "implemented");
  EXPECT_EQ(ewm->model_families, (std::vector<std::string>{"dit_ewm"}));
  EXPECT_TRUE(ewm->deploy_backends.empty());
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

// The order the new-job wizard shows the domain tabs in is data (`order` in each yaml), not directory order.
TEST(estimator_domain, list_domains_returns_them_in_the_registered_order_llm_vlm_vla_world_ewm) {
  auto ds = list_domains();
  ASSERT_TRUE(ds) << ds.error().message;
  std::vector<std::string> ids;
  for (const auto& d : *ds) ids.push_back(d.id);
  EXPECT_EQ(ids, (std::vector<std::string>{"llm", "vlm", "vla", "world", "ewm"}));
}

TEST(estimator_domain, order_is_part_of_the_json_and_a_missing_order_sorts_last_by_id) {
  auto d = load_domain("llm");
  ASSERT_TRUE(d);
  EXPECT_EQ(to_json(*d).at("order"), 10);
  const auto dir =
      std::filesystem::temp_directory_path() / ("mqde_domain_order_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  for (const auto& [id, order] :
       {std::pair<std::string, std::string>{"zeta", ""}, {"alpha", ""}, {"mid", "order: 5\n"}}) {
    std::ofstream(dir / (id + ".yaml")) << "id: " << id << "\ndisplay_name: " << id << "\n" << order;
  }
  auto ds = list_domains_in(dir);
  ASSERT_TRUE(ds) << ds.error().message;
  std::vector<std::string> ids;
  for (const auto& x : *ds) ids.push_back(x.id);
  EXPECT_EQ(ids, (std::vector<std::string>{"mid", "alpha", "zeta"}));
  std::filesystem::remove_all(dir);
}

TEST(estimator_domain, deploy_backends_is_loaded_and_emitted_so_the_wizard_can_gate_on_data) {
  auto llm = load_domain("llm");
  ASSERT_TRUE(llm) << llm.error().message;
  EXPECT_EQ(llm->deploy_backends, (std::vector<std::string>{"llamacpp", "vllm", "sglang"}));
  EXPECT_EQ(to_json(*llm).at("deploy_backends"), (Json{"llamacpp", "vllm", "sglang"}));
  for (const char* id : {"vlm"}) {
    auto d = load_domain(id);
    ASSERT_TRUE(d) << id << ": " << d.error().message;
    EXPECT_EQ(d->deploy_backends, (std::vector<std::string>{"vllm", "sglang"})) << id;
  }
  for (const char* id : {"vla", "world", "ewm"}) {
    auto d = load_domain(id);
    ASSERT_TRUE(d) << id << ": " << d.error().message;
    EXPECT_TRUE(d->deploy_backends.empty()) << id;
    EXPECT_TRUE(to_json(*d).at("deploy_backends").is_array()) << id;
    // an empty list is explained to the user, not left blank
    EXPECT_FALSE(d->deploy_note.empty()) << id;
    EXPECT_EQ(to_json(*d).at("deploy_note"), d->deploy_note) << id;
  }
}
