#include <gtest/gtest.h>

#include "mqde/fit/installed.h"

using mqde::fit::InstalledIndex;
using mqde::fit::InstalledModel;
using mqde::fit::normalize_installed_name;

TEST(installed, normalize_ollama_tag_and_quant) {
  // T-R24 acceptance: the ollama name must match the catalog directory name.
  EXPECT_EQ(normalize_installed_name("qwen2.5:7b-instruct-q4_K_M"),
            normalize_installed_name("Qwen2.5-7B-Instruct"));
}

TEST(installed, normalize_strips_provider_and_quant_suffix) {
  EXPECT_EQ(normalize_installed_name("openbmb/MiniCPM-o-GGUF"), "minicpm-o");
  EXPECT_EQ(normalize_installed_name("Qwen3-8B-AWQ"), "qwen3-8b");
  EXPECT_EQ(normalize_installed_name("/cache/hub/MiniCPM-RobotManip"), "minicpm-robotmanip");
}

TEST(installed, is_installed_and_providers) {
  InstalledIndex idx;
  for (auto [n, src] :
       {std::pair{"qwen2.5:7b-instruct-q4_K_M", "ollama"}, {"Qwen2.5-7B-Instruct", "hf_cache"}})
    idx.models.push_back(InstalledModel{n, normalize_installed_name(n), src});
  EXPECT_TRUE(idx.is_installed("Qwen/Qwen2.5-7B-Instruct"));
  EXPECT_FALSE(idx.is_installed("Qwen/Qwen2.5-14B-Instruct"));
  const auto p = idx.installed_providers("Qwen2.5-7B-Instruct");
  ASSERT_EQ(p.size(), 2u);
  EXPECT_EQ(p[0], "ollama");
}

TEST(installed, probe_finishes_within_deadline) {
  const auto idx = mqde::fit::probe_installed();
  EXPECT_LT(idx.elapsed_ms, 2500.0);
}
