#include <gtest/gtest.h>

#include "mqde/fit/installed.h"

using namespace mqde::fit;

TEST(installed, normalize_model_name) {
  EXPECT_EQ(normalize_model_name("Qwen/Qwen2.5-7B-Instruct"), "qwen25-7b-instruct");
  EXPECT_EQ(normalize_model_name("meta-llama/Llama-3.1-8B"), "llama-31-8b");
  EXPECT_EQ(normalize_model_name("openbmb/MiniCPM-RobotManip"), "minicpm-robotmanip");
  EXPECT_EQ(normalize_model_name("/cache/hub/models--Qwen--Qwen2.5-7B"), "qwen25-7b");
  EXPECT_EQ(normalize_model_name("llama3.1:8b"), "llama318b");
}

TEST(installed, ollama_style_name_match) {
  // Simulate ollama-style names
  std::set<std::string> installed;
  installed.insert(normalize_model_name("llama3.1:8b"));
  installed.insert(normalize_model_name("qwen2.5:7b"));
  
  EXPECT_TRUE(is_installed(installed, "meta-llama/Llama-3.1-8B", "llama3.1:8b"));
  EXPECT_TRUE(is_installed(installed, "Qwen/Qwen2.5-7B", "qwen2.5:7b"));
  EXPECT_FALSE(is_installed(installed, "meta-llama/Llama-3.3-70B"));
}

TEST(installed, installed_providers_helper) {
  std::vector<std::pair<std::string, std::string>> pm;
  pm.emplace_back("ollama", "llama3.1:8b");
  pm.emplace_back("vllm", "meta-llama/Llama-3.1-8B");
  pm.emplace_back("ollama", "qwen2.5:7b");
  pm.emplace_back("lmstudio", "Qwen/Qwen2.5-7B-Instruct");
  
  auto providers = installed_providers(pm, "meta-llama/Llama-3.1-8B");
  ASSERT_EQ(providers.size(), 2);
  EXPECT_NE(std::find(providers.begin(), providers.end(), "ollama"), providers.end());
  EXPECT_NE(std::find(providers.begin(), providers.end(), "vllm"), providers.end());
  
  auto qwen_providers = installed_providers(pm, "Qwen/Qwen2.5-7B");
  ASSERT_EQ(qwen_providers.size(), 2);
  EXPECT_NE(std::find(qwen_providers.begin(), qwen_providers.end(), "ollama"), qwen_providers.end());
  EXPECT_NE(std::find(qwen_providers.begin(), qwen_providers.end(), "lmstudio"), qwen_providers.end());
  
  EXPECT_TRUE(installed_providers(pm, "unknown-model").empty());
}
