#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "mqde/common/config.h"

namespace fs = std::filesystem;

TEST(common_config, format_bytes) {
  EXPECT_EQ(mqde::format_bytes(512), "512 B");
  EXPECT_EQ(mqde::format_bytes(1536), "1.50 KiB");
  EXPECT_EQ(mqde::format_bytes(3.0 * (1ull << 30)), "3.00 GiB");
}

TEST(common_config, yaml_scalars_are_typed) {
  auto p = fs::temp_directory_path() / "mqde_common_test.yaml";
  std::ofstream(p) << "a: 1\nb: 2.5\nc: true\nd: hello\ne: '42'\nf: [1, 2]\ng: {h: null}\n";
  auto j = mqde::load_structured_file(p);
  ASSERT_TRUE(j) << j.error().to_string();
  EXPECT_TRUE((*j)["a"].is_number_integer());
  EXPECT_DOUBLE_EQ((*j)["b"].get<double>(), 2.5);
  EXPECT_TRUE((*j)["c"].get<bool>());
  EXPECT_EQ((*j)["d"], "hello");
  EXPECT_TRUE((*j)["e"].is_string()) << "quoted scalars must stay strings";
  EXPECT_EQ((*j)["f"].size(), 2u);
  EXPECT_TRUE((*j)["g"]["h"].is_null());
  fs::remove(p);
}

TEST(common_config, missing_file_is_not_found) {
  auto j = mqde::load_structured_file("/nonexistent/x.yaml");
  ASSERT_FALSE(j);
  EXPECT_EQ(j.error().code, mqde::Errc::kNotFound);
}

TEST(common_config, get_required_reports_field) {
  mqde::Json j = {{"x", "str"}};
  auto missing = mqde::get_required<int>(j, "y", "cfg");
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().message.find("'y'"), std::string::npos);
  auto wrong = mqde::get_required<int>(j, "x", "cfg");
  ASSERT_FALSE(wrong);
  EXPECT_EQ(wrong.error().code, mqde::Errc::kParse);
}

TEST(common_config, resolve_data_file_case_insensitive) {
  auto p = mqde::resolve_data_file("recipes", "W4A16");
  ASSERT_TRUE(p) << p.error().to_string();
  EXPECT_EQ(p->filename(), "w4a16.yaml");
}
