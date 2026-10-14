// What the wizard may offer as calibration / evaluation data, and whether a path the user typed is something the executor can read.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "datasets.h"

using namespace mqde::server;
namespace fs = std::filesystem;

namespace {
fs::path fresh(const char* name) {
  const auto d = fs::temp_directory_path() / (std::string("mqde-datasets-") + name + "-" + std::to_string(::getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}
void write(const fs::path& p, const std::string& body) {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << body;
}
}  // namespace

TEST(datasets, a_missing_path_says_so) {
  const auto i = inspect_text_dataset("/nonexistent/corpus");
  EXPECT_EQ(i.kind, "missing");
  EXPECT_FALSE(i.usable);
  EXPECT_NE(i.reason.find("does not exist"), std::string::npos);
}

TEST(datasets, a_text_file_is_usable_and_counted) {
  const auto d = fresh("file");
  write(d / "c.txt", "hello world");
  const auto i = inspect_text_dataset((d / "c.txt").string());
  EXPECT_EQ(i.kind, "file");
  EXPECT_TRUE(i.usable) << i.reason;
  EXPECT_EQ(i.files, 1);
  EXPECT_EQ(i.bytes, 11);
  EXPECT_EQ(i.formats.at(".txt"), 1);
}

TEST(datasets, every_supported_suffix_is_accepted_whatever_its_case) {
  const auto d = fresh("suffix");
  for (const char* n : {"a.txt", "b.MD", "c.JsonL", "d.json", "e.PARQUET"}) write(d / n, "x");
  const auto i = inspect_text_dataset(d.string());
  EXPECT_TRUE(i.usable) << i.reason;
  EXPECT_EQ(i.files, 5);
  EXPECT_EQ(i.formats.at(".md"), 1) << "keys are lower-cased";
  EXPECT_EQ(i.formats.at(".parquet"), 1);
}

TEST(datasets, an_unsupported_file_names_what_is_supported) {
  const auto d = fresh("badfile");
  write(d / "w.bin", "123");
  const auto i = inspect_text_dataset((d / "w.bin").string());
  EXPECT_FALSE(i.usable);
  EXPECT_NE(i.reason.find(".bin"), std::string::npos);
  EXPECT_NE(i.reason.find(".parquet"), std::string::npos);
}

TEST(datasets, an_empty_file_is_not_usable) {
  const auto d = fresh("empty");
  write(d / "e.txt", "");
  const auto i = inspect_text_dataset((d / "e.txt").string());
  EXPECT_FALSE(i.usable);
  EXPECT_NE(i.reason.find("empty"), std::string::npos);
}

TEST(datasets, a_directory_is_scanned_recursively_and_other_files_are_ignored) {
  const auto d = fresh("dir");
  write(d / "a.txt", "aaaa");
  write(d / "sub" / "deep" / "b.jsonl", "{}");
  write(d / "weights.bin", std::string(5000, 'x'));
  write(d / "pic.png", "png");
  const auto i = inspect_text_dataset(d.string());
  EXPECT_EQ(i.kind, "dir");
  EXPECT_TRUE(i.usable) << i.reason;
  EXPECT_EQ(i.files, 2);
  EXPECT_EQ(i.bytes, 6) << "only the readable files count towards the size";
  EXPECT_EQ(i.formats.size(), 2u);
}

TEST(datasets, a_directory_with_nothing_readable_is_not_usable) {
  const auto d = fresh("nothing");
  write(d / "x.bin", "1");
  write(d / "sub" / "y.png", "2");
  const auto i = inspect_text_dataset(d.string());
  EXPECT_FALSE(i.usable);
  EXPECT_EQ(i.files, 0);
  EXPECT_NE(i.reason.find("supported"), std::string::npos);
}

TEST(datasets, a_directory_of_only_empty_readable_files_is_not_usable) {
  const auto d = fresh("allempty");
  write(d / "a.txt", "");
  write(d / "b.json", "");
  EXPECT_FALSE(inspect_text_dataset(d.string()).usable);
}

TEST(datasets, a_huge_directory_is_cut_off_and_says_so) {
  const auto d = fresh("many");
  for (int i = 0; i < 30; ++i) write(d / ("f" + std::to_string(i) + ".txt"), "x");
  const auto i = inspect_text_dataset(d.string(), /*max_files=*/10);
  EXPECT_TRUE(i.usable);
  EXPECT_TRUE(i.files_capped);
  EXPECT_EQ(i.files, 10);
  const auto all = inspect_text_dataset(d.string());
  EXPECT_FALSE(all.files_capped);
  EXPECT_EQ(all.files, 30);
}

TEST(datasets, a_lerobot_directory_needs_meta_info_and_data_parquet) {
  const auto d = fresh("lerobot");
  auto i = inspect_lerobot_dataset(d.string());
  EXPECT_FALSE(i.usable);
  EXPECT_NE(i.reason.find("meta/info.json"), std::string::npos);
  write(d / "meta" / "info.json", "{}");
  i = inspect_lerobot_dataset(d.string());
  EXPECT_FALSE(i.usable);
  EXPECT_NE(i.reason.find("data/chunk-*/file-*.parquet"), std::string::npos);
  write(d / "data" / "chunk-000" / "file-000.parquet", "pq");
  write(d / "data" / "chunk-001" / "file-003.parquet", "pq2");
  write(d / "data" / "chunk-000" / "notes.txt", "ignored");
  i = inspect_lerobot_dataset(d.string());
  EXPECT_TRUE(i.usable) << i.reason;
  EXPECT_EQ(i.files, 2);
  EXPECT_EQ(i.kind, "dir");
}

TEST(datasets, a_lerobot_path_that_is_a_file_or_missing_is_not_usable) {
  const auto d = fresh("lerobot2");
  write(d / "x.txt", "x");
  EXPECT_FALSE(inspect_lerobot_dataset((d / "x.txt").string()).usable);
  EXPECT_EQ(inspect_lerobot_dataset("/nonexistent/lr").kind, "missing");
}

TEST(datasets, builtin_ids_are_recognised) {
  EXPECT_TRUE(is_builtin_dataset("synthetic"));
  EXPECT_TRUE(is_builtin_dataset("wikitext-2-raw-v1:train"));
  EXPECT_TRUE(is_builtin_dataset("wikitext-2-raw-v1:test"));
  EXPECT_FALSE(is_builtin_dataset("/data/corpus"));
  EXPECT_FALSE(is_builtin_dataset(""));
  EXPECT_FALSE(is_builtin_dataset("synthetic2"));
}

TEST(datasets, the_builtin_list_reports_whether_wikitext_is_in_the_hf_cache) {
  const auto cache = fresh("hfcache");
  ::setenv("HF_HUB_CACHE", cache.c_str(), 1);
  auto find = [](const mqde::Json& list, const std::string& id) {
    for (const auto& e : list)
      if (e.at("id") == id) return e;
    return mqde::Json();
  };
  auto b = builtin_datasets();
  EXPECT_FALSE(find(b, "wikitext-2-raw-v1:train").at("available")) << "an empty cache";
  EXPECT_TRUE(find(b, "synthetic").at("available"));
  write(cache / "datasets--wikitext" / "snapshots" / "abc" / "wikitext-2-raw-v1" / "train-00000-of-00001.parquet", "pq");
  b = builtin_datasets();
  EXPECT_TRUE(find(b, "wikitext-2-raw-v1:train").at("available"));
  ::unsetenv("HF_HUB_CACHE");
}

TEST(datasets, json_form_carries_every_field_the_ui_reads) {
  const auto d = fresh("json");
  write(d / "a.txt", "abc");
  const auto j = to_json(inspect_text_dataset(d.string()));
  for (const char* k : {"kind", "files", "files_capped", "bytes", "formats", "usable"}) EXPECT_TRUE(j.contains(k)) << k;
  EXPECT_FALSE(j.contains("reason")) << "no reason when it is usable";
  const auto bad = to_json(inspect_text_dataset("/nonexistent/x"));
  EXPECT_TRUE(bad.contains("reason"));
}
