#include "datasets.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string_view>

namespace mqde::server {
namespace fs = std::filesystem;
namespace {

// python/mqde/data/text.py TEXT_SUFFIXES
constexpr std::array<std::string_view, 5> kTextSuffixes = {".txt", ".md", ".jsonl", ".json", ".parquet"};

std::string lower_suffix(const fs::path& p) {
  std::string s = p.extension().string();
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool text_suffix(const std::string& s) {
  return std::find(kTextSuffixes.begin(), kTextSuffixes.end(), s) != kTextSuffixes.end();
}

std::string supported_list() {
  std::string out;
  for (auto s : kTextSuffixes) out += (out.empty() ? "" : " ") + std::string(s);
  return out;
}

fs::path hf_hub_cache() {
  if (const char* e = std::getenv("HF_HUB_CACHE"); e && *e) return e;
  if (const char* h = std::getenv("HF_HOME"); h && *h) return fs::path(h) / "hub";
  const char* home = std::getenv("HOME");
  return fs::path(home ? home : "") / ".cache" / "huggingface" / "hub";
}

// python/mqde/data/text.py _wikitext_file: datasets--wikitext/snapshots/*/wikitext-2-raw-v1/<split>-*.parquet
bool wikitext_in_cache(const std::string& split) {
  std::error_code ec;
  const auto snaps = hf_hub_cache() / "datasets--wikitext" / "snapshots";
  for (const auto& snap : fs::directory_iterator(snaps, ec)) {
    std::error_code ec2;
    for (const auto& f : fs::directory_iterator(snap.path() / "wikitext-2-raw-v1", ec2)) {
      const auto n = f.path().filename().string();
      if (n.starts_with(split + "-") && n.ends_with(".parquet")) return true;
    }
  }
  return false;
}

}  // namespace

DatasetInfo inspect_text_dataset(const std::string& path, int64_t max_files) {
  DatasetInfo i;
  std::error_code ec;
  const fs::path p(path);
  if (!fs::exists(p, ec)) {
    i.reason = "path does not exist";
    return i;
  }
  if (fs::is_regular_file(p, ec)) {
    i.kind = "file";
    const auto suffix = lower_suffix(p);
    if (!text_suffix(suffix)) {
      i.reason = "unsupported file type '" + suffix + "' (supported: " + supported_list() + ")";
      return i;
    }
    const auto size = static_cast<int64_t>(fs::file_size(p, ec));
    if (ec || size == 0) {
      i.reason = "file is empty";
      return i;
    }
    i.files = 1;
    i.bytes = size;
    i.formats[suffix] = 1;
    i.usable = true;
    return i;
  }
  if (!fs::is_directory(p, ec)) {
    i.reason = "not a file or directory";
    return i;
  }
  i.kind = "dir";
  fs::recursive_directory_iterator it(p, fs::directory_options::skip_permission_denied, ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    std::error_code e2;
    if (!it->is_regular_file(e2)) continue;
    const auto suffix = lower_suffix(it->path());
    if (!text_suffix(suffix)) continue;
    if (i.files >= max_files) {
      i.files_capped = true;
      break;
    }
    const auto size = static_cast<int64_t>(it->file_size(e2));
    if (e2 || size == 0) continue;  // an empty file adds nothing to calibrate on
    ++i.files;
    i.bytes += size;
    ++i.formats[suffix];
  }
  i.usable = i.files > 0 && i.bytes > 0;
  if (!i.usable) i.reason = "no readable text in the directory (supported: " + supported_list() + ")";
  return i;
}

DatasetInfo inspect_lerobot_dataset(const std::string& path) {
  DatasetInfo i;
  std::error_code ec;
  const fs::path p(path);
  if (!fs::exists(p, ec)) {
    i.reason = "path does not exist";
    return i;
  }
  if (!fs::is_directory(p, ec)) {
    i.kind = "file";
    i.reason = "not a directory (a LeRobot v3 dataset is a directory)";
    return i;
  }
  i.kind = "dir";
  if (!fs::is_regular_file(p / "meta" / "info.json", ec)) {
    i.reason = "not a LeRobot v3 directory: meta/info.json is missing";
    return i;
  }
  for (const auto& chunk : fs::directory_iterator(p / "data", ec)) {
    if (!chunk.is_directory() || !chunk.path().filename().string().starts_with("chunk-")) continue;
    std::error_code e2;
    for (const auto& f : fs::directory_iterator(chunk.path(), e2)) {
      const auto n = f.path().filename().string();
      if (f.is_regular_file(e2) && n.starts_with("file-") && n.ends_with(".parquet")) {
        ++i.files;
        i.bytes += static_cast<int64_t>(f.file_size(e2));
      }
    }
  }
  i.usable = i.files > 0;
  if (!i.usable) i.reason = "not a LeRobot v3 directory: no data/chunk-*/file-*.parquet";
  return i;
}

Json to_json(const DatasetInfo& i) {
  Json j = {{"kind", i.kind},
            {"files", i.files},
            {"files_capped", i.files_capped},
            {"bytes", i.bytes},
            {"usable", i.usable}};
  Json formats = Json::object();
  for (const auto& [k, v] : i.formats) formats[k] = v;
  j["formats"] = formats;
  if (!i.usable) j["reason"] = i.reason;
  return j;
}

bool is_builtin_dataset(const std::string& id) {
  return id == "synthetic" || id == "wikitext-2-raw-v1:train" || id == "wikitext-2-raw-v1:test";
}

Json builtin_datasets() {
  const bool wiki = wikitext_in_cache("train");
  Json wikitext = {{"id", "wikitext-2-raw-v1:train"},
                   {"label", "wikitext-2（train 切分）"},
                   {"kind", "text"},
                   {"roles", Json::array({"calibration"})},
                   {"available", wiki}};
  if (!wiki) wikitext["reason"] = "not in this server's Hugging Face datasets cache (datasets--wikitext)";
  Json synthetic = {{"id", "synthetic"},
                    {"label", "synthetic（程序生成的图文样本）"},
                    {"kind", "vision"},
                    {"roles", Json::array({"calibration", "evaluation"})},
                    {"available", true}};
  return Json::array({wikitext, synthetic});
}

}  // namespace mqde::server
