#pragma once
// What the new-job wizard may offer as calibration / evaluation data, and whether a path the user typed is something the
// executor can read. Metadata only: no file content is read or returned.
//
// The supported text suffixes must stay what python/mqde/data/text.py TEXT_SUFFIXES says (a test keeps them in step).

#include <cstdint>
#include <map>
#include <string>

#include "mqde/common/config.h"

namespace mqde::server {

struct DatasetInfo {
  std::string kind = "missing";  // missing | file | dir
  int64_t files = 0;             // readable files found (text), or data parquet files (LeRobot)
  bool files_capped = false;     // the scan stopped at the limit: there are at least this many
  int64_t bytes = 0;             // total size of those files
  std::map<std::string, int64_t> formats;  // lower-case suffix -> count (text only)
  bool usable = false;
  std::string reason;  // why not, when it is not usable
};

// A file (.txt .md .jsonl .json .parquet) or a directory of them, scanned recursively (at most `max_files` readable files).
DatasetInfo inspect_text_dataset(const std::string& path, int64_t max_files = 20000);
// A LeRobot v3 directory: meta/info.json and at least one data/chunk-*/file-*.parquet.
DatasetInfo inspect_lerobot_dataset(const std::string& path);

Json to_json(const DatasetInfo& i);

// Names the executor resolves itself (not filesystem paths).
bool is_builtin_dataset(const std::string& id);
// The built-ins with whether they can be used on this machine (wikitext-2 needs the HF datasets cache).
Json builtin_datasets();

}  // namespace mqde::server
