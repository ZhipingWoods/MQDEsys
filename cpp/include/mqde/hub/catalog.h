#pragma once
// HF model catalog (docs/REFERENCES.md A.12; llmfit hf_models.json).
// Lives in data/models/catalog.json. Each entry keeps the config.json fields the estimator needs,
// so catalog models can be evaluated by /fit without downloading weights.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "mqde/common/config.h"
#include "mqde/common/result.h"
#include "mqde/hub/hf.h"
#include "mqde/modelspec/model_spec.h"

namespace mqde {
namespace hub {

struct CatalogEntry {
  // required (A.12 ★)
  std::string name;             // "Qwen/Qwen3-8B"
  std::string provider;         // "Qwen"
  std::string parameter_count;  // human form, "8B"
  std::string quantization;     // "BF16" | "FP8" | "INT4" ...
  std::string format;           // "safetensors" | "gguf"
  int64_t context_length = 0;
  // optional
  int64_t parameters_raw = 0;
  std::string architecture, pipeline_tag, license, release_date, use_case;
  std::vector<std::string> capabilities;
  int64_t hf_downloads = 0, hf_likes = 0;
  Json config = Json::object();  // estimator subset of config.json
  // hygiene (not stored; recomputed on load)
  bool demoted = false;
  std::string demote_reason;
  // T-1303: two-level grouping for the UI catalog browser, not stored (recomputed on load, same reasoning
  // as demoted/demote_reason: a better derivation rule should apply to old data too, not just new entries).
  std::string category;      // coarse, from pipeline_tag: "LLM" | "VLM" | "Diffusion" | "Embedding" | "Other"
  std::string family_group;  // same-series different-size variants collapse into one group, e.g. "Qwen2.5"
};

struct Catalog {
  std::vector<CatalogEntry> entries;
  std::string updated_at;
};

std::filesystem::path default_catalog_path();  // <data_dir>/models/catalog.json

Result<Catalog> load_catalog(const std::filesystem::path& p = default_catalog_path());
Result<void> save_catalog(const Catalog& c, const std::filesystem::path& p = default_catalog_path());

// Directory hygiene (A.12): draft / eagle / speculative models, missing size or context, and
// name-vs-raw parameter mismatches (> 2x) are demoted and excluded from ranking.
void apply_hygiene(CatalogEntry& e);

// T-1303: fills category/family_group from pipeline_tag and name, deterministically (regex/string rules,
// not ML). Called at the same points as apply_hygiene (entry_from_hub, load_catalog) so both freshly
// fetched and previously stored entries get the current rule.
void classify(CatalogEntry& e);

// Case-insensitive substring search over name; demoted entries only when include_demoted.
// If capability is non-empty, only returns entries that have that capability.
std::vector<const CatalogEntry*> search(const Catalog& c, const std::string& query,
                                        const std::string& provider, const std::string& capability,
                                        size_t limit, bool include_demoted = false);
const CatalogEntry* find(const Catalog& c, const std::string& name);

// Builds a weight-less ModelSpec from the stored config so the fit/estimator path works unchanged.
Result<modelspec::ModelSpec> to_spec(const CatalogEntry& e);

// Builds an entry from a Hub model summary + its config.json (field mapping per A.12).
CatalogEntry entry_from_hub(const Json& info, const Json& config);

struct UpdateOptions {
  size_t limit = 500;
  bool trending = true;
  bool downloads = true;
  int interval_ms = 250;  // between config requests (A.12)
};
// Fetches list(s) from the Hub, then each config.json. Keeps existing entries not re-fetched.
Result<Catalog> update_catalog(const HfOptions& o, const UpdateOptions& u, Catalog base,
                               const std::function<void(size_t, size_t, const std::string&)>& progress = {});

Json to_json(const CatalogEntry& e, bool with_config = false);

// T-1303: periodically re-runs a refresh function (production wires in update_catalog against the real
// Hub; tests inject a fake) on a background thread and saves the result, so a long-running `mqde serve`
// keeps its catalog current without a human running `mqde catalog update` by hand. A failed refresh is
// logged and never crashes the server or loses the existing catalog (the refresh function itself already
// keeps un-refetched entries, same as the CLI path); the thread just tries again next interval.
class CatalogRefresher {
 public:
  using RefreshFn = std::function<Result<Catalog>(Catalog)>;

  struct Status {
    int64_t interval_s = 0;
    bool last_refresh_ok = true;  // true before the first attempt too: nothing has failed yet
    std::string last_refresh_at;  // ISO8601 UTC, empty before the first attempt
  };

  // interval <= 0 disables the background thread entirely. Takes milliseconds (rather than seconds) so
  // tests can use short intervals; status().interval_s reports it rounded down to whole seconds, which is
  // all production (seconds-granularity CLI flag) ever needs.
  CatalogRefresher(std::filesystem::path catalog_path, std::chrono::milliseconds interval,
                   RefreshFn refresh_fn);
  ~CatalogRefresher();
  CatalogRefresher(const CatalogRefresher&) = delete;
  CatalogRefresher& operator=(const CatalogRefresher&) = delete;

  Status status() const;

 private:
  void loop();

  std::filesystem::path catalog_path_;
  std::chrono::milliseconds interval_;
  RefreshFn refresh_fn_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  Status status_;
  std::thread thread_;
};

}  // namespace hub
}  // namespace mqde
