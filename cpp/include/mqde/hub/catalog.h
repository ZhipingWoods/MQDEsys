#pragma once
// HF model catalog (docs/REFERENCES.md A.12; llmfit hf_models.json).
// Lives in data/models/catalog.json. Each entry keeps the config.json fields the estimator needs,
// so catalog models can be evaluated by /fit without downloading weights.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
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

}  // namespace hub
}  // namespace mqde
