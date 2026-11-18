#pragma once
// Measured throughput baselines (the reference notes A.8; llmfit benchmarks.rs + share.rs).
//
// Two stores under data/benchmarks/:
//   local/<hw-slug>/*.json      runs made by `mqde bench` on this machine  -> measured_local
//   community/<hw-slug>/*.json  imported third-party data (must carry `source`) -> measured_community
//
// Matching follows llmfit exactly: a measurement only transfers between IDENTICAL hardware
// (same GPU name, same CPU name, both case-insensitive). There is no nearest-hardware fallback.
// Within a store, runs for the same model are merged by the median tok/s; sample counts add up.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "mqde/common/config.h"

namespace mqde::fit {

enum class MeasuredSource { kLocal, kCommunity };

struct MeasuredTps {
  double tok_s = 0;
  std::optional<double> ttft_ms;  // median of the runs that reported it
  int sample_count = 0;           // number of runs behind the value
  std::string hardware_label;     // "this machine" | "identical hardware"
  MeasuredSource source = MeasuredSource::kLocal;
};

// Hardware identity used for matching (llmfit hardware_payload_matches).
struct HardwareId {
  std::string gpu_name;  // empty = no GPU
  std::string cpu_name;
};

// Plausible tok/s (llmfit bench::is_plausible_tps): finite, > 0, <= 10 000.
bool plausible_tps(double tps);

// Model-name normalization for lookups. The last path segment (so "org/Name" and a local directory
// "/cache/hub/Name" meet), lower-cased, with '-', '_', '.' removed (llmfit canonical_slug).
std::string canonical_slug(std::string_view name);

// Directory-safe hardware slug: lower-case, runs of non [a-z0-9] collapsed to '-'. For file layout
// only; matching compares the recorded names, not the slug.
std::string hardware_slug(std::string_view gpu_name);

class BenchmarkIndex {
 public:
  // Loads local/ and community/ under `root` (default <data_dir>/benchmarks). Files whose hardware
  // does not match `hw` are skipped; unreadable files are skipped and counted in `skipped()`.
  static BenchmarkIndex load(const HardwareId& hw, const std::filesystem::path& root = {});

  // Local wins over community. The lookup key is canonical_slug(model), matched against each
  // result's `model`. `quant` narrows the match when both sides specify one (case-insensitive);
  // a result with no recorded quant matches any.
  std::optional<MeasuredTps> lookup(std::string_view model, std::string_view quant = {}) const;

  int local_runs() const { return static_cast<int>(local_.size()); }
  int community_runs() const { return static_cast<int>(community_.size()); }
  int skipped() const { return skipped_; }

  struct Row {
    std::string slug, quant;
    double tps = 0;
    std::optional<double> ttft_ms;
    int runs = 1;
  };
  // Test hook: builds an index from rows directly.
  static BenchmarkIndex from_rows(std::vector<Row> local, std::vector<Row> community);

 private:
  std::optional<MeasuredTps> find(const std::vector<Row>& rows, const std::string& slug,
                                  std::string_view quant, MeasuredSource src) const;
  std::vector<Row> local_, community_;
  int skipped_ = 0;
};

// Parses one submission file. Returns the rows that match `hw` (empty when the hardware differs).
// `require_source` rejects community files lacking a top-level `source` (license attribution).
Result<std::vector<BenchmarkIndex::Row>> read_submission(const std::filesystem::path& file,
                                                         const HardwareId& hw, bool require_source);

// A new local submission as written by `mqde bench` (schema: data/benchmarks/schema.json).
struct Submission {
  HardwareId hw;
  double vram_gb = 0, ram_gb = 0;
  int cpu_cores = 0, gpu_count = 1;
  std::string tool_version;
  struct Entry {
    std::string model, quant, runtime;
    double avg_tps = 0, min_tps = 0, max_tps = 0;
    std::optional<double> avg_ttft_ms;
    int num_runs = 0;
    std::string unit = "tok/s";  // VLA: "steps/s"
    int64_t peak_mem_bytes = 0;  // 0 = not recorded
    int64_t context_tokens = 0;
  };
  std::vector<Entry> results;
};
Json to_json(const Submission& s);
// Writes <root>/local/<hardware_slug>/<unix>-<8 hex>.json and returns the path.
Result<std::filesystem::path> write_local_submission(const Submission& s,
                                                     const std::filesystem::path& root = {});

}  // namespace mqde::fit
