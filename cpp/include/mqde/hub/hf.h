#pragma once
// Minimal Hugging Face Hub client (docs/REFERENCES.md A.12-A.13, T-0204): model listing, config fetch,
// repo file listing, safetensors headers over HTTP Range (tensor shapes without downloading weights)
// and resumable-by-restart file download.
//
// Repos may carry a revision: "org/name@revision" (branch, tag or commit; default "main") everywhere a
// repo string is accepted. HF_ENDPOINT selects a mirror. Idempotent GETs are retried on network
// errors, HTTP 429 and 5xx (3 attempts, exponential backoff).
//
// Security: the token comes only from the caller (CLI --token) or the HF_TOKEN environment variable.
// It is sent as a bearer header and is never logged, stored, or echoed in errors.
// Network: honours https_proxy / HTTPS_PROXY (host:port form); follows cross-host redirects (CDN).

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "mqde/common/config.h"
#include "mqde/common/result.h"
#include "mqde/modelspec/model_spec.h"
#include "mqde/modelspec/safetensors.h"

namespace mqde {
namespace hub {

// HF_ENDPOINT if set (mirrors such as https://hf-mirror.com), else https://huggingface.co.
std::string default_endpoint();

struct HfOptions {
  std::string endpoint = default_endpoint();
  std::string token;       // empty -> read HF_TOKEN from the environment
  int timeout_s = 30;
};

struct RepoRef {
  std::string repo;              // "org/name"
  std::string revision = "main";
};

// "org/name" | "org/name@rev" | "name" (legacy single-segment ids). Rejects empty parts, "..", spaces.
Result<RepoRef> parse_repo_ref(std::string_view s);

struct RepoFile {
  std::string path;
  uint64_t size = 0;  // 0 when the Hub did not report it
};

// GET /api/models with the given query (e.g. "sort=downloads&limit=100&filter=text-generation").
Result<Json> list_models(const HfOptions& o, const std::string& query);

// GET /api/models/<repo>?expand[]=... (single model summary).
Result<Json> model_info(const HfOptions& o, const std::string& repo);

// GET /<repo>/resolve/main/<file> parsed as JSON (config.json etc.).
Result<Json> fetch_json_file(const HfOptions& o, const std::string& repo, const std::string& file);

// GET /api/models/<repo>/tree/main?recursive=1 -> files with sizes.
Result<std::vector<RepoFile>> list_files(const HfOptions& o, const std::string& repo);

// Tensor names / dtypes / shapes of one remote .safetensors file from two small Range requests
// (8-byte length, then the JSON header); the payload is never fetched.
Result<std::vector<modelspec::TensorInfo>> remote_safetensors_header(const HfOptions& o, const std::string& repo,
                                                                     const std::string& file);

// config.json + every weight shard's header -> the same ModelSpec inspect_local() builds, without downloading
// weights. `bytes_fetched` (optional) receives the number of body bytes transferred.
Result<modelspec::ModelSpec> inspect_remote(const HfOptions& o, const std::string& repo, uint64_t* bytes_fetched = nullptr);

// Download one file to `dest` (parent dirs created). Writes to "<dest>.part" then renames.
// `progress(done, total)` is called as bytes arrive; returning false (or `cancel`) aborts.
Result<uint64_t> download_file(const HfOptions& o, const std::string& repo, const std::string& file,
                               const std::filesystem::path& dest,
                               const std::function<bool(uint64_t, uint64_t)>& progress = {});

// ---- file selection for `download` (A.13) ----

struct FileGroup {
  std::string key;                 // quant / variant label, e.g. "Q4_K_M", "safetensors"
  std::vector<RepoFile> files;     // all shards of the group
  uint64_t bytes = 0;
  double quality = 0;              // higher is better (bits-per-weight proxy)
};

// Groups repo files into downloadable units: GGUF files by quant tag with shards
// ("-00001-of-00003") kept together; safetensors weights + config/tokenizer as one group.
std::vector<FileGroup> group_files(const std::vector<RepoFile>& files);

// Picks the best-quality group within `budget_bytes` (0 = unlimited). When nothing fits,
// falls back to the smallest group (llmfit behaviour). `want_quant` (case-insensitive) forces a group.
std::optional<FileGroup> choose_group(const std::vector<FileGroup>& groups, uint64_t budget_bytes,
                                      const std::string& want_quant = "");

// Percent-encodes a query-string value (RFC 3986 unreserved characters pass through).
std::string url_encode(const std::string& s);

// Parses "12G", "800M", "1.5T", "123456" into bytes. Returns nullopt on bad input.
std::optional<uint64_t> parse_size(const std::string& s);

}  // namespace hub
}  // namespace mqde
