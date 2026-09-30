#pragma once
// Installed runtime detection (docs/REFERENCES.md A.11).
// Probes 8 sources in parallel: HTTP-based (ollama, LM Studio, vLLM) and
// filesystem-based (llama.cpp, MLX, RamaLama, HF cache, Docker Model Runner).
// Single-source HTTP timeout: 400 ms. Overall deadline: 2 s.

#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "mqde/common/config.h"

namespace mqde {
namespace fit {

// A single detected model from any source.
struct InstalledModel {
  std::string name;       // raw name as returned by the source
  std::string canonical;  // normalized (lowercase, no quant suffix, no provider prefix)
  std::string source;     // "ollama" | "lm_studio" | "vllm" | "llama_cpp" | "hf_cache" | etc.
};

struct InstalledIndex {
  std::vector<InstalledModel> models;
  std::vector<std::string> active_sources;  // sources that responded / had files
  double elapsed_ms = 0;

  // Returns true when any installed model's canonical form matches the given name
  // (or a version with provider prefix stripped, e.g. "openbmb/minicpm-4b" -> "minicpm-4b").
  bool is_installed(std::string_view model_name) const;

  // Returns source names of all installed providers for this model.
  std::vector<std::string> installed_providers(std::string_view model_name) const;
};

// Probe all sources in parallel and return the aggregate index.
// Reads timeouts from data/installed.yaml (falls back to built-in defaults if absent).
InstalledIndex probe_installed();

// Normalize a model name for matching: lowercase, strip quant suffix tokens, strip provider prefix.
std::string normalize_installed_name(std::string_view raw);

nlohmann::json to_json(const InstalledIndex& idx);

}  // namespace fit
}  // namespace mqde
