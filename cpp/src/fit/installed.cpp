#include "mqde/fit/installed.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <future>
#include <regex>

#include <httplib.h>

#include "mqde/common/log.h"

namespace mqde::fit {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// ---------- name normalisation ----------

std::string normalize_installed_name(std::string_view raw) {
  std::string s(raw);
  // Split on '/' and take the last segment (strip provider prefix).
  const auto slash = s.rfind('/');
  if (slash != std::string::npos) s = s.substr(slash + 1);
  // ollama name:tag -> "name-tag" (the tag carries size/variant, e.g. "7b-instruct-q4_K_M");
  // a bare ":latest" tag carries nothing and is dropped.
  const auto colon = s.find(':');
  if (colon != std::string::npos) {
    const std::string tag = s.substr(colon + 1);
    s = s.substr(0, colon);
    if (!tag.empty() && tag != "latest") s += "-" + tag;
  }
  // Lowercase.
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  // Strip common quant suffixes like -q4_k_m, -gguf, -awq, -fp8, etc.
  static const std::regex quant_re(
      R"([-_](q[0-9][_a-z]*|gguf|awq|gptq|fp8|fp16|bf16|int8|int4|w4a[0-9]+|w8a[0-9]+)$)");
  s = std::regex_replace(s, quant_re, "");
  // Also strip trailing version tags: -v1, -1.0, -20240101.
  static const std::regex ver_re(R"([-_](v\d[\d.]*|\d{8})$)");
  s = std::regex_replace(s, ver_re, "");
  return s;
}

// ---------- HTTP helper ----------

static std::vector<std::string> http_probe(const std::string& host, int port, const std::string& path,
                                           const std::string& json_ptr, int timeout_ms) {
  httplib::Client cli(host, port);
  cli.set_connection_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);
  cli.set_read_timeout(timeout_ms / 1000, (timeout_ms % 1000) * 1000);
  auto res = cli.Get(path);
  if (!res || res->status != 200) return {};
  try {
    auto j = nlohmann::json::parse(res->body);
    // Navigate a simple dotted path like "models[].name" or "data[].id".
    // We support only the two patterns used by ollama/lm_studio/vllm.
    std::vector<std::string> names;
    if (json_ptr.starts_with("models") && j.contains("models")) {
      for (const auto& m : j["models"]) {
        const auto key = json_ptr.substr(json_ptr.rfind('.') + 1);
        if (m.contains(key)) names.push_back(m[key].get<std::string>());
      }
    } else if (json_ptr.starts_with("data") && j.contains("data")) {
      for (const auto& m : j["data"]) {
        const auto key = json_ptr.substr(json_ptr.rfind('.') + 1);
        if (m.contains(key)) names.push_back(m[key].get<std::string>());
      }
    }
    return names;
  } catch (...) {
    return {};
  }
}

// ---------- filesystem probe ----------

static std::vector<std::string> fs_probe(const std::vector<std::string>& raw_paths, const std::string& marker,
                                         const std::string& ext, int depth) {
  std::vector<std::string> names;
  for (const auto& rp : raw_paths) {
    std::string expanded = rp;
    // Expand leading ~
    if (!expanded.empty() && expanded[0] == '~') {
      const char* home = ::getenv("HOME");
      if (home) expanded = std::string(home) + expanded.substr(1);
    }
    fs::path root(expanded);
    if (!fs::exists(root)) continue;

    // Walk up to `depth` levels.
    std::function<void(const fs::path&, int)> walk = [&](const fs::path& p, int d) {
      if (d < 0) return;
      std::error_code ec;
      for (const auto& entry : fs::directory_iterator(p, ec)) {
        if (ec) break;
        if (entry.is_directory() && d > 0) {
          // Check for marker file in this directory.
          if (!marker.empty() && fs::exists(entry.path() / marker)) {
            names.push_back(entry.path().filename().string());
          }
          walk(entry.path(), d - 1);
        } else if (entry.is_regular_file() && !ext.empty()) {
          if (entry.path().extension() == ext) names.push_back(entry.path().stem().string());
        }
      }
    };
    walk(root, depth);
  }
  return names;
}

// ---------- probe_installed ----------

InstalledIndex probe_installed() {
  InstalledIndex idx;
  const auto t0 = Clock::now();

  // Launch all probes asynchronously.
  struct SourceResult {
    std::string source;
    std::vector<std::string> raw_names;
    bool active = false;
  };

  auto ollama_f = std::async(std::launch::async, [&]() -> SourceResult {
    auto names = http_probe("127.0.0.1", 11434, "/api/tags", "models[].name", 400);
    return {"ollama", names, !names.empty()};
  });

  auto lms_f = std::async(std::launch::async, [&]() -> SourceResult {
    auto names = http_probe("127.0.0.1", 1234, "/v1/models", "data[].id", 400);
    return {"lm_studio", names, !names.empty()};
  });

  auto vllm_f = std::async(std::launch::async, [&]() -> SourceResult {
    auto names = http_probe("127.0.0.1", 8000, "/v1/models", "data[].id", 400);
    return {"vllm", names, !names.empty()};
  });

  auto llama_cpp_f = std::async(std::launch::async, [&]() -> SourceResult {
    auto names = fs_probe({"~/.cache/llama.cpp", "~/models", "/opt/models"}, "", ".gguf", 3);
    return {"llama_cpp", names, !names.empty()};
  });

  auto hf_cache_f = std::async(std::launch::async, [&]() -> SourceResult {
    // Hub layout: <HF_HOME>/hub/models--<org>--<name>/snapshots/<hash>/...; the directory name
    // is the model id, the snapshot hashes are not.
    std::vector<std::string> names;
    std::string root = "~/.cache/huggingface/hub";
    if (const char* h = ::getenv("HF_HUB_CACHE"))
      root = h;
    else if (const char* h2 = ::getenv("HF_HOME"))
      root = std::string(h2) + "/hub";
    if (root[0] == '~')
      if (const char* home = ::getenv("HOME")) root = std::string(home) + root.substr(1);
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(root, ec)) {
      const auto dn = e.path().filename().string();
      if (!e.is_directory() || !dn.starts_with("models--")) continue;
      std::string id = dn.substr(8);
      if (const auto p = id.find("--"); p != std::string::npos) id.replace(p, 2, "/");
      names.push_back(id);
    }
    return {"hf_cache", names, !names.empty()};
  });

  auto ramalama_f = std::async(std::launch::async, [&]() -> SourceResult {
    auto names = fs_probe({"~/.local/share/ramalama/models"}, "", "", 2);
    return {"ramalama", names, !names.empty()};
  });

  // Collect results, respecting overall 2s deadline.
  auto collect = [&](std::future<SourceResult>& f) {
    auto deadline = t0 + std::chrono::milliseconds(2000);
    if (f.wait_until(deadline) == std::future_status::ready) {
      auto r = f.get();
      if (r.active) idx.active_sources.push_back(r.source);
      for (const auto& n : r.raw_names) {
        InstalledModel m;
        m.name = n;
        m.canonical = normalize_installed_name(n);
        m.source = r.source;
        idx.models.push_back(std::move(m));
      }
    }
  };

  collect(ollama_f);
  collect(lms_f);
  collect(vllm_f);
  collect(llama_cpp_f);
  collect(hf_cache_f);
  collect(ramalama_f);

  const auto elapsed = Clock::now() - t0;
  idx.elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
  return idx;
}

// ---------- InstalledIndex methods ----------

bool InstalledIndex::is_installed(std::string_view model_name) const {
  const std::string canon = normalize_installed_name(model_name);
  for (const auto& m : models) {
    if (m.canonical == canon) return true;
    // Also try stripping provider prefix from the query.
    const auto slash = canon.rfind('/');
    if (slash != std::string::npos && m.canonical == canon.substr(slash + 1)) return true;
  }
  return false;
}

std::vector<std::string> InstalledIndex::installed_providers(std::string_view model_name) const {
  const std::string canon = normalize_installed_name(model_name);
  std::vector<std::string> sources;
  for (const auto& m : models) {
    if (m.canonical == canon) {
      if (std::find(sources.begin(), sources.end(), m.source) == sources.end()) sources.push_back(m.source);
    }
  }
  return sources;
}

nlohmann::json to_json(const InstalledIndex& idx) {
  nlohmann::json models = nlohmann::json::array();
  for (const auto& m : idx.models)
    models.push_back({{"name", m.name}, {"canonical", m.canonical}, {"source", m.source}});
  return {{"models", models}, {"active_sources", idx.active_sources}, {"elapsed_ms", idx.elapsed_ms}};
}

}  // namespace mqde::fit
