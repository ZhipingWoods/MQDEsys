// Copyright 2024 MQDEsys
// SPDX-License-Identifier: Apache-2.0
#include "mqde/fit/installed.h"
#include "mqde/fit/fit.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <regex>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include "mqde/common/config.h"
#include "mqde/common/log.h"

namespace mqde::fit {
namespace fs = std::filesystem;
using Json = nlohmann::json;

Result<InstalledConfig> load_installed_config(const std::string& repo_root) {
  const fs::path p = fs::path(repo_root) / "data" / "installed.yaml";
  std::ifstream f(p);
  if (!f) return make_error(Errc::kNotFound, "cannot open '{}'", p.string());
  YAML::Node root;
  try {
    root = YAML::Load(f);
  } catch (const std::exception& e) {
    return make_error(Errc::kIo, "YAML parse error in '{}': {}", p.string(), e.what());
  }
  InstalledConfig cfg;
  for (const auto& s : root["sources"]) {
    InstalledSource src;
    src.name = s["name"].as<std::string>();
    src.type = s["type"].as<std::string>();
    if (s["url"]) src.url = s["url"].as<std::string>();
    if (s["paths"])
      for (const auto& pa : s["paths"]) src.paths.push_back(pa.as<std::string>());
    if (s["pattern"]) src.pattern = s["pattern"].as<std::string>();
    if (s["timeout_ms"]) src.timeout_ms = s["timeout_ms"].as<int>();
    cfg.sources.push_back(src);
  }
  return cfg;
}

std::string normalize_model_name(const std::string& name) {
  std::string out;
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)))
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

static std::set<std::string> probe_http(const InstalledSource& src) {
  std::set<std::string> names;
  try {
    httplib::Client cli(src.url);
    cli.set_connection_timeout(0, src.timeout_ms * 1000);
    cli.set_read_timeout(0, src.timeout_ms * 1000);
    auto res = cli.Get(src.url);
    if (!res || res->status != 200) return names;
    const Json j = Json::parse(res->body, nullptr, false);
    if (j.is_discarded()) return names;
    // ollama: {"models": [{"name": "..."}, ...]}
    if (j.contains("models") && j["models"].is_array())
      for (const auto& m : j["models"])
        if (m.contains("name") && m["name"].is_string()) names.insert(normalize_model_name(m["name"].get<std::string>()));
    // OpenAI-style: {"data": [{"id": "..."}, ...]}
    if (j.contains("data") && j["data"].is_array())
      for (const auto& m : j["data"])
        if (m.contains("id") && m["id"].is_string()) names.insert(normalize_model_name(m["id"].get<std::string>()));
  } catch (...) {
  }
  return names;
}

static std::set<std::string> probe_filesystem(const InstalledSource& src) {
  std::set<std::string> names;
  std::regex pat(src.pattern.empty() ? ".*" : src.pattern);
  for (const auto& p : src.paths) {
    fs::path dir = p;
    if (p.starts_with("$")) {
      const auto eq = p.find('=');
      std::string var = p.substr(1, eq == std::string::npos ? std::string::npos : eq - 1);
      if (const char* v = std::getenv(var.c_str())) dir = fs::path(v) / (eq == std::string::npos ? "" : p.substr(eq + 1));
    } else if (p.starts_with("~/")) {
      if (const char* h = std::getenv("HOME")) dir = fs::path(h) / p.substr(2);
    }
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) continue;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      if (!e.is_directory(ec)) continue;
      std::string n = e.path().filename().string();
      if (!std::regex_search(n, pat)) continue;
      // "models--org--name" → "org/name"
      if (n.starts_with("models--")) {
        n = n.substr(8);
        for (size_t i = 0; i < n.size(); ++i)
          if (n[i] == '-' && i + 1 < n.size() && n[i + 1] == '-') {
            n[i] = '/';
            n.erase(i + 1, 1);
          }
      }
      names.insert(normalize_model_name(n));
    }
  }
  return names;
}

Result<InstalledModels> detect_installed(const InstalledConfig& cfg, int deadline_ms) {
  const auto t0 = std::chrono::steady_clock::now();
  InstalledModels out;
  std::vector<std::future<std::pair<std::string, std::set<std::string>>>> futures;
  for (const auto& src : cfg.sources) {
    futures.push_back(std::async(std::launch::async, [&src]() -> std::pair<std::string, std::set<std::string>> {
      if (src.type == "http") return {src.name, probe_http(src)};
      if (src.type == "filesystem") return {src.name, probe_filesystem(src)};
      return {src.name, {}};
    }));
  }
  const auto deadline = t0 + std::chrono::milliseconds(deadline_ms);
  for (auto& fu : futures) {
    const auto rem = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    if (rem.count() <= 0 || fu.wait_for(rem) == std::future_status::timeout) continue;
    try {
      auto [prov, ns] = fu.get();
      for (const auto& n : ns) {
        out.all_names.insert(n);
        out.provider_models.emplace_back(prov, n);
      }
    } catch (...) {
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return out;
}

bool is_installed(const std::set<std::string>& installed, const std::string& canonical_name,
                  const std::string& ollama_style_name) {
  if (installed.count(normalize_model_name(canonical_name))) return true;
  if (!ollama_style_name.empty() && installed.count(normalize_model_name(ollama_style_name))) return true;
  return false;
}

std::vector<std::string> installed_providers(const std::vector<std::pair<std::string, std::string>>& pm,
                                              const std::string& canonical_name) {
  const std::string norm = normalize_model_name(canonical_name);
  std::vector<std::string> out;
  for (const auto& [p, n] : pm)
    if (normalize_model_name(n) == norm) out.push_back(p);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

}  // namespace mqde::fit


std::vector<std::string> mqde::fit::installed_providers(const mqde::fit::InstalledIndex& idx, const std::string& model_name) {
  return installed_providers(idx.provider_models, model_name);
}
