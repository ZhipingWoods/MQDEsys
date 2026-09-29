// Copyright 2024 MQDEsys
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "mqde/common/result.h"

namespace mqde::fit {

struct InstalledSource {
  std::string name;
  std::string type;  // "http" | "filesystem"
  std::string url;
  std::vector<std::string> paths;
  std::string pattern;
  int timeout_ms = 400;
};

struct InstalledConfig {
  std::vector<InstalledSource> sources;
};

struct InstalledModels {
  std::set<std::string> all_names;  // canonical normalized names
  std::vector<std::pair<std::string, std::string>> provider_models;  // (provider, original_name)
  double elapsed_ms = 0;

  // Returns the list of providers/runtimes that have this model installed.
  // model_name is normalized before lookup.
  std::vector<std::string> installed_providers(const std::string& model_name) const;
};


Result<InstalledConfig> load_installed_config(const std::string& repo_root = ".");

std::string normalize_model_name(const std::string& name);

Result<InstalledModels> detect_installed(const InstalledConfig& cfg, int deadline_ms = 2000);

bool is_installed(const std::set<std::string>& installed, const std::string& canonical_name,
                  const std::string& ollama_style_name = "");

std::vector<std::string> installed_providers(const std::vector<std::pair<std::string, std::string>>& pm,
                                              const std::string& canonical_name);

}  // namespace mqde::fit
