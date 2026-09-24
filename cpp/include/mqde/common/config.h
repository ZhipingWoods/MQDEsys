#pragma once
// File loading helpers: YAML/JSON -> nlohmann::json, data directory resolution, units.

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "mqde/common/result.h"

namespace mqde {

using Json = nlohmann::json;

// Loads a .yaml/.yml/.json file into JSON (YAML scalars are typed: int, float, bool, string).
Result<Json> load_structured_file(const std::filesystem::path& path);

// Data directory: $MQDE_DATA_DIR, else the source tree's data/ baked in at build time.
std::filesystem::path data_dir();

// Resolves "<data_dir>/<sub>/<name>.yaml"; `name` may also be an existing file path.
Result<std::filesystem::path> resolve_data_file(std::string_view sub, std::string_view name);

// Typed accessor with a JSON-pointer-style path in the error ("text_config.hidden_size").
template <class T>
Result<T> get_required(const Json& j, std::string_view key, std::string_view where) {
  auto it = j.find(std::string(key));
  if (it == j.end() || it->is_null())
    return make_error(Errc::kParse, "{}: missing required field '{}'", where, key);
  try {
    return it->get<T>();
  } catch (const std::exception& e) {
    return make_error(Errc::kParse, "{}: field '{}' has wrong type ({})", where, key, e.what());
  }
}

template <class T>
T get_or(const Json& j, std::string_view key, T fallback) {
  auto it = j.find(std::string(key));
  if (it == j.end() || it->is_null()) return fallback;
  try {
    return it->get<T>();
  } catch (...) {
    return fallback;
  }
}

// "3.39 GiB" style formatting for bytes.
std::string format_bytes(double bytes);

}  // namespace mqde
