#include "mqde/common/config.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

namespace mqde {
namespace {

Json yaml_to_json(const YAML::Node& n) {
  switch (n.Type()) {
    case YAML::NodeType::Null:
    case YAML::NodeType::Undefined:
      return nullptr;
    case YAML::NodeType::Sequence: {
      Json arr = Json::array();
      for (const auto& e : n) arr.push_back(yaml_to_json(e));
      return arr;
    }
    case YAML::NodeType::Map: {
      Json obj = Json::object();
      for (const auto& kv : n) obj[kv.first.as<std::string>()] = yaml_to_json(kv.second);
      return obj;
    }
    case YAML::NodeType::Scalar: {
      const std::string& s = n.Scalar();
      // Quoted scalars stay strings; plain scalars get typed.
      if (n.Tag() == "!") return s;
      if (s == "true" || s == "True") return true;
      if (s == "false" || s == "False") return false;
      if (s == "null" || s == "~") return nullptr;
      char* end = nullptr;
      long long i = std::strtoll(s.c_str(), &end, 10);
      if (!s.empty() && end && *end == '\0') return i;
      double d = std::strtod(s.c_str(), &end);
      if (!s.empty() && end && *end == '\0') return d;
      return s;
    }
  }
  return nullptr;
}

}  // namespace

Result<Json> load_structured_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return make_error(Errc::kNotFound, "cannot open '{}'", path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  const auto ext = path.extension().string();
  try {
    if (ext == ".json") return Json::parse(ss.str());
    return yaml_to_json(YAML::Load(ss.str()));
  } catch (const std::exception& e) {
    return make_error(Errc::kParse, "failed to parse '{}': {}", path.string(), e.what());
  }
}

std::filesystem::path data_dir() {
  if (const char* env = std::getenv("MQDE_DATA_DIR"); env && *env) return env;
  return MQDE_DEFAULT_DATA_DIR;
}

Result<std::filesystem::path> resolve_data_file(std::string_view sub, std::string_view name) {
  std::filesystem::path direct(name);
  if (std::filesystem::is_regular_file(direct)) return direct;
  const auto dir = data_dir() / sub;
  for (const char* ext : {".yaml", ".yml", ".json"}) {
    auto p = dir / (std::string(name) + ext);
    if (std::filesystem::is_regular_file(p)) return p;
  }
  // Allow case-insensitive recipe names like "W4A16".
  std::string lower(name);
  for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lower != name) return resolve_data_file(sub, lower);
  return make_error(Errc::kNotFound, "'{}' not found under {}", name, dir.string());
}

std::string format_bytes(double bytes) {
  const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  int u = 0;
  while (bytes >= 1024.0 && u < 4) {
    bytes /= 1024.0;
    ++u;
  }
  return u == 0 ? fmt::format("{:.0f} {}", bytes, units[u]) : fmt::format("{:.2f} {}", bytes, units[u]);
}

}  // namespace mqde
