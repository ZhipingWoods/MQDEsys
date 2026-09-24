#include <algorithm>

#include "mqde/modelspec/model_spec.h"

namespace mqde::modelspec {
namespace fs = std::filesystem;

Result<ModelFamily> load_model_family(const fs::path& file) {
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(file));
  const std::string where = file.filename().string();
  ModelFamily f;
  MQDE_ASSIGN_OR_RETURN(f.family, get_required<std::string>(j, "family", where));
  f.modality = get_or<std::string>(j, "modality", "llm");
  if (auto m = j.find("match"); m != j.end())
    f.match_model_types = get_or<std::vector<std::string>>(*m, "model_type", {});
  if (auto p = j.find("config_paths"); p != j.end()) {
    f.text_config_path = get_or<std::vector<std::string>>(*p, "text", {});
    f.vision_config_path = get_or<std::vector<std::string>>(*p, "vision", {});
  }
  f.tokens_per_image = get_or<int64_t>(j, "tokens_per_image", 0);
  if (auto a = j.find("action_head"); a != j.end()) {
    f.action_horizon_key = get_or<std::string>(*a, "horizon_key", "");
    f.action_steps_key = get_or<std::string>(*a, "steps_key", "");
    f.action_query_tokens = get_or<int64_t>(*a, "num_query_tokens", 0);
  }
  for (const auto& c : j.value("components", Json::array())) {
    ComponentRule r;
    MQDE_ASSIGN_OR_RETURN(r.name, get_required<std::string>(c, "name", where + ".components"));
    MQDE_ASSIGN_OR_RETURN(r.prefix, get_required<std::string>(c, "prefix", where + ".components"));
    r.modality = get_or<std::string>(c, "modality", "language");
    r.block_regex = get_or<std::string>(c, "block_regex", "");
    f.components.push_back(std::move(r));
  }
  f.default_keep_fp = get_or<std::vector<std::string>>(j, "default_keep_fp", {});
  // Validate regexes up front so a typo fails at load time, not mid-estimate.
  try {
    for (const auto& c : f.components)
      if (!c.block_regex.empty()) (void)std::regex(c.block_regex);
    for (const auto& p : f.default_keep_fp) (void)std::regex(p);
  } catch (const std::regex_error& e) {
    return make_error(Errc::kParse, "{}: invalid regex: {}", where, e.what());
  }
  return f;
}

Result<ModelFamily> find_model_family(std::string_view model_type) {
  const auto dir = data_dir() / "model_families";
  if (!fs::is_directory(dir)) return make_error(Errc::kNotFound, "model family directory '{}' missing", dir.string());
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".yaml") files.push_back(e.path());
  std::sort(files.begin(), files.end());
  for (const auto& f : files) {
    MQDE_ASSIGN_OR_RETURN(ModelFamily fam, load_model_family(f));
    if (std::find(fam.match_model_types.begin(), fam.match_model_types.end(), model_type) !=
        fam.match_model_types.end())
      return fam;
  }
  return load_model_family(dir / "generic_decoder.yaml");
}

}  // namespace mqde::modelspec
