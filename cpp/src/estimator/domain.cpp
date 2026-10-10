#include "mqde/estimator/estimator.h"

namespace mqde::estimator {
namespace fs = std::filesystem;

Result<Domain> load_domain(std::string_view id_or_path) {
  MQDE_ASSIGN_OR_RETURN(fs::path path, resolve_data_file("domains", id_or_path));
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(path));
  const std::string where = path.filename().string();
  Domain d;
  MQDE_ASSIGN_OR_RETURN(d.id, get_required<std::string>(j, "id", where));
  d.display_name = get_or<std::string>(j, "display_name", d.id);
  d.model_families = get_or<std::vector<std::string>>(j, "model_families", {});
  d.applicable_recipes = get_or<std::vector<std::string>>(j, "applicable_recipes", {});
  for (const auto& a : j.value("classic_algorithms", Json::array())) {
    Domain::ClassicAlgorithm c;
    MQDE_ASSIGN_OR_RETURN(c.name, get_required<std::string>(a, "name", where + ".classic_algorithms"));
    c.note = get_or<std::string>(a, "note", "");
    d.classic_algorithms.push_back(std::move(c));
  }
  d.default_eval_dataset = get_or<std::string>(j, "default_eval_dataset", "");
  d.pipeline_status = get_or<std::string>(j, "pipeline_status", "ui_only_pending_backend");
  if (d.pipeline_status != "implemented" && d.pipeline_status != "ui_only_pending_backend")
    return make_error(Errc::kInvalidArgument,
                      "domain '{}': pipeline_status must be 'implemented' or 'ui_only_pending_backend', got '{}'",
                      d.id, d.pipeline_status);
  return d;
}

Json to_json(const Domain& d) {
  Json algos = Json::array();
  for (const auto& a : d.classic_algorithms) algos.push_back({{"name", a.name}, {"note", a.note}});
  return {{"id", d.id},
          {"display_name", d.display_name},
          {"model_families", d.model_families},
          {"applicable_recipes", d.applicable_recipes},
          {"classic_algorithms", algos},
          {"default_eval_dataset", d.default_eval_dataset},
          {"pipeline_status", d.pipeline_status}};
}

}  // namespace mqde::estimator
