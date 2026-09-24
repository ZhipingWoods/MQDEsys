#include "mqde/estimator/estimator.h"

namespace mqde::estimator {

Result<HardwareProfile> match_hardware(std::string_view device_name) {
  MQDE_ASSIGN_OR_RETURN(auto ids, list_data_entries("hardware_db"));
  for (const auto& id : ids) {
    auto h = load_hardware(id);
    if (!h) continue;
    for (const auto& n : h->match_names)
      if (device_name.find(n) != std::string_view::npos) return h;
  }
  return make_error(Errc::kNotFound, "no hardware profile matches device '{}'", device_name);
}

}  // namespace mqde::estimator
