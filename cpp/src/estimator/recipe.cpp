#include <algorithm>

#include "mqde/estimator/estimator.h"

namespace mqde::estimator {
namespace fs = std::filesystem;

Result<Recipe> load_recipe(std::string_view name_or_path) {
  MQDE_ASSIGN_OR_RETURN(fs::path path, resolve_data_file("recipes", name_or_path));
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(path));
  const std::string where = path.filename().string();
  Recipe r;
  MQDE_ASSIGN_OR_RETURN(r.name, get_required<std::string>(j, "name", where));
  r.description = get_or<std::string>(j, "description", "");
  r.maturity = get_or<std::string>(j, "maturity", "experimental");
  r.default_algorithm = get_or<std::string>(j, "default_algorithm", "rtn");
  r.modalities = get_or<std::vector<std::string>>(j, "modalities", {"language"});
  MQDE_ASSIGN_OR_RETURN(r.weight, quant::parse_quant_spec(j.value("weight", Json()), where + ".weight"));
  MQDE_ASSIGN_OR_RETURN(r.act, quant::parse_quant_spec(j.value("act", Json()), where + ".act"));
  MQDE_ASSIGN_OR_RETURN(r.kv, quant::parse_quant_spec(j.value("kv", Json()), where + ".kv"));
  if (r.act.enabled() && r.act.gran == quant::Granularity::kPerGroup)
    return make_error(Errc::kInvalidArgument, "{}.act: per_group activation quantization is not supported",
                      where);
  return r;
}

Result<HardwareProfile> load_hardware(std::string_view id_or_path) {
  MQDE_ASSIGN_OR_RETURN(fs::path path, resolve_data_file("hardware_db", id_or_path));
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(path));
  const std::string where = path.filename().string();
  HardwareProfile h;
  MQDE_ASSIGN_OR_RETURN(h.id, get_required<std::string>(j, "id", where));
  h.vendor = get_or<std::string>(j, "vendor", "");
  h.name = get_or<std::string>(j, "name", h.id);
  h.arch = get_or<std::string>(j, "arch", "");
  h.match_names = get_or<std::vector<std::string>>(j, "match_names", {});
  MQDE_ASSIGN_OR_RETURN(h.mem_bytes, get_required<uint64_t>(j, "mem_bytes", where));
  MQDE_ASSIGN_OR_RETURN(h.mem_bw_gbps, get_required<double>(j, "mem_bw_gbps", where));
  h.peak_tflops = get_or<std::map<std::string, double>>(j, "peak_tflops", {});
  h.native_formats = get_or<std::vector<std::string>>(j, "native_formats", {});
  h.source = get_or<std::string>(j, "source", "");
  return h;
}

Result<BackendProfile> load_backend(std::string_view id_or_path) {
  MQDE_ASSIGN_OR_RETURN(fs::path path, resolve_data_file("backends", id_or_path));
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(path));
  BackendProfile b;
  b.id = get_or<std::string>(j, "id", b.id);
  b.runtime_overhead_bytes = get_or<uint64_t>(j, "runtime_overhead_bytes", b.runtime_overhead_bytes);
  b.allocator_factor = get_or<double>(j, "allocator_factor", b.allocator_factor);
  b.eta_bw = get_or<double>(j, "eta_bw", b.eta_bw);
  b.mfu = get_or<double>(j, "mfu", b.mfu);
  b.activation_model = get_or<std::string>(j, "activation_model", b.activation_model);
  b.max_batched_tokens = get_or<int64_t>(j, "max_batched_tokens", b.max_batched_tokens);
  if (b.activation_model != "per_seq" && b.activation_model != "batched")
    return make_error(Errc::kInvalidArgument,
                      "backend '{}': activation_model must be per_seq or batched, got '{}'", b.id,
                      b.activation_model);
  if (b.max_batched_tokens <= 0)
    return make_error(Errc::kInvalidArgument, "backend '{}': max_batched_tokens must be positive", b.id);
  return b;
}

Result<std::vector<std::string>> list_data_entries(std::string_view sub) {
  const auto dir = data_dir() / sub;
  if (!fs::is_directory(dir)) return make_error(Errc::kNotFound, "data directory '{}' missing", dir.string());
  std::vector<std::string> out;
  for (const auto& e : fs::recursive_directory_iterator(dir)) {
    if (e.path().extension() != ".yaml") continue;
    auto rel = fs::relative(e.path(), dir);
    rel.replace_extension();
    out.push_back(rel.generic_string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

Json to_json(const Recipe& r) {
  return {{"name", r.name},
          {"description", r.description},
          {"maturity", r.maturity},
          {"default_algorithm", r.default_algorithm},
          {"modalities", r.modalities},
          {"weight", quant::to_json(r.weight)},
          {"act", quant::to_json(r.act)},
          {"kv", quant::to_json(r.kv)},
          {"summary", fmt::format("W: {} | A: {} | KV: {}", quant::describe(r.weight), quant::describe(r.act),
                                  quant::describe(r.kv))}};
}

Json to_json(const HardwareProfile& h) {
  return {{"id", h.id},
          {"vendor", h.vendor},
          {"name", h.name},
          {"arch", h.arch},
          {"mem_bytes", h.mem_bytes},
          {"mem_bw_gbps", h.mem_bw_gbps},
          {"peak_tflops", h.peak_tflops},
          {"native_formats", h.native_formats},
          {"source", h.source}};
}

}  // namespace mqde::estimator
