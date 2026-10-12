#include "mqde/planner/validate.h"

#include <algorithm>
#include <set>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace mqde::planner {

using quant::NumFormat;
using quant::QuantSpec;

std::string_view verdict_name(Verdict v) {
  switch (v) {
    case Verdict::kNative:
      return "native";
    case Verdict::kEmulated:
      return "emulated";
    default:
      return "unsupported";
  }
}

namespace {

std::string weight_format_name(const QuantSpec& s) {
  if (!s.enabled()) return "none";
  if (s.fmt == NumFormat::kFp8E4M3) return "fp8_e4m3";
  if (s.fmt == NumFormat::kFp8E5M2) return "fp8_e5m2";
  return fmt::format("int{}", s.bits);
}

// The matrix's "weight_format" key is the format family (what the container needs), not the display name with
// its bit count: an int4 and an int8 recipe are both "int" as far as a container's format requirement goes;
// bits are checked separately via bits_ok.
std::string weight_format_family(const QuantSpec& s) {
  if (!s.enabled()) return "none";
  if (s.fmt == NumFormat::kFp8E4M3) return "fp8_e4m3";
  if (s.fmt == NumFormat::kFp8E5M2) return "fp8_e5m2";
  return "int";
}

Result<Json> matrix() {
  MQDE_ASSIGN_OR_RETURN(auto path, resolve_data_file("compat", "matrix"));
  return load_structured_file(path);
}

}  // namespace

Result<std::vector<AlgorithmInfo>> list_algorithms() {
  MQDE_ASSIGN_OR_RETURN(auto path, resolve_data_file("", "algorithms"));
  MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(path));
  std::vector<AlgorithmInfo> out;
  for (const auto& a : j.value("algorithms", Json::array()))
    out.push_back({a.value("name", ""), a.value("label", ""), a.value("kind", ""), a.value("description", ""),
                   a.value("needs_calibration", false)});
  return out;
}

Result<AlgorithmInfo> load_algorithm(const std::string& name) {
  MQDE_ASSIGN_OR_RETURN(auto all, list_algorithms());
  for (const auto& a : all)
    if (a.name == name) return a;
  std::vector<std::string> names;
  for (const auto& a : all) names.push_back(a.name);
  return make_error(Errc::kNotFound, "unknown algorithm '{}' (known: {})", name, fmt::join(names, ", "));
}

Result<std::vector<std::string>> resolve_algorithm(const std::string& name) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : name + "+") {
    if (c == '+') {
      std::string p = cur;
      std::transform(p.begin(), p.end(), p.begin(), ::tolower);
      p.erase(0, p.find_first_not_of(" \t"));
      if (auto e = p.find_last_not_of(" \t"); e != std::string::npos) p.erase(e + 1);
      if (!p.empty() && p != "none") parts.push_back(p);
      cur.clear();
    } else {
      cur += c;
    }
  }
  if (parts.empty()) return std::vector<std::string>{"rtn"};
  MQDE_ASSIGN_OR_RETURN(auto all, list_algorithms());
  std::set<std::string> known;
  for (const auto& a : all) known.insert(a.name);
  std::vector<std::string> names(known.begin(), known.end());
  for (const auto& p : parts)
    if (!known.count(p))
      return make_error(Errc::kInvalidArgument, "unknown quantization algorithm '{}' (known: {})", p,
                        fmt::join(names, ", "));
  std::vector<std::string> transforms, quantizers;
  for (const auto& p : parts)
    if (p == "awq" || p == "smoothquant")
      transforms.push_back(p);
    else if (p == "rtn" || p == "gptq")
      quantizers.push_back(p);
  if (transforms.size() > 1 || quantizers.size() > 1)
    return make_error(Errc::kInvalidArgument,
                      "algorithm '{}': at most one transform (awq|smoothquant) and one quantizer (rtn|gptq)",
                      name);
  std::vector<std::string> out = transforms;
  if (quantizers.empty())
    out.push_back("rtn");
  else
    out.insert(out.end(), quantizers.begin(), quantizers.end());
  return out;
}

CompatResult check_export(const std::string& format, const QuantSpec& weight, const QuantSpec& act,
                          std::optional<LayerShape> shape) {
  auto m = matrix();
  if (!m) return {Verdict::kUnsupported, m.error().message, ""};
  const auto& ef = (*m)["export_formats"];
  const std::string key = format == "compressed-tensors"
                              ? (weight.fmt == NumFormat::kFp8E4M3 || weight.fmt == NumFormat::kFp8E5M2
                                     ? "compressed-tensors-fp8"
                                     : "compressed-tensors-int")
                              : format;
  if (!ef.contains(key))
    return {Verdict::kUnsupported, fmt::format("unknown export format '{}'", format), ""};
  const auto& r = ef[key];
  const std::string want_fmt = r.value("weight_format", "");
  if (weight_format_family(weight) != want_fmt &&
      !(key == "compressed-tensors-fp8" && weight.fmt == NumFormat::kFp8E5M2 && want_fmt == "fp8_e4m3"))
    return {
        Verdict::kUnsupported,
        fmt::format("{} needs {} weights, this recipe has {}", format, want_fmt, weight_format_name(weight)),
        ""};
  if (r.value("rejects_per_tensor", false) && weight.gran == quant::Granularity::kPerTensor)
    return {Verdict::kUnsupported, "per-tensor weight scales are not representable in this container", ""};
  if (r.value("rejects_per_group", false) && weight.gran == quant::Granularity::kPerGroup)
    return {Verdict::kUnsupported,
            "per-group FP8 weight scales are not exported (use per-channel or per-tensor)", ""};
  if (r.contains("bits_ok")) {
    bool ok = false;
    for (const auto& b : r["bits_ok"]) ok = ok || b.get<int>() == weight.bits;
    if (!ok)
      return {Verdict::kUnsupported,
              fmt::format("{}-bit weights are not supported by {} (supported: {})", weight.bits, format,
                          r["bits_ok"].dump()),
              ""};
  }
  if (r.value("rejects_act", false) && act.enabled())
    return {Verdict::kUnsupported,
            "integer weights with quantized activations (W4A8/W8A8-int) are not exported yet", ""};
  if (r.value("act_format_must_match", false) && act.enabled() &&
      weight_format_family(act) != weight_format_family(weight))
    return {Verdict::kUnsupported,
            fmt::format("{} weights with {} activations are not a compressed-tensors scheme",
                        weight_format_name(weight), weight_format_name(act)),
            ""};
  // The real rule here ("a *static* activation scale must be per-tensor", compressed_tensors.py) depends on
  // whether a static scale was actually computed for a specific model instance (`m.act_scale is not None`),
  // which is a per-job runtime fact, not something the recipe's QuantSpec declares (it has no "dynamic"
  // field). It is not checked here; re-add it if QuantSpec ever gains that information.
  if (shape) {
    const int64_t gs = weight.gran == quant::Granularity::kPerGroup ? weight.group_size : shape->in_features;
    if (gs > 0 && r.value("needs_in_group", false) && shape->in_features % gs)
      return {Verdict::kUnsupported,
              fmt::format("in_features {} is not divisible by group_size {}", shape->in_features, gs), ""};
    const int64_t pf = weight.bits > 0 ? 32 / weight.bits : 0;
    if (pf > 0) {
      if (r.value("needs_in_pack", false) && shape->in_features % pf)
        return {Verdict::kUnsupported,
                fmt::format("in_features {} does not pack into 32-bit words at {} bit", shape->in_features,
                            weight.bits),
                ""};
      if (r.value("needs_out_pack", false) && shape->out_features % pf)
        return {Verdict::kUnsupported,
                fmt::format("out_features {} does not pack into 32-bit words at {} bit", shape->out_features,
                            weight.bits),
                ""};
      if (r.value("needs_out_pack_zero_point", false) && !weight.symmetric && shape->out_features % pf)
        return {Verdict::kUnsupported,
                fmt::format("out_features {} does not pack zero points into 32-bit words at {} bit",
                            shape->out_features, weight.bits),
                ""};
    }
  }
  return {Verdict::kNative, "", format};
}

CompatResult check_deploy(const std::string& format, const estimator::Recipe& recipe,
                          const std::string& backend, const estimator::HardwareProfile& hw) {
  auto m = matrix();
  if (!m) return {Verdict::kUnsupported, m.error().message, ""};
  const auto& backends = (*m)["backends"];
  if (!backends.contains(backend))
    return {Verdict::kUnsupported, fmt::format("unknown backend '{}'", backend), ""};
  const auto& b = backends[backend];
  const auto loads = b.value("loads", Json::array());
  // GGUF re-quantizes from the original FP weights with llama.cpp's own numerics (python/mqde/pipeline/
  // llm_job.py's _llamacpp), independent of the recipe's weight spec -- so whether it fits a device has
  // nothing to do with `recipe.weight`, and checking it against hw.native_formats below would be checking
  // the wrong thing. If the backend loads gguf at all, it is simply native.
  if (format == "gguf") {
    bool loads_gguf = false;
    for (const auto& f : loads) loads_gguf = loads_gguf || f.get<std::string>() == "gguf";
    return loads_gguf ? CompatResult{Verdict::kNative, "", "gguf"}
                      : CompatResult{Verdict::kUnsupported,
                                     fmt::format("{} does not load gguf checkpoints", backend), ""};
  }
  const std::string key = format == "compressed-tensors" ? (recipe.weight.fmt == NumFormat::kFp8E4M3 ||
                                                                    recipe.weight.fmt == NumFormat::kFp8E5M2
                                                                ? "compressed-tensors-fp8"
                                                                : "compressed-tensors-int")
                                                         : format;
  bool loads_it = false;
  for (const auto& f : loads) loads_it = loads_it || f.get<std::string>() == key;
  if (!loads_it)
    return {Verdict::kUnsupported, fmt::format("{} does not load {} checkpoints", backend, format), ""};
  for (const auto& issue : b.value("known_issues", Json::array())) {
    if (issue.value("format", "") != key) continue;
    const auto& when = issue.value("when", Json::object());
    bool matches = true;
    if (when.contains("symmetric"))
      matches = matches && (recipe.weight.symmetric == when["symmetric"].get<bool>());
    if (matches) {
      const std::string v = issue.value("verdict", "unsupported");
      return {v == "emulated" ? Verdict::kEmulated : Verdict::kUnsupported, issue.value("reason", ""), ""};
    }
  }
  const std::string wfmt = weight_format_name(recipe.weight);
  const bool native =
      std::find(hw.native_formats.begin(), hw.native_formats.end(), wfmt) != hw.native_formats.end();
  if (!native)
    return {Verdict::kEmulated,
            fmt::format("{} weights have no native kernel on {} ({}); accuracy can still be evaluated with "
                        "fake quantization",
                        wfmt, hw.name, hw.arch),
            format};
  return {Verdict::kNative, "", format};
}

// ---------------------------------------------------------------- validate_jobspec

namespace {

void add(std::vector<Problem>& out, std::string field, std::string message, std::string suggestion = "") {
  out.push_back({std::move(field), std::move(message), std::move(suggestion)});
}

}  // namespace

std::vector<Problem> validate_jobspec(const Json& job) {
  std::vector<Problem> out;

  const auto recipes = job.value(Json::json_pointer("/quant/recipe"), Json::array());
  if (!recipes.is_array() || recipes.empty()) {
    add(out, "/quant/recipe", "must be a non-empty list of recipe names",
        "pick at least one of w4a16, w4a8, w2a16, w2a8, fp8, bf16");
  } else {
    for (const auto& r : recipes) {
      if (!r.is_string()) {
        add(out, "/quant/recipe", "every entry must be a string", "");
        continue;
      }
      if (auto rr = estimator::load_recipe(r.get<std::string>()); !rr)
        add(out, "/quant/recipe", rr.error().message, "run `mqde recipes` for the list of valid names");
    }
  }

  const auto algo = job.value(Json::json_pointer("/quant/algorithm"), std::string("recipe"));
  std::vector<std::string> algo_parts;
  if (algo != "recipe") {
    if (auto a = resolve_algorithm(algo); !a)
      add(out, "/quant/algorithm", a.error().message, "run `mqde` with GET /api/v1/algorithms for the list");
    else
      algo_parts = *a;
    for (const auto& p : algo_parts) {
      auto info = load_algorithm(p);
      if (info && info->needs_calibration) {
        const int n = job.value(Json::json_pointer("/calibration/n_samples"), 0);
        if (n <= 0)
          add(out, "/calibration/n_samples",
              fmt::format("algorithm '{}' needs calibration data ({} samples requested)", p, n),
              "set calibration.n_samples to a positive number, or use algorithm rtn");
      }
    }
  }

  const auto mtype = job.value(Json::json_pointer("/model/type"), std::string("vla"));
  if (mtype != "vla" && mtype != "llm")
    add(out, "/model/type", fmt::format("'{}' is not 'vla' or 'llm'", mtype), "");

  // domain (T-1301/T-1305): optional today (every existing caller keys off model.type instead), but when
  // present it must name a registered domain whose pipeline actually exists -- rejecting here, before
  // JobRunner ever picks a Python module to launch, is how a domain with no backend yet (vlm/world/ewm)
  // fails with a clear reason instead of silently running the wrong pipeline or crashing deep inside it.
  if (job.contains("domain")) {
    const auto domain_id = job.value(Json::json_pointer("/domain"), std::string());
    if (auto d = estimator::load_domain(domain_id); !d) {
      add(out, "/domain", fmt::format("'{}' is not a registered domain", domain_id),
          "GET /api/v1/domains for the list of valid ids");
    } else if (d->pipeline_status != "implemented") {
      add(out, "/domain", fmt::format("domain '{}' has no real quantization pipeline yet", domain_id), "");
    }
  }
  // quantize_only (T-1304/T-1306): calibration + quantization only. Every stage that reads export/deploy/
  // sensitivity/logits_probe/attribution needs an evaluated baseline this skips (see llm_job.py's and
  // vla_job.py's own quantize_only branches), so combining them is a job the Python pipeline cannot honor.
  if (job.value(Json::json_pointer("/evaluation/skip"), false)) {
    if (mtype == "llm") {
      if (!job.value(Json::json_pointer("/export/formats"), Json::array()).empty())
        add(out, "/evaluation/skip",
            "quantize-only is incompatible with export.formats (export needs the fake-quant PPL this skips)",
            "remove export.formats, or turn off evaluation.skip");
      if (job.contains("deploy") && job["deploy"].is_object() && job["deploy"].contains("vllm"))
        add(out, "/evaluation/skip", "quantize-only is incompatible with deploy.vllm",
            "remove deploy.vllm, or turn off evaluation.skip");
      if (job.contains("deploy") && job["deploy"].is_object() && job["deploy"].contains("sglang"))
        add(out, "/evaluation/skip", "quantize-only is incompatible with deploy.sglang",
            "remove deploy.sglang, or turn off evaluation.skip");
      if (!job.value(Json::json_pointer("/deploy/llamacpp/types"), Json::array()).empty())
        add(out, "/evaluation/skip", "quantize-only is incompatible with deploy.llamacpp",
            "remove deploy.llamacpp, or turn off evaluation.skip");
      if (job.contains("analysis") && job["analysis"].is_object() &&
          (job["analysis"].contains("sensitivity") || job["analysis"].contains("logits_probe")))
        add(out, "/evaluation/skip",
            "quantize-only is incompatible with analysis.sensitivity/logits_probe (both need an evaluated "
            "baseline)",
            "remove them, or turn off evaluation.skip");
    } else if (mtype == "vla") {
      if (job.value(Json::json_pointer("/evaluation/attribution"), true))
        add(out, "/evaluation/skip",
            "quantize-only is incompatible with evaluation.attribution=true (attribution needs the BF16 "
            "baseline this skips)",
            "set evaluation.attribution to false, or turn off evaluation.skip");
    }
  }
  const auto device = job.value(Json::json_pointer("/deploy/device"), std::string());
  if (device == "cpu" && mtype != "llm")
    add(out, "/deploy/device", "cpu is only supported for model.type=llm",
        "set deploy.device to 'gpu', or model.type to 'llm'");

  if (const auto export_formats = job.value(Json::json_pointer("/export/formats"), Json::array());
      !export_formats.empty() && mtype == "llm") {
    // Export compatibility needs a recipe's actual weight/act spec, which needs the recipe to have loaded
    // above.
    for (const auto& r : recipes) {
      if (!r.is_string()) continue;
      auto rr = estimator::load_recipe(r.get<std::string>());
      if (!rr) continue;  // already reported against /quant/recipe
      for (const auto& f : export_formats) {
        if (!f.is_string()) continue;
        auto c = check_export(f.get<std::string>(), rr->weight, rr->act);
        if (!c.ok())
          add(out, "/export/formats",
              fmt::format("recipe '{}' cannot be exported as {}: {}", rr->name, f.get<std::string>(),
                          c.reason),
              "pick a different export format, or a recipe whose weight format that container supports");
      }
    }
  }

  return out;
}

// ---------------------------------------------------------------- recommend

std::string preferred_export_format(const QuantSpec& weight, const std::string& backend) {
  if (backend == "llamacpp") return "gguf";  // ignores weight entirely; see check_deploy's matching case
  if (!weight.enabled()) return "";  // bf16: nothing to export
  if (weight.fmt == NumFormat::kFp8E4M3 || weight.fmt == NumFormat::kFp8E5M2) return "compressed-tensors";
  return weight.bits == 4 ? "awq" : "gptq";
}

RecommendResult recommend(const modelspec::ModelSpec& spec,
                          const std::vector<estimator::BackendProfile>& backends,
                          const std::vector<estimator::HardwareProfile>& hardware,
                          const estimator::RuntimeParams& rt, int top_k) {
  RecommendResult out;
  auto recipe_names = estimator::list_data_entries("recipes");
  if (!recipe_names) return out;
  std::vector<Recommendation> all;
  for (const auto& rn : *recipe_names) {
    auto recipe = estimator::load_recipe(rn);
    if (!recipe) continue;
    for (const auto& hw : hardware) {
      for (const auto& be : backends) {
        const auto est = estimator::estimate(spec, *recipe, hw, be, rt);
        Recommendation rec;
        rec.recipe = recipe->name;
        rec.algorithm = recipe->default_algorithm;
        rec.maturity = recipe->maturity;
        rec.backend = be.id;
        rec.hardware = hw.id;
        rec.feasible = est.feasible;
        rec.peak_bytes = est.peak_bytes;
        rec.capacity_bytes = est.capacity_bytes;
        rec.max_concurrency = est.max_concurrency;
        rec.export_format = preferred_export_format(recipe->weight, be.id);
        if (!est.feasible) {
          rec.reasons = est.reasons;
          rec.reasons.push_back(fmt::format("peak {} exceeds capacity {} at {}% utilization", est.peak_bytes,
                                            est.capacity_bytes,
                                            static_cast<int>(rt.gpu_memory_utilization * 100)));
          ++out.excluded;
          continue;
        }
        if (!rec.export_format.empty()) {
          const auto dep = check_deploy(rec.export_format, *recipe, be.id, hw);
          rec.deploy_verdict = dep.verdict;
          if (dep.verdict == Verdict::kUnsupported) {
            rec.reasons.push_back(
                fmt::format("{} export does not deploy on {}: {}", rec.export_format, be.id, dep.reason));
            ++out.excluded;
            continue;
          }
          rec.reasons.push_back(dep.verdict == Verdict::kNative
                                    ? fmt::format("native {} kernel on {}", rec.export_format, hw.id)
                                    : fmt::format("emulated: {}", dep.reason));
        } else {
          rec.deploy_verdict = Verdict::kNative;
          rec.reasons.push_back("bf16: no quantization, no export needed");
        }
        rec.reasons.push_back(fmt::format("fits: peak {:.1f} GiB of {:.1f} GiB capacity",
                                          static_cast<double>(est.peak_bytes) / (1 << 30),
                                          static_cast<double>(est.capacity_bytes) / (1 << 30)));
        rec.reasons.push_back(
            fmt::format("{} recipe, estimated max_concurrency {}", recipe->maturity, est.max_concurrency));
        all.push_back(std::move(rec));
      }
    }
  }
  std::stable_sort(all.begin(), all.end(), [](const Recommendation& a, const Recommendation& b) {
    if ((a.deploy_verdict == Verdict::kNative) != (b.deploy_verdict == Verdict::kNative))
      return a.deploy_verdict == Verdict::kNative;
    if (a.max_concurrency != b.max_concurrency) return a.max_concurrency > b.max_concurrency;
    if ((a.maturity == "production") != (b.maturity == "production")) return a.maturity == "production";
    return a.recipe < b.recipe;
  });
  if (static_cast<int>(all.size()) > top_k) all.resize(static_cast<size_t>(top_k));
  out.top = std::move(all);
  return out;
}

}  // namespace mqde::planner
