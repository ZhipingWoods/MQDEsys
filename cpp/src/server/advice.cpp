#include "advice.h"

#include <algorithm>
#include <fstream>

namespace mqde::server {
namespace fs = std::filesystem;

const char* risk_name(Risk r) {
  switch (r) {
    case Risk::kLow:
      return "low";
    case Risk::kMedium:
      return "medium";
    case Risk::kHigh:
      return "high";
  }
  return "medium";
}

Risk prior_risk(const estimator::Recipe& r) {
  const int w = r.weight.enabled() ? r.weight.bits : 16;
  const int a = r.act.enabled() ? r.act.bits : 16;
  if (w <= 3 || a <= 4) return Risk::kHigh;
  if (w < 8 || a < 8) return Risk::kMedium;
  return Risk::kLow;
}

std::optional<Measured> measured_from_summary(const std::string& kind, const Json& s) {
  auto num = [&](const char* k) -> std::optional<double> {
    if (!s.is_object() || !s.contains(k) || !s[k].is_number()) return std::nullopt;
    return s[k].get<double>();
  };
  Measured m;
  m.kind = kind;
  if (kind == "llm") {
    const auto v = num("ppl_delta_pct");
    if (!v) return std::nullopt;
    m.metric = "ppl_delta_pct";
    m.value = *v;
    m.risk = *v < 3 ? Risk::kLow : *v < 15 ? Risk::kMedium : Risk::kHigh;
  } else if (kind == "vlm") {
    const auto v = num("top1_agreement");
    if (!v) return std::nullopt;
    m.metric = "top1_agreement";
    m.value = *v;
    m.risk = *v >= 0.98 ? Risk::kLow : *v >= 0.9 ? Risk::kMedium : Risk::kHigh;
  } else {
    const auto v = num("action_rel_l2_over_noise_floor");
    if (!v) return std::nullopt;
    m.metric = "action_rel_l2_over_noise_floor";
    m.value = *v;
    m.risk = *v <= 2 ? Risk::kLow : *v <= 5 ? Risk::kMedium : Risk::kHigh;
  }
  return m;
}

bool finished_ok(const Json& rep) {
  if (!rep.is_object()) return false;
  if (rep.contains("verdict")) return rep["verdict"] == "succeeded";
  // llm_job writes no verdict: done = no error and its last stage, "report", succeeded
  if (rep.contains("error")) return false;
  const auto it = rep.find("stages");
  if (it == rep.end() || !it->is_array()) return false;
  for (const auto& st : *it)
    if (st.value("stage", "") == "report") return st.value("status", "") == "succeeded";
  return false;
}

std::map<std::string, Measured> measured_for(const fs::path& runs_dir, const std::string& model_path) {
  std::map<std::string, Measured> out;
  std::error_code ec;
  const auto want = fs::weakly_canonical(model_path, ec).string();
  std::vector<fs::path> dirs;
  for (const auto& e : fs::directory_iterator(runs_dir, ec))
    if (e.is_directory()) dirs.push_back(e.path());
  std::sort(dirs.begin(), dirs.end());  // job ids start with a timestamp: later ones overwrite earlier ones
  for (const auto& d : dirs) {
    std::ifstream f(d / "report.json");
    if (!f) continue;
    const Json rep = Json::parse(f, nullptr, false);
    if (!finished_ok(rep)) continue;
    const auto path = rep.value("/spec/model_path"_json_pointer, std::string());
    if (path.empty() || fs::weakly_canonical(path, ec).string() != want) continue;
    const auto k = rep.value("kind", std::string());
    const auto kind = k == "llm" || k == "vlm" ? k : std::string("vla");  // old VLA reports say kind "JobReport"
    const auto recs = rep.find("recipes");
    if (recs == rep.end() || !recs->is_object()) continue;
    for (const auto& [name, rc] : recs->items()) {
      if (!rc.is_object() || !rc.contains("summary")) continue;
      auto m = measured_from_summary(kind, rc["summary"]);
      if (!m) continue;
      m->job_id = d.filename().string();  // the id /jobs/<id> serves, whatever the report itself says
      m->algorithm = rc.value("algorithm", "");
      out[name] = *m;
    }
  }
  return out;
}

bool valid_objective(const std::string& o) { return o == "accuracy" || o == "balanced" || o == "throughput"; }

void rank(std::vector<Json>& recs, const std::string& objective) {
  auto risk = [](const Json& r) {
    const auto st = r.value("risk", std::string("medium"));
    return st == "low" ? 0 : st == "medium" ? 1 : 2;
  };
  auto feasible = [](const Json& r) { return r.value("feasible", false); };
  auto native = [](const Json& r) { return r.value("deploy_verdict", "") == "native"; };
  auto conc = [](const Json& r) { return r.value("max_concurrency", int64_t{0}); };
  // balanced: options within 10% of the best usable concurrency count as equally fast -- memory is not what
  // limits them, so the more accurate one wins there; below that band, concurrency decides as usual. The band
  // is fixed before sorting, so the order stays a strict weak ordering.
  int64_t best = 0;
  for (const auto& r : recs)
    if (feasible(r) && risk(r) != 2) best = std::max(best, conc(r));
  auto in_band = [&](const Json& r) { return best > 0 && conc(r) * 10 >= best * 9; };
  std::stable_sort(recs.begin(), recs.end(), [&](const Json& a, const Json& b) {
    if (feasible(a) != feasible(b)) return feasible(a);
    if (objective == "accuracy") {
      if (risk(a) != risk(b)) return risk(a) < risk(b);
      if (native(a) != native(b)) return native(a);
      return conc(a) > conc(b);
    }
    if (objective == "balanced") {  // a high-risk recipe only wins when nothing else is usable
      if ((risk(a) == 2) != (risk(b) == 2)) return risk(b) == 2;
      if (native(a) != native(b)) return native(a);
      if (in_band(a) != in_band(b)) return in_band(a);
      if (in_band(a) && risk(a) != risk(b)) return risk(a) < risk(b);
      if (conc(a) != conc(b)) return conc(a) > conc(b);
      return risk(a) < risk(b);
    }
    if (native(a) != native(b)) return native(a);  // throughput: planner::recommend's own order
    return conc(a) > conc(b);
  });
}

}  // namespace mqde::server
