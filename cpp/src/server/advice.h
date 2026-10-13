#pragma once
// Deployment advice (the new-job wizard's "部署建议"): planner::recommend ranks recipe x backend x hardware by
// what fits and how many sequences it serves, which on its own puts the most aggressive recipe (W2) first.
// This adds the accuracy side -- measured on this model by finished jobs when there are any, else a prior from
// the bit widths, labelled as such -- and orders by the user's objective.
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "mqde/common/config.h"
#include "mqde/estimator/estimator.h"

namespace mqde::server {

enum class Risk { kLow = 0, kMedium = 1, kHigh = 2 };
const char* risk_name(Risk r);

// Prior from the recipe alone: >= 8-bit everything is low, 4-bit weights medium, <= 3-bit weights high; 4-bit
// weights with 8-bit activations are medium too (measured worse than W4A16, but still usable on both domains).
Risk prior_risk(const estimator::Recipe& r);

struct Measured {
  std::string job_id, kind, algorithm, metric;
  double value = 0;
  Risk risk = Risk::kMedium;
};

// A finished report's summary for one recipe -> the accuracy number that matters for its kind, or nothing when
// that recipe was not evaluated (quantize-only). llm: PPL change %; vlm: answer top-1 agreement; vla: action
// error over the flow-matching noise floor.
std::optional<Measured> measured_from_summary(const std::string& kind, const Json& summary);

// A report of a job that ran to the end: verdict "succeeded" (vlm/vla jobs), or -- llm_job writes no verdict --
// no error and a succeeded "report" stage.
bool finished_ok(const Json& report);

// Newest measurement per recipe name (report key, e.g. "W4A16") for `model_path`, from runs/*/report.json.
std::map<std::string, Measured> measured_for(const std::filesystem::path& runs_dir, const std::string& model_path);

// objective: "accuracy" | "balanced" | "throughput". Every entry needs feasible, deploy_verdict, max_concurrency,
// risk (as written by annotate). Infeasible always last.
void rank(std::vector<Json>& recs, const std::string& objective);
bool valid_objective(const std::string& objective);

}  // namespace mqde::server
