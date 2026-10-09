#pragma once
// Compatibility matrix and JobSpec validation (T-0210): everything that can be checked statically, before
// quantizing or deploying anything. Rules come from data/compat/matrix.yaml (L2) plus the generic
// native-format check already used by the estimator; see that file's header comment for where each rule was
// verified.

#include <optional>
#include <string>
#include <vector>

#include "mqde/common/config.h"
#include "mqde/common/result.h"
#include "mqde/estimator/estimator.h"
#include "mqde/quant/quant_spec.h"

namespace mqde::planner {

struct AlgorithmInfo {
  std::string name, label, kind;  // kind: quantizer | transform | combo
  std::string description;
  bool needs_calibration = false;
};

Result<std::vector<AlgorithmInfo>> list_algorithms();
Result<AlgorithmInfo> load_algorithm(const std::string& name);
// "awq+gptq" -> {"awq", "gptq"}; "none"/"" -> {"rtn"}. Mirrors python/mqde/quant/algorithms.py:resolve()
// exactly (at most one transform (awq|smoothquant) and one quantizer (rtn|gptq)), so the two never disagree.
Result<std::vector<std::string>> resolve_algorithm(const std::string& name);

enum class Verdict { kNative, kEmulated, kUnsupported };
std::string_view verdict_name(Verdict v);

struct CompatResult {
  Verdict verdict = Verdict::kNative;
  std::string reason;  // empty when kNative
  std::string kernel;  // backend kernel name when known, else empty
  bool ok() const { return verdict != Verdict::kUnsupported; }
};

// Optional concrete layer shape for the packing / group-size checks that need real dimensions; omit for a
// recipe-level (shape-independent) check.
struct LayerShape {
  int64_t out_features = 0, in_features = 0;
};

// Whether `weight`/`act` can be written as export format `format` ("gptq" | "awq" | "compressed-tensors"; the
// fp8 vs int compressed-tensors scheme is picked from `weight.fmt` automatically). Unknown format ->
// kUnsupported.
CompatResult check_export(const std::string& format, const quant::QuantSpec& weight,
                          const quant::QuantSpec& act, std::optional<LayerShape> shape = std::nullopt);

// Whether `backend` can serve an export of `format` for this recipe on this hardware: combines matrix.yaml's
// known per-backend issues with the generic "is the resulting format in hw.native_formats" rule.
CompatResult check_deploy(const std::string& format, const estimator::Recipe& recipe,
                          const std::string& backend, const estimator::HardwareProfile& hw);

struct Problem {
  std::string field;  // JSON pointer into the job, e.g. "/quant/algorithm"
  std::string message;
  std::string suggestion;
};

// Every static problem found in `job` (the same JSON a job submission POSTs), aggregated rather than stopping
// at the first one. An empty result means every static check this module knows passed; it does not mean the
// job will succeed (model-specific and runtime failures are still possible).
std::vector<Problem> validate_jobspec(const Json& job);

}  // namespace mqde::planner
