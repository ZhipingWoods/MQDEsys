#pragma once
// Claim-boundary estimation (docs/REFERENCES.md A.10): given a model + system + min_tps,
// produce conservative and optimistic throughput bounds and the largest context that still holds.

#include <cstdint>
#include <string>
#include <string_view>

#include "mqde/estimator/estimator.h"
#include "mqde/fit/fit.h"
#include "mqde/modelspec/model_spec.h"

namespace mqde {
namespace fit {

struct ClaimBounds {
  double conservative_tps = 0;   // worst-case (lower bw_efficiency)
  double optimistic_tps = 0;     // best-case (higher bw_efficiency)
  bool holds = false;            // conservative_tps >= min_tps
  std::string best_quant;        // quantization used for this estimate
  // Largest power-of-two context at which conservative_tps still >= min_tps. 0 if holds=false.
  int64_t max_context_at_min_tps = 0;
  double memory_gb = 0;
  double efficiency_used = 0;    // the efficiency factor used for the conservative estimate
};

// Compute claim bounds for a model on a system.
// min_tps: minimum acceptable throughput (tok/s or steps/s).
// Picks the best-fitting quant from the hierarchy; conservative uses efficiency * 0.80,
// optimistic uses efficiency * 1.15, both clamped to (0, 1].
ClaimBounds fit_bounds(const modelspec::ModelSpec& spec,
                       const System& sys,
                       const FitConfig& cfg,
                       const estimator::BackendProfile& backend,
                       double min_tps);

nlohmann::json to_json(const ClaimBounds& b);

}  // namespace fit
}  // namespace mqde
