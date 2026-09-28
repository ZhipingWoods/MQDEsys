#pragma once
// Concurrency solve-back: given a model + recipe + system, how many parallel users fit?
// Implements the memory model from docs/REFERENCES.md A.9 and llmfit analysis.rs concurrency helpers.

#include <cstdint>
#include <string>
#include <string_view>

#include "mqde/common/result.h"
#include "mqde/estimator/estimator.h"
#include "mqde/modelspec/model_spec.h"

namespace mqde {
namespace fit {

// Memory breakdown for one (users, context) configuration.
struct ConcurrencyEstimate {
  int64_t requested_users = 0;
  int64_t max_users = 0;       // largest N such that the model still fits at the given context
  int64_t context = 0;         // context tokens used for this estimate
  double weights_gb = 0;       // static weight bytes (incl. quant metadata)
  double kv_per_user_gb = 0;   // KV cache + recurrent state per concurrent user, at this context
  double overhead_gb = 0;      // runtime overhead (allocator headroom + activations)
  double total_gb = 0;         // weights + kv_per_user * requested_users + overhead
  double capacity_gb = 0;      // vram_gb * gpu_memory_utilization * gpu_count
  double utilization_pct = 0;  // total / capacity * 100
  bool fits = false;           // total <= capacity
  bool high_pressure = false;  // utilization_pct > 90

  // Largest power-of-two context at which requested_users all fit. 0 when even context=1 doesn't fit.
  int64_t max_context_for_users = 0;
};

// Forward-solve: compute memory breakdown for (users, context).
// Uses the same memory model as estimator::estimate() so totals are consistent.
// The recipe is identified by name (e.g. "bf16", "w4a16"); loaded from data/recipes/.
ConcurrencyEstimate estimate_concurrency(const modelspec::ModelSpec& spec,
                                         std::string_view recipe_name,
                                         const estimator::HardwareProfile& hw,
                                         const estimator::BackendProfile& backend,
                                         int64_t users,
                                         int64_t context_tokens);

// Expose to_json for the API endpoint.
nlohmann::json to_json(const ConcurrencyEstimate& c);

}  // namespace fit
}  // namespace mqde
