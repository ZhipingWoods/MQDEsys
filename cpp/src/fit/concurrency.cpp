#include "mqde/fit/concurrency.h"

#include <algorithm>
#include <cmath>

#include "mqde/estimator/estimator.h"

namespace mqde::fit {

// Memory model (llmfit A.9): total = weights + users * kv_per_user + overhead.
// kv_per_user = (kv_bytes_per_token * context + state_bytes_per_seq).
// We derive these from a single estimate() call at concurrency=1 to avoid
// re-implementing the KV formula.
static ConcurrencyEstimate compute(const modelspec::ModelSpec& spec, const estimator::Recipe& recipe,
                                   const estimator::HardwareProfile& hw,
                                   const estimator::BackendProfile& backend, int64_t users,
                                   int64_t context_tokens) {
  const int64_t ctx = context_tokens > 0 ? context_tokens : 4096;

  // Single-user estimate to extract the per-user KV cost at this context length.
  estimator::RuntimeParams rt1;
  rt1.concurrency = 1;
  rt1.context_tokens = ctx;
  const estimator::Estimate e1 = estimator::estimate(spec, recipe, hw, backend, rt1);

  const double weights_gb = static_cast<double>(e1.weights_bytes) / (1ull << 30);
  // kv_bytes includes (kv_per_token * context + recurrent_state) * concurrency.
  // Dividing by concurrency (=1) gives per-user cost.
  const double kv_per_user_gb = static_cast<double>(e1.kv_bytes) / (1ull << 30);
  // overhead = activations + runtime (constant, not per-user).
  const double overhead_gb = static_cast<double>(e1.activation_bytes + e1.runtime_bytes) / (1ull << 30);

  const double capacity_gb = static_cast<double>(hw.mem_bytes) / (1ull << 30) * rt1.gpu_memory_utilization;
  const int64_t n_gpus = std::max<int64_t>(1, 1);  // single GPU for now
  const double cap = capacity_gb * n_gpus;

  const double total_gb = weights_gb + kv_per_user_gb * users + overhead_gb;
  const double util_pct = cap > 0 ? total_gb / cap * 100.0 : 100.0;

  // Solve max_users: floor((capacity - weights - overhead) / kv_per_user).
  int64_t max_users = 0;
  if (kv_per_user_gb > 0) {
    const double avail = cap - weights_gb - overhead_gb;
    max_users = avail > 0 ? static_cast<int64_t>(std::floor(avail / kv_per_user_gb)) : 0;
  }
  max_users = std::max<int64_t>(0, max_users);

  // Solve max_context_for_users: binary search in powers of two.
  // Start at the requested context and walk down until it fits; walk up while it still fits.
  int64_t max_ctx_for = 0;
  if (weights_gb + overhead_gb < cap && users > 0 && e1.kv_bytes_per_token > 0) {
    // Solve max_context_for_users: find the largest power-of-two context at which a fresh
    // estimate() still fits for `users` concurrent sequences. We do a forward-validated binary
    // search over powers of two so the result is always consistent with estimate().
    int64_t max_pos = spec.text.max_position_embeddings > 0 ? spec.text.max_position_embeddings : (1 << 20);
    // Collect candidate power-of-two context lengths up to max_pos.
    std::vector<int64_t> candidates;
    for (int64_t p = 1; p <= max_pos; p *= 2) candidates.push_back(p);
    // Walk from largest down; take the first one that fits.
    for (int64_t i = static_cast<int64_t>(candidates.size()) - 1; i >= 0; --i) {
      const int64_t cand_ctx = candidates[static_cast<size_t>(i)];
      estimator::RuntimeParams rt_cand;
      rt_cand.concurrency = users;
      rt_cand.context_tokens = cand_ctx;
      const estimator::Estimate ec = estimator::estimate(spec, recipe, hw, backend, rt_cand);
      if (ec.peak_bytes <= ec.capacity_bytes) {
        max_ctx_for = cand_ctx;
        break;
      }
    }
  }

  ConcurrencyEstimate out;
  out.requested_users = users;
  out.max_users = max_users;
  out.context = ctx;
  out.weights_gb = weights_gb;
  out.kv_per_user_gb = kv_per_user_gb;
  out.overhead_gb = overhead_gb;
  out.total_gb = total_gb;
  out.capacity_gb = cap;
  out.utilization_pct = util_pct;
  out.fits = total_gb <= cap + 1e-9;
  out.high_pressure = util_pct > 90.0;
  out.max_context_for_users = max_ctx_for;
  return out;
}

ConcurrencyEstimate estimate_concurrency(const modelspec::ModelSpec& spec, std::string_view recipe_name,
                                         const estimator::HardwareProfile& hw,
                                         const estimator::BackendProfile& backend, int64_t users,
                                         int64_t context_tokens) {
  auto recipe = estimator::load_recipe(recipe_name);
  if (!recipe) {
    ConcurrencyEstimate err;
    err.requested_users = users;
    return err;
  }
  return compute(spec, *recipe, hw, backend, users, context_tokens);
}

Json to_json(const ConcurrencyEstimate& c) {
  return {{"requested_users", c.requested_users},
          {"max_users", c.max_users},
          {"context", c.context},
          {"weights_gb", c.weights_gb},
          {"kv_per_user_gb", c.kv_per_user_gb},
          {"overhead_gb", c.overhead_gb},
          {"total_gb", c.total_gb},
          {"capacity_gb", c.capacity_gb},
          {"utilization_pct", c.utilization_pct},
          {"fits", c.fits},
          {"high_pressure", c.high_pressure},
          {"max_context_for_users", c.max_context_for_users}};
}

}  // namespace mqde::fit
