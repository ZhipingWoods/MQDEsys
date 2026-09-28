#include "mqde/fit/claim.h"

#include <algorithm>
#include <cmath>

namespace mqde::fit {

ClaimBounds fit_bounds(const modelspec::ModelSpec& spec,
                       const System& sys,
                       const FitConfig& cfg,
                       const estimator::BackendProfile& backend,
                       double min_tps) {
  // Run analyze() to get the chosen quant and base formula_tps.
  Request req;
  auto f = analyze(spec, sys, cfg, backend, req);
  if (!f || f->fit_level == FitLevel::kTooTight || f->formula_tps <= 0)
    return {};

  // Base estimate came from cfg.bw_efficiency. Derive conservative/optimistic variants.
  const double base_eff = cfg.bw_efficiency;
  const double cons_eff = std::clamp(base_eff * 0.80, 0.01, 1.0);
  const double opt_eff  = std::clamp(base_eff * 1.15, 0.01, 1.0);

  // Scale formula_tps linearly by efficiency ratio (bandwidth-roofline formula is linear in eff).
  const double base_tps = f->formula_tps > 0 ? f->formula_tps : f->estimated_tps;
  const double cons_tps = base_tps * (cons_eff / base_eff);
  const double opt_tps  = base_tps * (opt_eff  / base_eff);

  ClaimBounds b;
  b.conservative_tps = cons_tps;
  b.optimistic_tps   = opt_tps;
  b.holds            = cons_tps >= min_tps;
  b.best_quant       = f->best_quant;
  b.memory_gb        = f->memory_required_gb;
  b.efficiency_used  = cons_eff;

  // Binary search for max_context_at_min_tps: find largest power-of-two context
  // at which the conservative estimate still >= min_tps.
  // Since context only affects KV cache (not decode speed directly for bandwidth formula),
  // the decode tps stays constant w.r.t. context for the roofline formula.
  // However more context means more memory -> may force a worse quant.
  // We solve this by re-running analyze() at each candidate context.
  if (b.holds) {
    const int64_t max_pos = spec.text.max_position_embeddings > 0 ? spec.text.max_position_embeddings : (1 << 20);
    b.max_context_at_min_tps = 0;
    for (int64_t p = 1; p <= max_pos; p *= 2) {
      Request rctx;
      rctx.context = p;
      auto fc = analyze(spec, sys, cfg, backend, rctx);
      if (!fc || fc->fit_level == FitLevel::kTooTight) break;
      const double ctps = fc->formula_tps * (cons_eff / base_eff);
      if (ctps < min_tps) break;
      b.max_context_at_min_tps = p;
    }
  }

  return b;
}

nlohmann::json to_json(const ClaimBounds& b) {
  return {{"conservative_tps", b.conservative_tps},
          {"optimistic_tps", b.optimistic_tps},
          {"holds", b.holds},
          {"best_quant", b.best_quant},
          {"max_context_at_min_tps", b.max_context_at_min_tps},
          {"memory_gb", b.memory_gb},
          {"efficiency_used", b.efficiency_used}};
}

}  // namespace mqde::fit
