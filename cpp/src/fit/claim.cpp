#include "mqde/fit/claim.h"

#include <algorithm>
#include <cmath>

namespace mqde::fit {

ClaimBounds fit_bounds(const modelspec::ModelSpec& spec, const System& sys, const FitConfig& cfg,
                       const estimator::BackendProfile& backend, double min_tps, const Request& base) {
  // Run analyze() to get the chosen quant and base formula_tps.
  const Request& req = base;
  auto f = analyze(spec, sys, cfg, backend, req);
  if (!f || f->fit_level == FitLevel::kTooTight || f->formula_tps <= 0) return {};

  // Base estimate came from cfg.bw_efficiency. Derive conservative/optimistic variants.
  const double base_eff = cfg.bw_efficiency;
  const double cons_eff = std::clamp(base_eff * 0.80, 0.01, 1.0);
  const double opt_eff = std::clamp(base_eff * 1.15, 0.01, 1.0);

  // Scale formula_tps linearly by efficiency ratio (bandwidth-roofline formula is linear in eff).
  const double base_tps = f->formula_tps > 0 ? f->formula_tps : f->estimated_tps;
  double cons_tps = base_tps * (cons_eff / base_eff);
  double opt_tps = base_tps * (opt_eff / base_eff);
  // A measurement beats any efficiency assumption.
  const double measured = f->measured_tps ? f->measured_tps->tok_s : 0;
  if (measured > 0) cons_tps = measured * 0.9, opt_tps = measured * 1.1;

  ClaimBounds b;
  b.conservative_tps = cons_tps;
  b.optimistic_tps = opt_tps;
  b.holds = cons_tps >= min_tps;
  b.best_quant = f->best_quant;
  b.memory_gb = f->memory_required_gb;
  b.efficiency_used = cons_eff;
  b.speed_unit = f->speed_unit;
  b.basis = measured > 0 ? "measured" : "formula";

  // Binary search for max_context_at_min_tps: find largest power-of-two context
  // at which the conservative estimate still >= min_tps.
  // Since context only affects KV cache (not decode speed directly for bandwidth formula),
  // the decode tps stays constant w.r.t. context for the roofline formula.
  // However more context means more memory -> may force a worse quant.
  // We solve this by re-running analyze() at each candidate context.
  if (b.holds) {
    const int64_t max_pos =
        spec.text.max_position_embeddings > 0 ? spec.text.max_position_embeddings : (1 << 20);
    b.max_context_at_min_tps = 0;
    for (int64_t p = 1; p <= max_pos; p *= 2) {
      Request rctx = req;
      rctx.context = p;
      rctx.measured.reset();
      rctx.bench = nullptr;
      auto fc = analyze(spec, sys, cfg, backend, rctx);
      if (!fc || fc->fit_level == FitLevel::kTooTight) break;
      // Scale the measured value by how the formula moves with context; otherwise use the formula.
      const double ctps = measured > 0 ? cons_tps * fc->formula_tps / std::max(1e-9, base_tps)
                                       : fc->formula_tps * (cons_eff / base_eff);
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
          {"efficiency_used", b.efficiency_used},
          {"speed_unit", b.speed_unit},
          {"basis", b.basis}};
}

}  // namespace mqde::fit
