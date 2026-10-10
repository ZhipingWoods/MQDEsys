#pragma once
// Feedback loop (T-0905, pipeline stage S15): turn one real measurement of a deployed job into an update of
// storage::CalibrationStore, so the next estimate for the same (hardware, backend, recipe) is closer to
// reality.
//
// This module exists because estimator::infer_eta_bw / infer_mfu / infer_overhead_bytes (which coefficient a
// measurement can support) and storage::CalibrationStore (where coefficients persist) are in modules that
// must not depend on each other (estimator is pure computation; storage only knows about common). Something
// has to sit above both -- this is that something.

#include <optional>

#include "mqde/common/result.h"
#include "mqde/estimator/estimator.h"
#include "mqde/storage/calibration.h"

namespace mqde::feedback {

// What was actually measured after deploying and load-testing one job. Any subset may be known; a 0 / unset
// value is skipped rather than treated as a real zero.
struct Observation {
  std::optional<double> peak_bytes;
  std::optional<double> ttft_ms;
  std::optional<double> tpot_ms;
};

struct CalibrateResult {
  storage::Calibration calibration;  // the store's new state (or the defaults, if nothing could be inferred)
  std::vector<std::string> applied;  // which coefficients were updated, e.g. {"eta_bw", "overhead_bytes"}
  std::vector<std::string>
      skipped;  // which were not, and why (e.g. "mfu: prefill was memory-bound, not compute-bound")
};

// Infers whatever `obs` supports against the model implied by (spec, recipe, hw, be, p), applies each
// inferred value to `store` at `key` (EMA, same as storage::CalibrationStore::observe), and reports what
// happened. An observation that supports nothing (e.g. every phase bound by the "wrong" roof, or `obs`
// entirely empty) changes nothing in the store and returns the current (hardware, backend, recipe)
// calibration with `applied` empty, not an error -- a single inconclusive measurement is a normal outcome of
// real hardware, not a bug.
Result<CalibrateResult> calibrate(storage::CalibrationStore& store, const storage::CalibrationKey& key,
                                  const storage::Calibration& defaults, const estimator::Estimate& e,
                                  const estimator::HardwareProfile& hw, const estimator::BackendProfile& be,
                                  const estimator::PerfParams& p, const Observation& obs);

// `be` with its eta_bw / mfu / runtime_overhead_bytes replaced by `cal`'s (storage::CalibrationStore's own
// estimator::with_coefficients, so callers do not need to depend on both modules just for this one line).
estimator::BackendProfile backend_with_calibration(estimator::BackendProfile be,
                                                   const storage::Calibration& cal);

}  // namespace mqde::feedback
