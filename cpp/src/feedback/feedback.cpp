#include "mqde/feedback/feedback.h"

namespace mqde::feedback {

Result<CalibrateResult> calibrate(storage::CalibrationStore& store, const storage::CalibrationKey& key,
                                  const storage::Calibration& defaults, const estimator::Estimate& e,
                                  const estimator::HardwareProfile& hw, const estimator::BackendProfile& be,
                                  const estimator::PerfParams& p, const Observation& obs) {
  CalibrateResult out;
  std::optional<double> eta, mfu, overhead;

  if (obs.tpot_ms) {
    if (auto r = estimator::infer_eta_bw(e, hw, be, p, *obs.tpot_ms)) {
      eta = *r;
      out.applied.emplace_back("eta_bw");
    } else {
      out.skipped.push_back("eta_bw: " + r.error().message);
    }
  }
  if (obs.ttft_ms) {
    if (auto r = estimator::infer_mfu(e, hw, be, p, *obs.ttft_ms)) {
      mfu = *r;
      out.applied.emplace_back("mfu");
    } else {
      out.skipped.push_back("mfu: " + r.error().message);
    }
  }
  if (obs.peak_bytes) {
    if (auto r = estimator::infer_overhead_bytes(e, be, *obs.peak_bytes)) {
      overhead = *r;
      out.applied.emplace_back("overhead_bytes");
    } else {
      out.skipped.push_back("overhead_bytes: " + r.error().message);
    }
  }

  if (!eta && !mfu && !overhead) {
    auto cur = store.get(key, defaults);
    if (!cur) return std::unexpected(cur.error());
    out.calibration = *cur;
    return out;
  }
  auto updated = store.observe(key, defaults, eta, mfu, overhead);
  if (!updated) return std::unexpected(updated.error());
  out.calibration = *updated;
  return out;
}

estimator::BackendProfile backend_with_calibration(estimator::BackendProfile be,
                                                   const storage::Calibration& cal) {
  return estimator::with_coefficients(std::move(be), cal.eta_bw, cal.mfu, cal.overhead_bytes);
}

}  // namespace mqde::feedback
