#pragma once
// Calibration store (T-0209): per (hardware, backend, recipe) coefficients of the performance model, learned
// from measurements.
//
// Starting point = the backend profile's own coefficients (passed in as `defaults`, so this module does not
// depend on the estimator). Every observation moves a coefficient by an exponential moving average
//     x_new = (1 - alpha) * x_old + alpha * measured          (alpha = 0.3 by default)
// and increments the sample count. Confidence: 0 samples = "low", 1 or 2 = "medium", 3 or more = "high".

#include <optional>
#include <string>

#include "mqde/common/result.h"
#include "mqde/storage/db.h"

namespace mqde::storage {

struct Calibration {
  double eta_bw = 0.6;
  double mfu = 0.35;
  double overhead_bytes = 0;
  int samples = 0;
  std::string confidence = "low";
};

struct CalibrationKey {
  std::string hardware, backend, recipe;
};

// "low" / "medium" / "high" from the number of samples.
std::string confidence_for_samples(int samples);

class CalibrationStore {
 public:
  explicit CalibrationStore(Db& db, double alpha = 0.3) : db_(db), alpha_(alpha) {}

  // The stored coefficients, or `defaults` with samples = 0 when nothing was measured for this key.
  Result<Calibration> get(const CalibrationKey& k, const Calibration& defaults);
  // Applies one observation (any subset of the three coefficients) and returns the new state. Values must be
  // finite and in range: eta_bw and mfu in (0, 1.5], overhead_bytes >= 0. A rejected observation changes
  // nothing.
  Result<Calibration> observe(const CalibrationKey& k, const Calibration& defaults,
                              std::optional<double> eta_bw, std::optional<double> mfu,
                              std::optional<double> overhead_bytes);
  Result<void> reset(const CalibrationKey& k);

 private:
  Db& db_;
  double alpha_;
};

}  // namespace mqde::storage
