#include "mqde/storage/calibration.h"

#include <chrono>
#include <cmath>

namespace mqde::storage {

namespace {
double now_s() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}
std::vector<Value> key_params(const CalibrationKey& k) { return {k.hardware, k.backend, k.recipe}; }
}  // namespace

std::string confidence_for_samples(int samples) {
  return samples >= 3 ? "high" : samples >= 1 ? "medium" : "low";
}

Result<Calibration> CalibrationStore::get(const CalibrationKey& k, const Calibration& defaults) {
  auto r = db_.query(
      "SELECT eta_bw, mfu, overhead_bytes, samples FROM calibration WHERE hardware=? AND backend=? AND "
      "recipe=?",
      key_params(k));
  if (!r) return std::unexpected(r.error());
  Calibration c = defaults;
  if (r->empty()) {
    c.samples = 0;
  } else {
    c.eta_bw = as_real((*r)[0][0]);
    c.mfu = as_real((*r)[0][1]);
    c.overhead_bytes = as_real((*r)[0][2]);
    c.samples = static_cast<int>(as_int((*r)[0][3]));
  }
  c.confidence = confidence_for_samples(c.samples);
  return c;
}

Result<Calibration> CalibrationStore::observe(const CalibrationKey& k, const Calibration& defaults,
                                              std::optional<double> eta_bw, std::optional<double> mfu,
                                              std::optional<double> overhead_bytes) {
  if (!eta_bw && !mfu && !overhead_bytes)
    return make_error(Errc::kInvalidArgument, "an observation needs at least one coefficient");
  auto bad_frac = [](const std::optional<double>& v) {
    return v && !(std::isfinite(*v) && *v > 0 && *v <= 1.5);
  };
  if (bad_frac(eta_bw))
    return make_error(Errc::kInvalidArgument, "eta_bw must be in (0, 1.5], got {}", *eta_bw);
  if (bad_frac(mfu)) return make_error(Errc::kInvalidArgument, "mfu must be in (0, 1.5], got {}", *mfu);
  if (overhead_bytes && !(std::isfinite(*overhead_bytes) && *overhead_bytes >= 0))
    return make_error(Errc::kInvalidArgument, "overhead_bytes must be >= 0, got {}", *overhead_bytes);

  Calibration out;
  auto tx = db_.transaction([&](Db& db) -> Result<void> {  // read-modify-write under one write lock: two
                                                           // feedback jobs cannot lose an update
    CalibrationStore inner(db, alpha_);
    auto cur = inner.get(k, defaults);
    if (!cur) return std::unexpected(cur.error());
    out = *cur;
    if (eta_bw) out.eta_bw = (1 - alpha_) * out.eta_bw + alpha_ * *eta_bw;
    if (mfu) out.mfu = (1 - alpha_) * out.mfu + alpha_ * *mfu;
    if (overhead_bytes) out.overhead_bytes = (1 - alpha_) * out.overhead_bytes + alpha_ * *overhead_bytes;
    out.samples = cur->samples + 1;
    out.confidence = confidence_for_samples(out.samples);
    auto w = db.execute(
        "INSERT INTO calibration(hardware, backend, recipe, eta_bw, mfu, overhead_bytes, samples, "
        "updated_at) VALUES(?,?,?,?,?,?,?,?) "
        "ON CONFLICT(hardware, backend, recipe) DO UPDATE SET eta_bw=excluded.eta_bw, mfu=excluded.mfu, "
        "overhead_bytes=excluded.overhead_bytes, "
        "samples=excluded.samples, updated_at=excluded.updated_at",
        {k.hardware, k.backend, k.recipe, out.eta_bw, out.mfu, out.overhead_bytes, int64_t{out.samples},
         now_s()});
    if (!w) return std::unexpected(w.error());
    return {};
  });
  if (!tx) return std::unexpected(tx.error());
  return out;
}

Result<void> CalibrationStore::reset(const CalibrationKey& k) {
  auto r = db_.execute("DELETE FROM calibration WHERE hardware=? AND backend=? AND recipe=?", key_params(k));
  if (!r) return std::unexpected(r.error());
  return {};
}

}  // namespace mqde::storage
