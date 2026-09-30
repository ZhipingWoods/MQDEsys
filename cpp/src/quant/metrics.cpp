#include "mqde/quant/metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mqde::quant {

ErrorMetrics measure(const float* ref, const float* approx, int64_t n) {
  ErrorMetrics m;
  if (n <= 0) return m;
  double sig = 0, noise = 0, dot = 0, aa = 0, max_abs = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double r = ref[i], a = approx[i], d = r - a;
    sig += r * r;
    noise += d * d;
    dot += r * a;
    aa += a * a;
    max_abs = std::max(max_abs, std::fabs(d));
  }
  m.mse = noise / static_cast<double>(n);
  m.rmse = std::sqrt(m.mse);
  m.max_abs = max_abs;
  m.sqnr_db = noise == 0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(sig / noise);
  m.cosine = (sig == 0 || aa == 0) ? (sig == aa ? 1.0 : 0.0) : dot / (std::sqrt(sig) * std::sqrt(aa));
  m.rel_l2 = sig == 0 ? (noise == 0 ? 0.0 : std::numeric_limits<double>::infinity()) : std::sqrt(noise / sig);
  return m;
}

TensorStats tensor_stats(const float* x, int64_t n) {
  TensorStats s;
  if (n <= 0) return s;
  double sum = 0;
  s.min = s.max = x[0];
  for (int64_t i = 0; i < n; ++i) {
    sum += x[i];
    s.min = std::min<double>(s.min, x[i]);
    s.max = std::max<double>(s.max, x[i]);
  }
  s.mean = sum / static_cast<double>(n);
  s.absmax = std::max(std::fabs(s.min), std::fabs(s.max));
  double m2 = 0, m4 = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = x[i] - s.mean, d2 = d * d;
    m2 += d2;
    m4 += d2 * d2;
  }
  m2 /= static_cast<double>(n);
  m4 /= static_cast<double>(n);
  s.std = std::sqrt(m2);
  s.kurtosis = m2 > 0 ? m4 / (m2 * m2) - 3.0 : 0.0;
  if (s.std > 0) {
    int64_t out = 0;
    const double thr = 6.0 * s.std;
    for (int64_t i = 0; i < n; ++i) out += std::fabs(x[i] - s.mean) > thr;
    s.outlier_ratio = static_cast<double>(out) / static_cast<double>(n);
  }
  return s;
}

namespace {
Json finite_or_null(double v) { return std::isfinite(v) ? Json(v) : Json(nullptr); }
}  // namespace

Json to_json(const ErrorMetrics& m) {
  return {{"mse", m.mse},         {"rmse", m.rmse},
          {"max_abs", m.max_abs}, {"sqnr_db", finite_or_null(m.sqnr_db)},
          {"cosine", m.cosine},   {"rel_l2", finite_or_null(m.rel_l2)}};
}

Json to_json(const TensorStats& s) {
  return {{"min", s.min},
          {"max", s.max},
          {"absmax", s.absmax},
          {"mean", s.mean},
          {"std", s.std},
          {"kurtosis", s.kurtosis},
          {"outlier_ratio", s.outlier_ratio}};
}

}  // namespace mqde::quant
