#include "mqde/quant/ops.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>

namespace mqde::quant {
namespace {

// All rounding is IEEE round-half-to-even (the default FE_TONEAREST mode), matching torch.round.
inline float rne(float v) { return std::nearbyint(v); }

struct GroupParams {
  float scale;
  float zero;
};

// Range -> (scale, zero). Order of float32 operations is part of the numerical contract
// (python/mqde/quant/numerics.py mirrors it exactly).
GroupParams params_from_range(float mn, float mx, const QuantSpec& s) {
  if (s.fmt == NumFormat::kInt) {
    const float qmin = static_cast<float>(s.qmin());
    const float qmax = static_cast<float>(s.qmax());
    if (s.symmetric) {
      const float amax = std::max(std::fabs(mn), std::fabs(mx));
      float scale = amax / qmax;
      if (!(scale > 0.0f)) scale = 1.0f;
      return {scale, 0.0f};
    }
    mn = std::min(mn, 0.0f);
    mx = std::max(mx, 0.0f);
    float scale = (mx - mn) / (qmax - qmin);
    if (!(scale > 0.0f)) scale = 1.0f;
    float zero = rne(qmin - mn / scale);
    zero = std::clamp(zero, qmin, qmax);
    return {scale, zero};
  }
  const float amax = std::max(std::fabs(mn), std::fabs(mx));
  float scale = amax / s.fp8_max();
  if (!(scale > 0.0f)) scale = 1.0f;
  return {scale, 0.0f};
}

inline float qdq_one(float v, const GroupParams& p, const QuantSpec& s) {
  if (s.fmt == NumFormat::kInt) {
    const float q = std::clamp(rne(v / p.scale) + p.zero, static_cast<float>(s.qmin()), static_cast<float>(s.qmax()));
    return (q - p.zero) * p.scale;
  }
  const float fm = s.fp8_max();
  const float q = fp8_round(std::clamp(v / p.scale, -fm, fm), s.fmt);
  return q * p.scale;
}

struct GroupView {
  const float* x;
  int64_t n_rows;  // rows spanned by the group (per_tensor spans all rows)
  int64_t n_cols;  // columns per row in the group
  int64_t stride;  // row stride in the source matrix
};

template <class F>
inline void for_each_in_group(const GroupView& g, F&& f) {
  for (int64_t r = 0; r < g.n_rows; ++r) {
    const float* row = g.x + r * g.stride;
    for (int64_t c = 0; c < g.n_cols; ++c) f(row[c]);
  }
}

GroupParams calibrate_group(const GroupView& g, const QuantSpec& s) {
  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for_each_in_group(g, [&](float v) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  });
  if (!std::isfinite(mn) || !std::isfinite(mx)) return {1.0f, 0.0f};
  GroupParams best = params_from_range(mn, mx, s);
  if (s.calib != CalibAlgo::kMse) return best;

  // Grid search over shrunken ranges, minimizing squared reconstruction error.
  double best_err = std::numeric_limits<double>::infinity();
  for (int i = 0; i <= s.grid; ++i) {
    const float shrink = 1.0f - static_cast<float>(i) * (1.0f - s.maxshrink) / static_cast<float>(s.grid);
    const GroupParams p = params_from_range(mn * shrink, mx * shrink, s);
    double err = 0.0;
    for_each_in_group(g, [&](float v) {
      const double d = static_cast<double>(qdq_one(v, p, s)) - v;
      err += d * d;
    });
    if (err < best_err) {
      best_err = err;
      best = p;
    }
  }
  return best;
}

Result<void> check_shape(int64_t rows, int64_t cols, const QuantSpec& s) {
  if (!s.enabled()) return make_error(Errc::kInvalidArgument, "quantization spec is disabled");
  if (rows <= 0 || cols <= 0) return make_error(Errc::kInvalidArgument, "empty tensor [{}, {}]", rows, cols);
  if (s.gran == Granularity::kPerGroup && cols % s.group_size != 0)
    return make_error(Errc::kInvalidArgument, "in_features ({}) is not divisible by group_size ({})", cols,
                      s.group_size);
  return {};
}

template <class F>
void for_each_group(const float* x, int64_t rows, int64_t cols, const QuantSpec& s, const QParams& qp, F&& f) {
  if (qp.per_tensor) {
    f(0, GroupView{x, rows, cols, cols}, 0, 0);
    return;
  }
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t g = 0; g < qp.groups_per_row; ++g)
      f(r * qp.groups_per_row + g, GroupView{x + r * cols + g * qp.group_cols, 1, qp.group_cols, cols}, r,
        g * qp.group_cols);
}

QParams layout(int64_t rows, int64_t cols, const QuantSpec& s) {
  QParams qp;
  qp.per_tensor = s.gran == Granularity::kPerTensor;
  qp.group_cols = s.gran == Granularity::kPerGroup ? s.group_size : cols;
  qp.groups_per_row = qp.per_tensor ? 1 : cols / qp.group_cols;
  const size_t n = qp.per_tensor ? 1 : static_cast<size_t>(rows * qp.groups_per_row);
  qp.scale.assign(n, 1.0f);
  if (s.fmt == NumFormat::kInt && !s.symmetric) qp.zero.assign(n, 0.0f);
  return qp;
}

}  // namespace

float fp8_round(float v, NumFormat fmt) {
  const int mbits = fmt == NumFormat::kFp8E4M3 ? 3 : 2;
  const int emin = fmt == NumFormat::kFp8E4M3 ? -6 : -14;  // exponent of the smallest normal
  const float fmax = fmt == NumFormat::kFp8E4M3 ? 448.0f : 57344.0f;
  if (std::isnan(v)) return v;
  const float a = std::fabs(v);
  if (a == 0.0f) return v;
  int e = 0;
  std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)  =>  floor(log2 a) = e - 1
  const int exp = std::max(e - 1, emin);
  const float step = std::ldexp(1.0f, exp - mbits);
  float r = rne(a / step) * step;  // exact: step is a power of two
  r = std::min(r, fmax);
  return std::copysign(r, v);
}

Result<QParams> compute_qparams(const float* x, int64_t rows, int64_t cols, const QuantSpec& s) {
  if (auto ok = check_shape(rows, cols, s); !ok) return std::unexpected(ok.error());
  QParams qp = layout(rows, cols, s);
  for_each_group(x, rows, cols, s, qp, [&](int64_t gi, const GroupView& g, int64_t, int64_t) {
    const GroupParams p = calibrate_group(g, s);
    qp.scale[static_cast<size_t>(gi)] = p.scale;
    if (!qp.zero.empty()) qp.zero[static_cast<size_t>(gi)] = p.zero;
  });
  return qp;
}

Result<void> fake_quant(const float* x, int64_t rows, int64_t cols, const QuantSpec& s, float* out,
                        QParams* qparams_out) {
  auto qp = compute_qparams(x, rows, cols, s);
  if (!qp) return std::unexpected(qp.error());
  for_each_group(x, rows, cols, s, *qp, [&](int64_t gi, const GroupView& g, int64_t r0, int64_t c0) {
    const GroupParams p{qp->scale[static_cast<size_t>(gi)], qp->zero.empty() ? 0.0f : qp->zero[static_cast<size_t>(gi)]};
    for (int64_t r = 0; r < g.n_rows; ++r)
      for (int64_t c = 0; c < g.n_cols; ++c) {
        const int64_t idx = (r0 + r) * cols + c0 + c;
        out[idx] = qdq_one(x[idx], p, s);
      }
  });
  if (qparams_out) *qparams_out = std::move(*qp);
  return {};
}

Result<void> quantize_int(const float* x, int64_t rows, int64_t cols, const QuantSpec& s, const QParams& qp,
                          int16_t* codes) {
  if (s.fmt != NumFormat::kInt) return make_error(Errc::kUnsupported, "quantize_int requires an integer format");
  if (auto ok = check_shape(rows, cols, s); !ok) return std::unexpected(ok.error());
  const float qmin = static_cast<float>(s.qmin()), qmax = static_cast<float>(s.qmax());
  for_each_group(x, rows, cols, s, qp, [&](int64_t gi, const GroupView& g, int64_t r0, int64_t c0) {
    const float scale = qp.scale[static_cast<size_t>(gi)];
    const float zero = qp.zero.empty() ? 0.0f : qp.zero[static_cast<size_t>(gi)];
    for (int64_t r = 0; r < g.n_rows; ++r)
      for (int64_t c = 0; c < g.n_cols; ++c) {
        const int64_t idx = (r0 + r) * cols + c0 + c;
        codes[idx] = static_cast<int16_t>(std::clamp(rne(x[idx] / scale) + zero, qmin, qmax));
      }
  });
  return {};
}

Result<void> dequantize_int(const int16_t* codes, int64_t rows, int64_t cols, const QuantSpec& s, const QParams& qp,
                            float* out) {
  if (s.fmt != NumFormat::kInt) return make_error(Errc::kUnsupported, "dequantize_int requires an integer format");
  if (auto ok = check_shape(rows, cols, s); !ok) return std::unexpected(ok.error());
  // Group iteration only needs geometry; pass `out` as a dummy source pointer.
  for_each_group(out, rows, cols, s, qp, [&](int64_t gi, const GroupView& g, int64_t r0, int64_t c0) {
    const float scale = qp.scale[static_cast<size_t>(gi)];
    const float zero = qp.zero.empty() ? 0.0f : qp.zero[static_cast<size_t>(gi)];
    for (int64_t r = 0; r < g.n_rows; ++r)
      for (int64_t c = 0; c < g.n_cols; ++c) {
        const int64_t idx = (r0 + r) * cols + c0 + c;
        out[idx] = (static_cast<float>(codes[idx]) - zero) * scale;
      }
  });
  return {};
}

}  // namespace mqde::quant
