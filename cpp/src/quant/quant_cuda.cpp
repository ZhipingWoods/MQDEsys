#include "quant_cuda.h"

#include "quant_cuda_impl.h"

namespace mqde::quant::cuda {
namespace {
double g_last_ms = 0;
}

bool available() { return mqde_cuda_device_count() > 0; }
double last_kernel_ms() { return g_last_ms; }

Result<void> fake_quant(const float* x, int64_t rows, int64_t cols, const QuantSpec& s, float* out,
                        QParams* qparams_out) {
  if (s.fmt != NumFormat::kInt)
    return make_error(Errc::kUnsupported, "the CUDA quantizer handles integer formats only");
  if (!s.enabled()) return make_error(Errc::kInvalidArgument, "quantization spec is disabled");
  if (rows <= 0 || cols <= 0) return make_error(Errc::kInvalidArgument, "empty tensor [{}, {}]", rows, cols);
  if (s.gran == Granularity::kPerGroup && cols % s.group_size != 0)
    return make_error(Errc::kInvalidArgument, "in_features ({}) is not divisible by group_size ({})", cols,
                      s.group_size);
  if (!available()) return make_error(Errc::kUnsupported, "no CUDA device");

  QParams qp;
  qp.per_tensor = s.gran == Granularity::kPerTensor;
  qp.group_cols = s.gran == Granularity::kPerGroup ? s.group_size : cols;
  qp.groups_per_row = qp.per_tensor ? 1 : cols / qp.group_cols;
  const size_t groups = qp.per_tensor ? 1 : static_cast<size_t>(rows * qp.groups_per_row);
  qp.scale.assign(groups, 1.0f);
  const bool asym = !s.symmetric;
  if (asym) qp.zero.assign(groups, 0.0f);

  MqdeCudaQuantArgs a{rows,
                      cols,
                      qp.group_cols,
                      qp.per_tensor,
                      s.symmetric,
                      s.calib == CalibAlgo::kMse,
                      s.grid,
                      s.maxshrink,
                      static_cast<float>(s.qmin()),
                      static_cast<float>(s.qmax())};
  char err[256] = {};
  if (mqde_cuda_fake_quant(x, out, qp.scale.data(), asym ? qp.zero.data() : nullptr, a, &g_last_ms, err,
                           sizeof err) != 0)
    return make_error(Errc::kInternal, "CUDA quantizer: {}", err);
  if (qparams_out) *qparams_out = std::move(qp);
  return {};
}

}  // namespace mqde::quant::cuda
