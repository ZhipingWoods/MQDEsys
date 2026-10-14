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
  const bool is_int = s.fmt == NumFormat::kInt;
  const bool asym = is_int && !s.symmetric;
  if (asym) qp.zero.assign(groups, 0.0f);

  MqdeCudaQuantArgs a{rows,
                      cols,
                      qp.group_cols,
                      qp.per_tensor,
                      s.symmetric,
                      s.calib == CalibAlgo::kMse,
                      s.grid,
                      s.maxshrink,
                      is_int ? static_cast<float>(s.qmin()) : 0.0f,
                      is_int ? static_cast<float>(s.qmax()) : 0.0f,
                      is_int ? 0 : (s.fmt == NumFormat::kFp8E4M3 ? 1 : 2),
                      is_int ? 0.0f : s.fp8_max()};
  char err[256] = {};
  if (mqde_cuda_fake_quant(x, out, qp.scale.data(), asym ? qp.zero.data() : nullptr, a, &g_last_ms, err,
                           sizeof err) != 0)
    return make_error(Errc::kInternal, "CUDA quantizer: {}", err);
  if (qparams_out) *qparams_out = std::move(qp);
  return {};
}

Result<std::vector<uint32_t>> pack_bits(std::span<const uint8_t> codes, int bits) {
  if (bits < 1 || bits > 8)
    return make_error(Errc::kInvalidArgument, "pack_bits: {} bits per code (1..8)", bits);
  if (codes.empty()) return std::vector<uint32_t>{};
  if (!available()) return make_error(Errc::kUnsupported, "no CUDA device");
  std::vector<uint32_t> words((codes.size() * static_cast<size_t>(bits) + 31) / 32);
  char err[256] = {};
  if (mqde_cuda_pack_bits(codes.data(), words.data(), static_cast<int64_t>(codes.size()), bits, &g_last_ms,
                          err, sizeof err) != 0)
    return make_error(Errc::kInternal, "CUDA pack_bits: {}", err);
  return words;
}

Result<std::vector<uint8_t>> unpack_bits(std::span<const uint32_t> words, int bits, size_t count) {
  if (bits < 1 || bits > 8)
    return make_error(Errc::kInvalidArgument, "unpack_bits: {} bits per code (1..8)", bits);
  if (words.size() < (count * static_cast<size_t>(bits) + 31) / 32)
    return make_error(Errc::kInvalidArgument, "unpack_bits: {} words hold fewer than {} codes of {} bits",
                      words.size(), count, bits);
  if (count == 0) return std::vector<uint8_t>{};
  if (!available()) return make_error(Errc::kUnsupported, "no CUDA device");
  std::vector<uint8_t> codes(count);
  char err[256] = {};
  if (mqde_cuda_unpack_bits(words.data(), codes.data(), static_cast<int64_t>(count), bits, &g_last_ms, err,
                            sizeof err) != 0)
    return make_error(Errc::kInternal, "CUDA unpack_bits: {}", err);
  return codes;
}

Result<pack::IntPacked> pack_gptq(const uint8_t* codes, const uint8_t* zeros, const float* scales,
                                  int64_t out, int64_t in, int64_t group_size, int bits, bool v1) {
  // the same checks as pack::pack_gptq, so a bad shape is the same error on either path
  if (bits < 1 || bits > 8) return make_error(Errc::kInvalidArgument, "bits must be in [1, 8], got {}", bits);
  if (out <= 0 || in <= 0) return make_error(Errc::kInvalidArgument, "empty weight [{}, {}]", out, in);
  if (group_size <= 0 || in % group_size != 0)
    return make_error(Errc::kInvalidArgument, "in_features {} is not divisible by group_size {}", in,
                      group_size);
  if ((in * bits) % 32 != 0 || (out * bits) % 32 != 0)
    return make_error(Errc::kInvalidArgument,
                      "GPTQ packing needs in*bits and out*bits divisible by 32 ([{}, {}], {} bit)", out, in,
                      bits);
  if (!available()) return make_error(Errc::kUnsupported, "no CUDA device");
  const int64_t groups = in / group_size;
  pack::IntPacked p;
  p.qweight_rows = in * bits / 32;
  p.qweight_cols = out;
  p.qzeros_rows = groups;
  p.qzeros_cols = out * bits / 32;
  p.qweight.assign(static_cast<size_t>(p.qweight_rows * out), 0);
  p.qzeros.assign(static_cast<size_t>(groups * p.qzeros_cols), 0);
  p.scales.assign(static_cast<size_t>(groups * out), 0);
  char err[256] = {};
  if (mqde_cuda_pack_gptq(codes, zeros, scales, out, in, group_size, bits, v1, p.qweight.data(),
                          p.qzeros.data(), p.scales.data(), &g_last_ms, err, sizeof err) != 0)
    return make_error(Errc::kInternal, "CUDA pack_gptq: {}", err);
  return p;
}

Result<pack::IntPacked> pack_awq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                                 int64_t in, int64_t group_size, int bits) {
  if (bits < 1 || bits > 8) return make_error(Errc::kInvalidArgument, "bits must be in [1, 8], got {}", bits);
  if (out <= 0 || in <= 0) return make_error(Errc::kInvalidArgument, "empty weight [{}, {}]", out, in);
  if (group_size <= 0 || in % group_size != 0)
    return make_error(Errc::kInvalidArgument, "in_features {} is not divisible by group_size {}", in,
                      group_size);
  if ((out * bits) % 32 != 0)
    return make_error(Errc::kInvalidArgument, "AWQ packing needs out*bits divisible by 32 (out={}, {} bit)",
                      out, bits);
  if (!available()) return make_error(Errc::kUnsupported, "no CUDA device");
  const int64_t groups = in / group_size;
  pack::IntPacked p;
  p.qweight_rows = in;
  p.qweight_cols = out * bits / 32;
  p.qzeros_rows = groups;
  p.qzeros_cols = p.qweight_cols;
  p.qweight.assign(static_cast<size_t>(in * p.qweight_cols), 0);
  p.qzeros.assign(static_cast<size_t>(groups * p.qzeros_cols), 0);
  p.scales.assign(static_cast<size_t>(groups * out), 0);
  char err[256] = {};
  if (mqde_cuda_pack_awq(codes, zeros, scales, out, in, group_size, bits, p.qweight.data(), p.qzeros.data(),
                         p.scales.data(), &g_last_ms, err, sizeof err) != 0)
    return make_error(Errc::kInternal, "CUDA pack_awq: {}", err);
  return p;
}

}  // namespace mqde::quant::cuda
