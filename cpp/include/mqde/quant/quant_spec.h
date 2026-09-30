#pragma once
// QuantSpec: how one tensor (weight / activation / KV) is quantized.
// Numerical conventions are documented in docs/guides/quant-numerics.md and are the single
// source of truth for every fake/real quantization path in the system.

#include <cstdint>
#include <string>

#include "mqde/common/config.h"
#include "mqde/common/result.h"

namespace mqde::quant {

enum class NumFormat { kInt, kFp8E4M3, kFp8E5M2, kNone /* keep original precision */ };
enum class Granularity {
  kPerTensor,
  kPerChannel /* per row: out-channel for weights, token for acts */,
  kPerGroup
};
enum class CalibAlgo { kMinMax, kMse };

struct QuantSpec {
  NumFormat fmt = NumFormat::kNone;
  int bits = 16;
  bool symmetric = true;
  Granularity gran = Granularity::kPerChannel;
  int64_t group_size = 0;
  CalibAlgo calib = CalibAlgo::kMinMax;
  float maxshrink = 0.8f;  // MSE search lower bound on range shrink
  int grid = 20;           // MSE search steps

  bool enabled() const { return fmt != NumFormat::kNone; }
  int32_t qmin() const;  // integer formats only
  int32_t qmax() const;
  float fp8_max() const;  // 448 (E4M3) / 57344 (E5M2)
  // Storage bits per element, e.g. 4 for INT4, 8 for FP8, 16 when disabled.
  double storage_bits() const { return fmt == NumFormat::kNone ? 16.0 : static_cast<double>(bits); }
};

// JSON form (as used in data/recipes/*.yaml):
//   {format: int|fp8_e4m3|fp8_e5m2|none, bits: 4, symmetric: true,
//    granularity: per_tensor|per_channel|per_token|per_group, group_size: 128, calib: minmax|mse}
Result<QuantSpec> parse_quant_spec(const Json& j, std::string_view where);
Json to_json(const QuantSpec& s);
std::string describe(const QuantSpec& s);  // "INT4 asym g128 mse"

}  // namespace mqde::quant
