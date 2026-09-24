#pragma once
// Quantization kernels on row-major float32 matrices [rows, cols].
// Weights are [out_features, in_features]; activations are [tokens, hidden].
// Grouping: per_tensor = 1 group; per_channel = 1 group per row; per_group = cols/group_size per row.
// Fake and real paths share compute_qparams(), which is what keeps them bit-consistent.

#include <cstdint>
#include <vector>

#include "mqde/quant/quant_spec.h"

namespace mqde::quant {

struct QParams {
  std::vector<float> scale;  // one per group, row-major [rows, groups_per_row] (or [1] for per_tensor)
  std::vector<float> zero;   // integer asymmetric only; empty otherwise (stored as float, integral values)
  int64_t groups_per_row = 1;
  int64_t group_cols = 0;    // columns per group
  bool per_tensor = false;
};

Result<QParams> compute_qparams(const float* x, int64_t rows, int64_t cols, const QuantSpec& spec);

// Quantize -> dequantize in one go. `out` may alias `x`. Optionally returns the qparams used.
Result<void> fake_quant(const float* x, int64_t rows, int64_t cols, const QuantSpec& spec, float* out,
                        QParams* qparams_out = nullptr);

// Real integer path: codes in [qmin, qmax] (int16 storage so 8-bit asymmetric 0..255 fits; bit packing is a later stage).
Result<void> quantize_int(const float* x, int64_t rows, int64_t cols, const QuantSpec& spec, const QParams& qp,
                          int16_t* codes);
Result<void> dequantize_int(const int16_t* codes, int64_t rows, int64_t cols, const QuantSpec& spec,
                            const QParams& qp, float* out);

// Round a value (already scaled into range) to the nearest FP8 value, ties-to-even, saturating.
float fp8_round(float v, NumFormat fmt);

}  // namespace mqde::quant
