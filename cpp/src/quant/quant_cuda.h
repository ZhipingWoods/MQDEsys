#pragma once
// CUDA implementation of the integer fake-quantizer (calibration + quantize-dequantize), built only with
// MQDE_WITH_CUDA. Same contract as the CPU kernels: the scales, zero points and outputs equal the scalar
// path's BITS. One thread owns one group and walks it in the scalar's order (the MSE error is summed in
// double, element after element), with explicit IEEE-754 round-to-nearest operations and no FMA. Internal
// header: not part of the public API.

#include "mqde/quant/ops.h"

namespace mqde::quant::cuda {

bool available();  // a CUDA device is present and usable
// Same as quant::fake_quant for integer formats (any granularity, minmax or MSE calibration), run on the GPU.
// `x` and `out` are host pointers; `out` may alias `x`. FP8 is not implemented here (kUnsupported).
Result<void> fake_quant(const float* x, int64_t rows, int64_t cols, const QuantSpec& spec, float* out,
                        QParams* qparams_out = nullptr);
// Wall-clock of the last call's kernels alone (calibration + qdq), in milliseconds, without the host<->device
// copies.
double last_kernel_ms();

}  // namespace mqde::quant::cuda
