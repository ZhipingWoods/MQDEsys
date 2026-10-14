#pragma once
// CUDA implementation of the fake-quantizer (calibration + quantize-dequantize), built only with
// MQDE_WITH_CUDA. Same contract as the CPU kernels: the scales, zero points and outputs equal the scalar
// path's BITS. One thread owns one group and walks it in the scalar's order (the MSE error is summed in
// double, element after element), with explicit IEEE-754 round-to-nearest operations and no FMA. Internal
// header: not part of the public API.

#include <cstdint>
#include <span>
#include <vector>

#include "mqde/quant/ops.h"

namespace mqde::quant::cuda {

bool available();  // a CUDA device is present and usable
// Same as quant::fake_quant (integer and FP8 formats, any granularity, minmax or MSE calibration), run on the
// GPU. `x` and `out` are host pointers; `out` may alias `x`.
Result<void> fake_quant(const float* x, int64_t rows, int64_t cols, const QuantSpec& spec, float* out,
                        QParams* qparams_out = nullptr);
// pack.h pack_bits / unpack_bits on the GPU (LSB-first bit stream over 32-bit words, 1..8 bits per code); the
// words and the codes equal the CPU functions'. A code is masked to `bits`, as the CPU does.
Result<std::vector<uint32_t>> pack_bits(std::span<const uint8_t> codes, int bits);
Result<std::vector<uint8_t>> unpack_bits(std::span<const uint32_t> words, int bits, size_t count);
// Wall-clock of the last call's kernels alone (calibration + qdq), in milliseconds, without the host<->device
// copies.
double last_kernel_ms();

}  // namespace mqde::quant::cuda
