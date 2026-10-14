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
#include "mqde/quant/pack.h"

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
// pack.h pack_gptq on the GPU (qweight, qzeros and the fp16 scales); every field equals the CPU function's.
Result<pack::IntPacked> pack_gptq(const uint8_t* codes, const uint8_t* zeros, const float* scales,
                                  int64_t out, int64_t in, int64_t group_size, int bits, bool v1 = true);
// pack.h pack_awq on the GPU; every field equals the CPU function's.
Result<pack::IntPacked> pack_awq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                                 int64_t in, int64_t group_size, int bits = 4);
// pack.h gguf_quantize / gguf_dequantize on the GPU: the plain per-block scale-and-round GGUF types
// (Q8_0/Q4_0/Q5_0), not a tensor-core-shaped format like Marlin or a k-quant super-block (Q2_K/Q4_K) or the
// non-linear codebook of IQ4_NL -- those are each their own bitstream and are not implemented here. `n`
// must be a multiple of 32.
Result<std::vector<uint8_t>> gguf_quantize_q8_0(const float* x, int64_t n);
Result<void> gguf_dequantize_q8_0(const uint8_t* data, int64_t n, float* out);
Result<std::vector<uint8_t>> gguf_quantize_q4_0(const float* x, int64_t n);
Result<void> gguf_dequantize_q4_0(const uint8_t* data, int64_t n, float* out);
Result<std::vector<uint8_t>> gguf_quantize_q5_0(const float* x, int64_t n);
Result<void> gguf_dequantize_q5_0(const uint8_t* data, int64_t n, float* out);
// Wall-clock of the last call's kernels alone (calibration + qdq), in milliseconds, without the host<->device
// copies.
double last_kernel_ms();

}  // namespace mqde::quant::cuda
