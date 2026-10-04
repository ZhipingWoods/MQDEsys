#pragma once
// Bit packing into the checkpoint layouts that real low-bit kernels consume (T-0405).
//
//   pack_bits / unpack_bits  LSB-first contiguous bit stream over little-endian 32-bit words: code i
//                            occupies bits [i*bits, (i+1)*bits). Every GPTQ width (2/4/8 and the
//                            10-1-10-1-10 3-bit scheme) and AWQ are exactly this stream.
//   GPTQ (GPTQModel/AutoGPTQ) qweight [in*bits/32, out]  packs along the input dim (one column per output)
//                            qzeros  [groups, out*bits/32] packs along the output dim
//                            scales  [groups, out] fp16; v1 checkpoints store zero-1 (mod 2^bits), v2 store
//                            zero
//   AWQ GEMM (llm-awq)       qweight [in, out*bits/32] packs along the output dim; qzeros [groups,
//   out*bits/32];
//                            4-bit words hold outputs in the order [0,2,4,6,1,3,5,7]
//   GGUF blocks              Q8_0 / Q4_0 / Q5_0 / IQ4_NL / Q4_K / Q2_K; quantizers are bit-identical
//                            ports of what llama-quantize runs without an imatrix (ggml_quantize_chunk:
//                            quantize_row_*_ref, and quantize_iq4_nl with its 7-step scale search for
//                            IQ4_NL), dequantizers match dequantize_row_*.
//
// Integer codes are unsigned in [0, 2^bits - 1] with dequant w = scale * (code - zero). Symmetric specs
// (codes in [-2^(b-1), 2^(b-1)-1], zero 0) are exported by adding 2^(b-1) to both codes and zero.

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "mqde/common/result.h"

namespace mqde::quant::pack {

// IEEE binary16, round-to-nearest-even (same bit pattern as ggml's GGML_FP32_TO_FP16).
uint16_t fp32_to_fp16(float f);
float fp16_to_fp32(uint16_t h);

std::vector<uint32_t> pack_bits(std::span<const uint8_t> codes, int bits);
std::vector<uint8_t> unpack_bits(std::span<const uint32_t> words, int bits, size_t count);

struct IntPacked {
  std::vector<int32_t> qweight, qzeros;
  std::vector<uint16_t> scales;  // fp16 bits, [groups, out]
  int64_t qweight_rows = 0, qweight_cols = 0, qzeros_rows = 0, qzeros_cols = 0;
};

// codes [out, in] and zeros/scales [out, groups] are row-major (the QParams layout); groups = in /
// group_size.
Result<IntPacked> pack_gptq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                            int64_t in, int64_t group_size, int bits, bool v1 = true);
Result<void> unpack_gptq(const IntPacked& p, int64_t out, int64_t in, int64_t group_size, int bits, bool v1,
                         uint8_t* codes, uint8_t* zeros);
Result<IntPacked> pack_awq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                           int64_t in, int64_t group_size, int bits = 4);
Result<void> unpack_awq(const IntPacked& p, int64_t out, int64_t in, int64_t group_size, int bits,
                        uint8_t* codes, uint8_t* zeros);

// Marlin (vLLM gptq_marlin / awq_marlin with 16-bit activations), 4 or 8 bit, desc_act=False. Ports of
// vllm marlin_utils_test.marlin_weights / marlin_utils.marlin_permute_scales / marlin_zero_points (v0.30):
//   qweight [in/16, out*16*bits/32]  16x16 tiles of the [in, out] code matrix, each 1024-element run
//                                    permuted for the tensor-core fragment order, then LSB-first packed
//   scales  [groups, out] fp16       column-permuted (64-wide perm when grouped, 32-wide when per-channel)
//   qzeros  [groups, out*bits/32]    awq_marlin only (with_zeros): raw zero, 64-wide perm + interleave
// Symmetric GPTQ (uint4b8 / uint8b128) carries no zeros: codes already hold the 2^(b-1) offset.
// Layout needs in % 16 == 0 and out % 64 == 0; the vLLM kernel additionally needs in % 128 == 0.
struct MarlinPacked {
  std::vector<int32_t> qweight, qzeros;
  std::vector<uint16_t> scales;
  int64_t qweight_rows = 0, qweight_cols = 0, groups = 0, qzeros_cols = 0;
};
Result<MarlinPacked> pack_marlin(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                                 int64_t in, int64_t group_size, int bits, bool with_zeros);
// zeros / scales may be null; scales come back as fp32 [out, groups].
Result<void> unpack_marlin(const MarlinPacked& p, int64_t out, int64_t in, int64_t group_size, int bits,
                           uint8_t* codes, uint8_t* zeros, float* scales);

// ggml_type ids, so tensors can be written into GGUF files unchanged.
enum class GgufType : uint32_t { kQ4_0 = 2, kQ5_0 = 6, kQ8_0 = 8, kQ2_K = 10, kQ4_K = 12, kIQ4_NL = 20 };

struct GgufTraits {
  int64_t block_elems;
  int64_t block_bytes;
  std::string_view name;
};

// "q4_0" | "q5_0" | "q8_0" | "iq4_nl" | "q2_k" | "q4_k" (any case)
Result<GgufType> gguf_type_from_name(std::string_view name);
GgufTraits gguf_traits(GgufType t);
Result<std::vector<uint8_t>> gguf_quantize(GgufType t, const float* x, int64_t n);  // n % block_elems == 0
Result<void> gguf_dequantize(GgufType t, const uint8_t* data, int64_t n, float* out);

}  // namespace mqde::quant::pack
