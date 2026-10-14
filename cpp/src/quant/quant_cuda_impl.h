#pragma once
// The plain-C boundary between quant_cuda.cpp (C++23, the project's types) and quant_cuda.cu (nvcc, no
// project headers).
#include <cstddef>
#include <cstdint>

struct MqdeCudaQuantArgs {
  int64_t rows, cols;
  int64_t group_cols;  // columns per group (cols for per-channel and per-tensor)
  int per_tensor;      // one group spanning the whole matrix
  int symmetric;
  int mse;  // MSE grid search instead of min/max
  int grid;
  float maxshrink;
  float qmin, qmax;  // integer formats
  int fp8;           // 0 integer, 1 E4M3, 2 E5M2
  float fp8_max;     // 448 / 57344
};

// All return 0 on success; otherwise they write a message into err[errlen].
int mqde_cuda_device_count();
int mqde_cuda_fake_quant(const float* x, float* out, float* scale, float* zero /* nullptr when symmetric */,
                         const MqdeCudaQuantArgs& a, double* kernel_ms, char* err, size_t errlen);

// LSB-first bit stream over little-endian 32-bit words (pack.h pack_bits / unpack_bits), 1..8 bits per code.
int mqde_cuda_pack_bits(const uint8_t* codes, uint32_t* words, int64_t count, int bits, double* kernel_ms,
                        char* err, size_t errlen);
int mqde_cuda_unpack_bits(const uint32_t* words, uint8_t* codes, int64_t count, int bits, double* kernel_ms,
                          char* err, size_t errlen);

// GPTQ checkpoint layout (pack.h pack_gptq): qweight [in*bits/32, out], qzeros [groups, out*bits/32], scales
// fp16 [groups, out]. Shapes are validated by the caller.
int mqde_cuda_pack_gptq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                        int64_t in, int64_t group_size, int bits, int v1, int32_t* qweight, int32_t* qzeros,
                        uint16_t* fp16_scales, double* kernel_ms, char* err, size_t errlen);

// AWQ GEMM layout (pack.h pack_awq): qweight [in, out*bits/32], qzeros [groups, out*bits/32], scales fp16
// [groups, out]; 4-bit words hold the outputs of each block of 8 in the order [0,2,4,6,1,3,5,7].
int mqde_cuda_pack_awq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                       int64_t in, int64_t group_size, int bits, int32_t* qweight, int32_t* qzeros,
                       uint16_t* fp16_scales, double* kernel_ms, char* err, size_t errlen);

// GGUF Q8_0 (pack.h gguf_quantize/gguf_dequantize, GgufType::kQ8_0): blocks of 32 elements, each a 2-byte
// fp16 scale followed by 32 signed int8 codes (34 bytes/block, pack.cpp's BlockQ8_0 layout). n must be a
// multiple of 32; out holds n/32*34 bytes.
int mqde_cuda_gguf_quantize_q8_0(const float* x, uint8_t* out, int64_t n, double* kernel_ms, char* err,
                                 size_t errlen);
int mqde_cuda_gguf_dequantize_q8_0(const uint8_t* data, float* out, int64_t n, double* kernel_ms, char* err,
                                   size_t errlen);
// GGUF Q4_0 (GgufType::kQ4_0): blocks of 32, a 2-byte fp16 scale + 16 bytes of packed 4-bit codes (18
// bytes/block, BlockQ4_0). n must be a multiple of 32; out holds n/32*18 bytes.
int mqde_cuda_gguf_quantize_q4_0(const float* x, uint8_t* out, int64_t n, double* kernel_ms, char* err,
                                 size_t errlen);
int mqde_cuda_gguf_dequantize_q4_0(const uint8_t* data, float* out, int64_t n, double* kernel_ms, char* err,
                                   size_t errlen);
// GGUF Q5_0 (GgufType::kQ5_0): blocks of 32, a 2-byte fp16 scale + 4 bytes holding each code's 5th bit +
// 16 bytes of packed low 4 bits (22 bytes/block, BlockQ5_0). n must be a multiple of 32; out holds
// n/32*22 bytes.
int mqde_cuda_gguf_quantize_q5_0(const float* x, uint8_t* out, int64_t n, double* kernel_ms, char* err,
                                 size_t errlen);
int mqde_cuda_gguf_dequantize_q5_0(const uint8_t* data, float* out, int64_t n, double* kernel_ms, char* err,
                                   size_t errlen);
