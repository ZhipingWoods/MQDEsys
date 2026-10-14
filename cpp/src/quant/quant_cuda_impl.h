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
