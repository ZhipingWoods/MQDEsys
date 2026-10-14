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
  float qmin, qmax;
};

// All return 0 on success; otherwise they write a message into err[errlen].
int mqde_cuda_device_count();
int mqde_cuda_fake_quant(const float* x, float* out, float* scale, float* zero /* nullptr when symmetric */,
                         const MqdeCudaQuantArgs& a, double* kernel_ms, char* err, size_t errlen);
