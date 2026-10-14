// Built with --fmad=false: a fused multiply-add rounds once where the scalar path rounds twice.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#include "quant_cuda_impl.h"

namespace {

struct GP {
  float scale, zero;
};

__device__ __forceinline__ float rne(float v) { return rintf(v); }  // ties to even, as std::nearbyint

// ops.cpp fp8_round: nearest FP8 value, ties to even, saturating. ldexpf / frexpf are exact.
__device__ float fp8_round(float v, int fp8) {
  const int mbits = fp8 == 1 ? 3 : 2;
  const int emin = fp8 == 1 ? -6 : -14;  // exponent of the smallest normal
  const float fmax = fp8 == 1 ? 448.0f : 57344.0f;
  if (isnan(v)) return v;
  const float a = fabsf(v);
  if (a == 0.0f) return v;
  int e = 0;
  frexpf(a, &e);
  const int exp = max(e - 1, emin);
  const float step = ldexpf(1.0f, exp - mbits);
  float r = __fmul_rn(rne(__fdiv_rn(a, step)), step);
  r = fmax < r ? fmax : r;  // std::min(r, fmax)
  return copysignf(r, v);
}

// Order of float operations is the numerical contract (ops.cpp params_from_range).
__device__ GP params_from_range(float mn, float mx, const MqdeCudaQuantArgs& a) {
  const int symmetric = a.symmetric;
  const float qmin = a.qmin, qmax = a.qmax;
  if (a.fp8) {
    const float amax = fmaxf(fabsf(mn), fabsf(mx));
    float scale = __fdiv_rn(amax, a.fp8_max);
    if (!(scale > 0.0f)) scale = 1.0f;
    return {scale, 0.0f};
  }
  if (symmetric) {
    const float amax = fmaxf(fabsf(mn), fabsf(mx));
    float scale = __fdiv_rn(amax, qmax);
    if (!(scale > 0.0f)) scale = 1.0f;
    return {scale, 0.0f};
  }
  mn = mn < 0.0f ? mn : 0.0f;  // std::min(mn, 0.0f)
  mx = 0.0f < mx ? mx : 0.0f;  // std::max(mx, 0.0f)
  float scale = __fdiv_rn(__fsub_rn(mx, mn), __fsub_rn(qmax, qmin));
  if (!(scale > 0.0f)) scale = 1.0f;
  float zero = rne(__fsub_rn(qmin, __fdiv_rn(mn, scale)));
  zero = zero < qmin ? qmin : (qmax < zero ? qmax : zero);  // std::clamp
  return {scale, zero};
}

__device__ __forceinline__ float qdq1(float v, GP p, const MqdeCudaQuantArgs& a) {
  const float qmin = a.qmin, qmax = a.qmax;
  if (a.fp8) {
    const float fm = a.fp8_max;
    const float t = __fdiv_rn(v, p.scale);
    const float c = t < -fm ? -fm : (fm < t ? fm : t);  // std::clamp keeps a NaN
    return __fmul_rn(fp8_round(c, a.fp8), p.scale);
  }
  float q = __fadd_rn(rne(__fdiv_rn(v, p.scale)), p.zero);
  q = q < qmin ? qmin : (qmax < q ? qmax : q);  // std::clamp keeps a NaN
  return __fmul_rn(__fsub_rn(q, p.zero), p.scale);
}

// One group = n_seg segments of seg_len floats, seg_stride apart (a per-tensor group is every row of the matrix).
__global__ void calibrate(const float* x, float* scale, float* zero, int64_t n_groups, MqdeCudaQuantArgs a) {
  const int64_t gi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gi >= n_groups) return;
  const float* base;
  int64_t n_seg, seg_len, seg_stride;
  if (a.per_tensor) {
    base = x;
    n_seg = a.rows;
    seg_len = a.cols;
    seg_stride = a.cols;
  } else {
    const int64_t gpr = a.cols / a.group_cols;
    base = x + (gi / gpr) * a.cols + (gi % gpr) * a.group_cols;
    n_seg = 1;
    seg_len = a.group_cols;
    seg_stride = a.cols;
  }
  float mn = INFINITY, mx = -INFINITY;
  for (int64_t s = 0; s < n_seg; ++s) {
    const float* row = base + s * seg_stride;
    for (int64_t c = 0; c < seg_len; ++c) {
      const float v = row[c];
      mn = v < mn ? v : mn;  // std::min(mn, v): a NaN never replaces
      mx = mx < v ? v : mx;
    }
  }
  GP best;
  if (!isfinite(mn) || !isfinite(mx)) {
    best = {1.0f, 0.0f};
  } else {
    best = params_from_range(mn, mx, a);
    if (a.mse) {
      double best_err = INFINITY;
      for (int i = 0; i <= a.grid; ++i) {
        const float shrink = __fsub_rn(1.0f, __fdiv_rn(__fmul_rn(static_cast<float>(i), __fsub_rn(1.0f, a.maxshrink)),
                                                       static_cast<float>(a.grid)));
        const GP p = params_from_range(__fmul_rn(mn, shrink), __fmul_rn(mx, shrink), a);
        double err = 0.0;
        for (int64_t s = 0; s < n_seg; ++s) {
          const float* row = base + s * seg_stride;
          for (int64_t c = 0; c < seg_len; ++c) {
            const double d = __dsub_rn(static_cast<double>(qdq1(row[c], p, a)), static_cast<double>(row[c]));
            err = __dadd_rn(err, __dmul_rn(d, d));
          }
        }
        if (err < best_err) {
          best_err = err;
          best = p;
        }
      }
    }
  }
  scale[gi] = best.scale;
  if (zero) zero[gi] = best.zero;
}

__global__ void qdq(const float* x, float* out, const float* scale, const float* zero, MqdeCudaQuantArgs a) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= a.rows * a.cols) return;
  const int64_t gi = a.per_tensor ? 0 : (i / a.cols) * (a.cols / a.group_cols) + (i % a.cols) / a.group_cols;
  out[i] = qdq1(x[i], {scale[gi], zero ? zero[gi] : 0.0f}, a);
}


// One thread per output word: gather the codes whose bits fall in [32w, 32w + 32). A code that started in the previous
// word contributes its high part (a right shift), one that starts here its low part (a left shift): pack.cpp pack_bits.
__global__ void pack_words(const uint8_t* codes, uint32_t* words, int64_t count, int64_t n_words, int bits) {
  const int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (w >= n_words) return;
  const uint32_t mask = (1u << bits) - 1u;
  const int64_t lo = w * 32, hi = lo + 32;  // bit range of this word
  int64_t first = lo / bits;                // the first code that touches it
  uint32_t acc = 0;
  for (int64_t i = first; i < count && i * bits < hi; ++i) {
    const uint32_t c = codes[i] & mask;
    const int64_t sh = i * bits - lo;  // negative: the code began in an earlier word
    acc |= sh >= 0 ? (c << sh) : (c >> (-sh));
  }
  words[w] = acc;
}

__global__ void unpack_codes(const uint32_t* words, uint8_t* codes, int64_t count, int bits) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t mask = (1u << bits) - 1u;
  const int64_t bit = i * bits;
  const int64_t word = bit / 32;
  const unsigned shift = static_cast<unsigned>(bit % 32);
  uint32_t v = words[word] >> shift;
  if (shift + bits > 32) v |= words[word + 1] << (32 - shift);
  codes[i] = static_cast<uint8_t>(v & mask);
}

struct Dev {
  float *x = nullptr, *out = nullptr, *scale = nullptr, *zero = nullptr;
  ~Dev() {
    cudaFree(x);
    cudaFree(out);
    cudaFree(scale);
    cudaFree(zero);
  }
};

int fail(char* err, size_t n, const char* what, cudaError_t e) {
  std::snprintf(err, n, "%s: %s", what, cudaGetErrorString(e));
  return 1;
}

}  // namespace

int mqde_cuda_device_count() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}

int mqde_cuda_fake_quant(const float* x, float* out, float* scale, float* zero, const MqdeCudaQuantArgs& a,
                         double* kernel_ms, char* err, size_t errlen) {
  const int64_t n = a.rows * a.cols;
  const int64_t groups = a.per_tensor ? 1 : a.rows * (a.cols / a.group_cols);
  Dev d;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&d.x, n * sizeof(float)), "cudaMalloc x");
  CK(cudaMalloc(&d.out, n * sizeof(float)), "cudaMalloc out");
  CK(cudaMalloc(&d.scale, groups * sizeof(float)), "cudaMalloc scale");
  if (zero) CK(cudaMalloc(&d.zero, groups * sizeof(float)), "cudaMalloc zero");
  CK(cudaMemcpy(d.x, x, n * sizeof(float), cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  calibrate<<<static_cast<unsigned>((groups + 127) / 128), 128>>>(d.x, d.scale, d.zero, groups, a);
  CK(cudaGetLastError(), "calibrate launch");
  qdq<<<static_cast<unsigned>((n + 255) / 256), 256>>>(d.x, d.out, d.scale, d.zero, a);
  CK(cudaGetLastError(), "qdq launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "kernels");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(out, d.out, n * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
  CK(cudaMemcpy(scale, d.scale, groups * sizeof(float), cudaMemcpyDeviceToHost), "copy scale");
  if (zero) CK(cudaMemcpy(zero, d.zero, groups * sizeof(float), cudaMemcpyDeviceToHost), "copy zero");
#undef CK
  return 0;
}

namespace {
template <class T>
struct Buf {
  T* p = nullptr;
  ~Buf() { cudaFree(p); }
};
}  // namespace

int mqde_cuda_pack_bits(const uint8_t* codes, uint32_t* words, int64_t count, int bits, double* kernel_ms, char* err,
                        size_t errlen) {
  const int64_t n_words = (count * bits + 31) / 32;
  Buf<uint8_t> dc;
  Buf<uint32_t> dw;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&dc.p, count), "cudaMalloc codes");
  CK(cudaMalloc(&dw.p, n_words * sizeof(uint32_t)), "cudaMalloc words");
  CK(cudaMemcpy(dc.p, codes, count, cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  pack_words<<<static_cast<unsigned>((n_words + 255) / 256), 256>>>(dc.p, dw.p, count, n_words, bits);
  CK(cudaGetLastError(), "pack launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "pack");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(words, dw.p, n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}

int mqde_cuda_unpack_bits(const uint32_t* words, uint8_t* codes, int64_t count, int bits, double* kernel_ms, char* err,
                          size_t errlen) {
  const int64_t n_words = (count * bits + 31) / 32;
  Buf<uint8_t> dc;
  Buf<uint32_t> dw;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&dc.p, count), "cudaMalloc codes");
  CK(cudaMalloc(&dw.p, n_words * sizeof(uint32_t)), "cudaMalloc words");
  CK(cudaMemcpy(dw.p, words, n_words * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  unpack_codes<<<static_cast<unsigned>((count + 255) / 256), 256>>>(dw.p, dc.p, count, bits);
  CK(cudaGetLastError(), "unpack launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "unpack");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(codes, dc.p, count, cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}
