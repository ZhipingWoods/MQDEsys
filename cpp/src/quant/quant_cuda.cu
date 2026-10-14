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


// Word `w` of an LSB-first stream of `count` codes of `bits` bits: gather the codes whose bits fall in [32w, 32w + 32). A code
// that started in the previous word contributes its high part (a right shift), one that starts here its low part (a left
// shift): pack.cpp pack_bits. `get(i)` returns code i (unmasked; masked here).
template <class Get>
__device__ __forceinline__ uint32_t stream_word(Get get, int64_t count, int64_t w, int bits) {
  const uint32_t mask = (1u << bits) - 1u;
  const int64_t lo = w * 32, hi = lo + 32;  // bit range of this word
  uint32_t acc = 0;
  for (int64_t i = lo / bits; i < count && i * bits < hi; ++i) {
    const uint32_t c = get(i) & mask;
    const int64_t sh = i * bits - lo;  // negative: the code began in an earlier word
    acc |= sh >= 0 ? (c << sh) : (c >> (-sh));
  }
  return acc;
}

__global__ void pack_words(const uint8_t* codes, uint32_t* words, int64_t count, int64_t n_words, int bits) {
  const int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (w >= n_words) return;
  words[w] = stream_word([&](int64_t i) { return static_cast<uint32_t>(codes[i]); }, count, w, bits);
}

// pack.cpp pack_gptq: qweight[r * out + o] = word r of the stream of row o of the code matrix [out, in]
__global__ void gptq_qweight(const uint8_t* codes, int32_t* qweight, int64_t out, int64_t in, int64_t rows, int bits) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= rows * out) return;
  const int64_t r = t / out, o = t % out;
  const uint8_t* row = codes + o * in;
  qweight[t] = static_cast<int32_t>(stream_word([&](int64_t i) { return static_cast<uint32_t>(row[i]); }, in, r, bits));
}

// qzeros[g * cols + c] = word c of the stream of the zero points of group g over all outputs (v1 stores zero - 1)
__global__ void gptq_qzeros(const uint8_t* zeros, int32_t* qzeros, int64_t out, int64_t groups, int64_t cols, int bits, int v1) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= groups * cols) return;
  const int64_t g = t / cols, c = t % cols;
  const uint32_t mask = (1u << bits) - 1u;
  qzeros[t] = static_cast<int32_t>(stream_word(
      [&](int64_t o) {
        const uint32_t z = zeros[o * groups + g];
        return v1 ? ((z - 1u) & mask) : z;
      },
      out, c, bits));
}

// pack.cpp awq_interleave: with 4 bits, stream position j of a block of 8 holds output (j / 8) * 8 + {0,2,4,6,1,3,5,7}[j % 8]
__device__ __forceinline__ int64_t awq_source(int64_t j, int bits) {
  if (bits != 4) return j;
  const int order = (j % 8) < 4 ? 2 * static_cast<int>(j % 8) : 2 * static_cast<int>(j % 8 - 4) + 1;
  return (j / 8) * 8 + order;
}

// qweight[i * cols + c] = word c of the stream, over the outputs, of the codes of input i
__global__ void awq_qweight(const uint8_t* codes, int32_t* qweight, int64_t out, int64_t in, int64_t cols, int bits) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= in * cols) return;
  const int64_t i = t / cols, c = t % cols;
  qweight[t] = static_cast<int32_t>(
      stream_word([&](int64_t j) { return static_cast<uint32_t>(codes[awq_source(j, bits) * in + i]); }, out, c, bits));
}

__global__ void awq_qzeros(const uint8_t* zeros, int32_t* qzeros, int64_t out, int64_t groups, int64_t cols, int bits) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= groups * cols) return;
  const int64_t g = t / cols, c = t % cols;
  qzeros[t] = static_cast<int32_t>(
      stream_word([&](int64_t j) { return static_cast<uint32_t>(zeros[awq_source(j, bits) * groups + g]); }, out, c, bits));
}

// pack.cpp fp32_to_fp16: the same branch-free round-to-nearest-even port (not __float2half_rn: NaN is the canonical 0x7E00)
__device__ uint16_t fp32_to_fp16(float f) {
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
  float base = __fmul_rn(__fmul_rn(fabsf(f), scale_to_inf), scale_to_zero);
  const uint32_t w = __float_as_uint(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & 0x80000000u;
  uint32_t bias = shl1_w & 0xFF000000u;
  if (bias < 0x71000000u) bias = 0x71000000u;
  base = __fadd_rn(__uint_as_float((bias >> 1) + 0x07800000u), base);
  const uint32_t bits = __float_as_uint(base);
  const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
  const uint32_t mantissa_bits = bits & 0x00000FFFu;
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

__global__ void gptq_scales(const float* scales, uint16_t* out16, int64_t out, int64_t groups) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= groups * out) return;
  const int64_t g = t / out, o = t % out;
  out16[t] = fp32_to_fp16(scales[o * groups + g]);
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

int mqde_cuda_pack_gptq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out, int64_t in,
                        int64_t group_size, int bits, int v1, int32_t* qweight, int32_t* qzeros, uint16_t* fp16_scales,
                        double* kernel_ms, char* err, size_t errlen) {
  const int64_t groups = in / group_size, rows = in * bits / 32, zcols = out * bits / 32;
  Buf<uint8_t> dc, dz;
  Buf<float> ds;
  Buf<int32_t> dqw, dqz;
  Buf<uint16_t> d16;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&dc.p, out * in), "cudaMalloc codes");
  CK(cudaMalloc(&dz.p, out * groups), "cudaMalloc zeros");
  CK(cudaMalloc(&ds.p, out * groups * sizeof(float)), "cudaMalloc scales");
  CK(cudaMalloc(&dqw.p, rows * out * sizeof(int32_t)), "cudaMalloc qweight");
  CK(cudaMalloc(&dqz.p, groups * zcols * sizeof(int32_t)), "cudaMalloc qzeros");
  CK(cudaMalloc(&d16.p, groups * out * sizeof(uint16_t)), "cudaMalloc fp16 scales");
  CK(cudaMemcpy(dc.p, codes, out * in, cudaMemcpyHostToDevice), "copy codes");
  CK(cudaMemcpy(dz.p, zeros, out * groups, cudaMemcpyHostToDevice), "copy zeros");
  CK(cudaMemcpy(ds.p, scales, out * groups * sizeof(float), cudaMemcpyHostToDevice), "copy scales");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  gptq_qweight<<<static_cast<unsigned>((rows * out + 255) / 256), 256>>>(dc.p, dqw.p, out, in, rows, bits);
  CK(cudaGetLastError(), "qweight launch");
  gptq_qzeros<<<static_cast<unsigned>((groups * zcols + 255) / 256), 256>>>(dz.p, dqz.p, out, groups, zcols, bits, v1);
  CK(cudaGetLastError(), "qzeros launch");
  gptq_scales<<<static_cast<unsigned>((groups * out + 255) / 256), 256>>>(ds.p, d16.p, out, groups);
  CK(cudaGetLastError(), "scales launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "gptq pack");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(qweight, dqw.p, rows * out * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy qweight");
  CK(cudaMemcpy(qzeros, dqz.p, groups * zcols * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy qzeros");
  CK(cudaMemcpy(fp16_scales, d16.p, groups * out * sizeof(uint16_t), cudaMemcpyDeviceToHost), "copy fp16 scales");
#undef CK
  return 0;
}

int mqde_cuda_pack_awq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out, int64_t in,
                       int64_t group_size, int bits, int32_t* qweight, int32_t* qzeros, uint16_t* fp16_scales,
                       double* kernel_ms, char* err, size_t errlen) {
  const int64_t groups = in / group_size, cols = out * bits / 32;
  Buf<uint8_t> dc, dz;
  Buf<float> ds;
  Buf<int32_t> dqw, dqz;
  Buf<uint16_t> d16;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&dc.p, out * in), "cudaMalloc codes");
  CK(cudaMalloc(&dz.p, out * groups), "cudaMalloc zeros");
  CK(cudaMalloc(&ds.p, out * groups * sizeof(float)), "cudaMalloc scales");
  CK(cudaMalloc(&dqw.p, in * cols * sizeof(int32_t)), "cudaMalloc qweight");
  CK(cudaMalloc(&dqz.p, groups * cols * sizeof(int32_t)), "cudaMalloc qzeros");
  CK(cudaMalloc(&d16.p, groups * out * sizeof(uint16_t)), "cudaMalloc fp16 scales");
  CK(cudaMemcpy(dc.p, codes, out * in, cudaMemcpyHostToDevice), "copy codes");
  CK(cudaMemcpy(dz.p, zeros, out * groups, cudaMemcpyHostToDevice), "copy zeros");
  CK(cudaMemcpy(ds.p, scales, out * groups * sizeof(float), cudaMemcpyHostToDevice), "copy scales");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  awq_qweight<<<static_cast<unsigned>((in * cols + 255) / 256), 256>>>(dc.p, dqw.p, out, in, cols, bits);
  CK(cudaGetLastError(), "qweight launch");
  awq_qzeros<<<static_cast<unsigned>((groups * cols + 255) / 256), 256>>>(dz.p, dqz.p, out, groups, cols, bits);
  CK(cudaGetLastError(), "qzeros launch");
  gptq_scales<<<static_cast<unsigned>((groups * out + 255) / 256), 256>>>(ds.p, d16.p, out, groups);
  CK(cudaGetLastError(), "scales launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "awq pack");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(qweight, dqw.p, in * cols * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy qweight");
  CK(cudaMemcpy(qzeros, dqz.p, groups * cols * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy qzeros");
  CK(cudaMemcpy(fp16_scales, d16.p, groups * out * sizeof(uint16_t), cudaMemcpyDeviceToHost), "copy fp16 scales");
#undef CK
  return 0;
}

namespace {

// pack.cpp's BlockQ8_0 layout: a 2-byte fp16 scale followed by 32 signed int8 codes, 34 bytes, no padding
// (both fields' natural alignment is already satisfied since every block starts at a multiple-of-34, and
// 34 is even, offset so the uint16_t stays 2-byte aligned).
constexpr int kGgufQ8K = 32;
constexpr int kGgufQ8BlockBytes = 2 + kGgufQ8K;

// pack.cpp fp16_to_fp32: ported bit for bit (half -> float widening is always exact, so this only has to
// match the CPU's bit-manipulation path, not round the same way -- there is no rounding to match).
__device__ float fp16_to_fp32(uint16_t h) {
  const uint32_t w = static_cast<uint32_t>(h) << 16;
  const uint32_t sign = w & 0x80000000u;
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = 0xE0u << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized = __uint_as_float((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask = 126u << 23;
  const float magic_bias = 0.5f;
  const float denormalized = __uint_as_float((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = 1u << 27;
  const uint32_t result = sign | (two_w < denormalized_cutoff ? __float_as_uint(denormalized) : __float_as_uint(normalized));
  return __uint_as_float(result);
}

// pack.cpp quantize_q8_0: one thread per 32-element block, same sequential amax and the same two divisions
// (amax/127, then 1/that) and round-half-away-from-zero (std::round) as the scalar path.
__global__ void gguf_q8_0_quantize(const float* x, uint8_t* out, int64_t nb) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  const float* blk = x + i * kGgufQ8K;
  float amax = 0.0f;
  for (int j = 0; j < kGgufQ8K; ++j) amax = fmaxf(amax, fabsf(blk[j]));
  const float d = __fdiv_rn(amax, 127.0f);
  const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
  uint8_t* o = out + i * kGgufQ8BlockBytes;
  *reinterpret_cast<uint16_t*>(o) = fp32_to_fp16(d);
  for (int j = 0; j < kGgufQ8K; ++j)
    o[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(roundf(__fmul_rn(blk[j], id))));
}

// pack.cpp dequantize_q8_0: one thread per output element.
__global__ void gguf_q8_0_dequantize(const uint8_t* data, float* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t blk = i / kGgufQ8K, j = i % kGgufQ8K;
  const uint8_t* b = data + blk * kGgufQ8BlockBytes;
  const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(b));
  const int8_t q = static_cast<int8_t>(b[2 + j]);
  out[i] = __fmul_rn(static_cast<float>(q), d);
}

}  // namespace

int mqde_cuda_gguf_quantize_q8_0(const float* x, uint8_t* out, int64_t n, double* kernel_ms, char* err,
                                 size_t errlen) {
  const int64_t nb = n / kGgufQ8K;
  Buf<float> dx;
  Buf<uint8_t> dout;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&dx.p, n * sizeof(float)), "cudaMalloc x");
  CK(cudaMalloc(&dout.p, nb * kGgufQ8BlockBytes), "cudaMalloc out");
  CK(cudaMemcpy(dx.p, x, n * sizeof(float), cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  gguf_q8_0_quantize<<<static_cast<unsigned>((nb + 255) / 256), 256>>>(dx.p, dout.p, nb);
  CK(cudaGetLastError(), "quantize launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "quantize");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(out, dout.p, nb * kGgufQ8BlockBytes, cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}

int mqde_cuda_gguf_dequantize_q8_0(const uint8_t* data, float* out, int64_t n, double* kernel_ms, char* err,
                                   size_t errlen) {
  const int64_t nb = n / kGgufQ8K;
  Buf<uint8_t> din;
  Buf<float> dout;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&din.p, nb * kGgufQ8BlockBytes), "cudaMalloc data");
  CK(cudaMalloc(&dout.p, n * sizeof(float)), "cudaMalloc out");
  CK(cudaMemcpy(din.p, data, nb * kGgufQ8BlockBytes, cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  gguf_q8_0_dequantize<<<static_cast<unsigned>((n + 255) / 256), 256>>>(din.p, dout.p, n);
  CK(cudaGetLastError(), "dequantize launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "dequantize");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(out, dout.p, n * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}

namespace {

// pack.cpp's BlockQ4_0 (18 bytes: fp16 d, 16 bytes of packed 4-bit codes) and BlockQ5_0 (22 bytes: fp16 d,
// 4 bytes holding every code's 5th bit, 16 bytes of packed low-4-bits).
constexpr int kGgufQ4K = 32;
constexpr int kGgufQ4BlockBytes = 2 + kGgufQ4K / 2;
constexpr int kGgufQ5BlockBytes = 2 + 4 + kGgufQ4K / 2;

// pack.cpp quantize_q4_0: one thread per 32-element block. amax/max track together (ties keep the earlier
// element, like the CPU's strict `<`), d = max/-8 (not amax: the sign survives), the fast
// round-and-bias-by-8.5 trick clamped to 15, two nibbles per output byte.
__global__ void gguf_q4_0_quantize(const float* x, uint8_t* out, int64_t nb) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  const float* blk = x + i * kGgufQ4K;
  float amax = 0.0f, mx = 0.0f;
  for (int j = 0; j < kGgufQ4K; ++j) {
    const float v = blk[j];
    const float av = fabsf(v);
    if (amax < av) { amax = av; mx = v; }
  }
  const float d = __fdiv_rn(mx, -8.0f);
  const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
  uint8_t* o = out + i * kGgufQ4BlockBytes;
  *reinterpret_cast<uint16_t*>(o) = fp32_to_fp16(d);
  uint8_t* qs = o + 2;
  for (int j = 0; j < kGgufQ4K / 2; ++j) {
    const float x0 = __fmul_rn(blk[j], id);
    const float x1 = __fmul_rn(blk[j + kGgufQ4K / 2], id);
    const int xi0 = min(15, static_cast<int>(static_cast<int8_t>(__fadd_rn(x0, 8.5f))));
    const int xi1 = min(15, static_cast<int>(static_cast<int8_t>(__fadd_rn(x1, 8.5f))));
    qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
  }
}

// pack.cpp dequantize_q4_0: one thread per output element.
__global__ void gguf_q4_0_dequantize(const uint8_t* data, float* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t blk = i / kGgufQ4K, r = i % kGgufQ4K;
  const uint8_t* b = data + blk * kGgufQ4BlockBytes;
  const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(b));
  const uint8_t* qs = b + 2;
  const int val = r < kGgufQ4K / 2 ? (qs[r] & 0x0F) - 8 : (qs[r - kGgufQ4K / 2] >> 4) - 8;
  out[i] = __fmul_rn(static_cast<float>(val), d);
}

// pack.cpp quantize_q5_0: like Q4_0, plus a 5th bit per code packed into a 32-bit `qh` (one thread owns the
// whole block, so the bitwise-OR accumulation order matches the scalar loop exactly, though OR itself does
// not care about order).
__global__ void gguf_q5_0_quantize(const float* x, uint8_t* out, int64_t nb) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  const float* blk = x + i * kGgufQ4K;
  float amax = 0.0f, mx = 0.0f;
  for (int j = 0; j < kGgufQ4K; ++j) {
    const float v = blk[j];
    const float av = fabsf(v);
    if (amax < av) { amax = av; mx = v; }
  }
  const float d = __fdiv_rn(mx, -16.0f);
  const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
  uint8_t* o = out + i * kGgufQ5BlockBytes;
  *reinterpret_cast<uint16_t*>(o) = fp32_to_fp16(d);
  uint8_t* qh_bytes = o + 2;
  uint8_t* qs = o + 6;
  uint32_t qh = 0;
  for (int j = 0; j < kGgufQ4K / 2; ++j) {
    const float x0 = __fmul_rn(blk[j], id);
    const float x1 = __fmul_rn(blk[j + kGgufQ4K / 2], id);
    const uint32_t xi0 = static_cast<uint32_t>(min(31, static_cast<int>(static_cast<int8_t>(__fadd_rn(x0, 16.5f)))));
    const uint32_t xi1 = static_cast<uint32_t>(min(31, static_cast<int>(static_cast<int8_t>(__fadd_rn(x1, 16.5f)))));
    qs[j] = static_cast<uint8_t>((xi0 & 0x0Fu) | ((xi1 & 0x0Fu) << 4));
    qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
    qh |= ((xi1 & 0x10u) >> 4) << (j + kGgufQ4K / 2);
  }
  memcpy(qh_bytes, &qh, sizeof(qh));
}

// pack.cpp dequantize_q5_0: one thread per output element.
__global__ void gguf_q5_0_dequantize(const uint8_t* data, float* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t blk = i / kGgufQ4K, r = i % kGgufQ4K;
  const uint8_t* b = data + blk * kGgufQ5BlockBytes;
  const float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(b));
  uint32_t qh;
  memcpy(&qh, b + 2, sizeof(qh));
  const uint8_t* qs = b + 6;
  int32_t val;
  if (r < kGgufQ4K / 2) {
    const uint32_t xh0 = ((qh >> (r + 0)) << 4) & 0x10u;
    val = static_cast<int32_t>((qs[r] & 0x0Fu) | xh0) - 16;
  } else {
    const int64_t j = r - kGgufQ4K / 2;
    const uint32_t xh1 = (qh >> (j + 12)) & 0x10u;
    val = static_cast<int32_t>((qs[j] >> 4) | xh1) - 16;
  }
  out[i] = __fmul_rn(static_cast<float>(val), d);
}

}  // namespace

int mqde_cuda_gguf_quantize_q4_0(const float* x, uint8_t* out, int64_t n, double* kernel_ms, char* err,
                                 size_t errlen) {
  const int64_t nb = n / kGgufQ4K;
  Buf<float> dx;
  Buf<uint8_t> dout;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&dx.p, n * sizeof(float)), "cudaMalloc x");
  CK(cudaMalloc(&dout.p, nb * kGgufQ4BlockBytes), "cudaMalloc out");
  CK(cudaMemcpy(dx.p, x, n * sizeof(float), cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  gguf_q4_0_quantize<<<static_cast<unsigned>((nb + 255) / 256), 256>>>(dx.p, dout.p, nb);
  CK(cudaGetLastError(), "quantize launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "quantize");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(out, dout.p, nb * kGgufQ4BlockBytes, cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}

int mqde_cuda_gguf_dequantize_q4_0(const uint8_t* data, float* out, int64_t n, double* kernel_ms, char* err,
                                   size_t errlen) {
  const int64_t nb = n / kGgufQ4K;
  Buf<uint8_t> din;
  Buf<float> dout;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&din.p, nb * kGgufQ4BlockBytes), "cudaMalloc data");
  CK(cudaMalloc(&dout.p, n * sizeof(float)), "cudaMalloc out");
  CK(cudaMemcpy(din.p, data, nb * kGgufQ4BlockBytes, cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  gguf_q4_0_dequantize<<<static_cast<unsigned>((n + 255) / 256), 256>>>(din.p, dout.p, n);
  CK(cudaGetLastError(), "dequantize launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "dequantize");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(out, dout.p, n * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}

int mqde_cuda_gguf_quantize_q5_0(const float* x, uint8_t* out, int64_t n, double* kernel_ms, char* err,
                                 size_t errlen) {
  const int64_t nb = n / kGgufQ4K;
  Buf<float> dx;
  Buf<uint8_t> dout;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&dx.p, n * sizeof(float)), "cudaMalloc x");
  CK(cudaMalloc(&dout.p, nb * kGgufQ5BlockBytes), "cudaMalloc out");
  CK(cudaMemcpy(dx.p, x, n * sizeof(float), cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  gguf_q5_0_quantize<<<static_cast<unsigned>((nb + 255) / 256), 256>>>(dx.p, dout.p, nb);
  CK(cudaGetLastError(), "quantize launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "quantize");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(out, dout.p, nb * kGgufQ5BlockBytes, cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}

int mqde_cuda_gguf_dequantize_q5_0(const uint8_t* data, float* out, int64_t n, double* kernel_ms, char* err,
                                   size_t errlen) {
  const int64_t nb = n / kGgufQ4K;
  Buf<uint8_t> din;
  Buf<float> dout;
  cudaError_t e;
#define CK(call, what) \
  if ((e = (call)) != cudaSuccess) return fail(err, errlen, what, e)
  CK(cudaMalloc(&din.p, nb * kGgufQ5BlockBytes), "cudaMalloc data");
  CK(cudaMalloc(&dout.p, n * sizeof(float)), "cudaMalloc out");
  CK(cudaMemcpy(din.p, data, nb * kGgufQ5BlockBytes, cudaMemcpyHostToDevice), "copy in");
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  gguf_q5_0_dequantize<<<static_cast<unsigned>((n + 255) / 256), 256>>>(din.p, dout.p, n);
  CK(cudaGetLastError(), "dequantize launch");
  cudaEventRecord(t1);
  CK(cudaEventSynchronize(t1), "dequantize");
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  if (kernel_ms) *kernel_ms = ms;
  CK(cudaMemcpy(out, dout.p, n * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
#undef CK
  return 0;
}
