#include "quant_kernels.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MQDE_KERNELS_AVX2 1
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define MQDE_KERNELS_NEON 1
#endif

namespace mqde::quant::kernels {
namespace {

inline float qdq1(float v, const IntQdq& p) {
  const float q = std::clamp(std::nearbyint(v / p.scale) + p.zero, p.qmin, p.qmax);
  return (q - p.zero) * p.scale;
}

// ---------------------------------------------------------------- scalar (the reference)
void minmax_scalar(const float* x, int64_t n, float* mn, float* mx) {
  float lo = *mn, hi = *mx;
  for (int64_t i = 0; i < n; ++i) {
    lo = std::min(lo, x[i]);
    hi = std::max(hi, x[i]);
  }
  *mn = lo;
  *mx = hi;
}

double sq_err_scalar(const float* x, int64_t n, const IntQdq& p, double acc) {
  for (int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(qdq1(x[i], p)) - x[i];
    acc += d * d;
  }
  return acc;
}

void qdq_scalar(const float* x, int64_t n, const IntQdq& p, float* out) {
  for (int64_t i = 0; i < n; ++i) out[i] = qdq1(x[i], p);
}

void quantize_scalar(const float* x, int64_t n, const IntQdq& p, int16_t* codes) {
  for (int64_t i = 0; i < n; ++i)
    codes[i] = static_cast<int16_t>(std::clamp(std::nearbyint(x[i] / p.scale) + p.zero, p.qmin, p.qmax));
}

void dequantize_scalar(const int16_t* codes, int64_t n, const IntQdq& p, float* out) {
  for (int64_t i = 0; i < n; ++i) out[i] = (static_cast<float>(codes[i]) - p.zero) * p.scale;
}

const Set kScalar{"scalar", minmax_scalar, sq_err_scalar, qdq_scalar, quantize_scalar, dequantize_scalar};

#ifdef MQDE_KERNELS_AVX2
// ---------------------------------------------------------------- AVX2 (8 floats)
// No FMA target on purpose: a fused multiply-add would round differently from the scalar's separate
// operations.
#define AVX2 __attribute__((target("avx2")))

AVX2 inline __m256 qdq8(__m256 v, __m256 scale, __m256 zero, __m256 qmin, __m256 qmax) {
  __m256 q = _mm256_div_ps(v, scale);
  q = _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);  // ties to even, like nearbyint
  q = _mm256_add_ps(q, zero);
  // std::clamp(q, lo, hi) = q < lo ? lo : hi < q ? hi : q -- a compare and a blend keep its NaN behaviour
  // (min/max would not)
  q = _mm256_blendv_ps(q, qmin, _mm256_cmp_ps(q, qmin, _CMP_LT_OQ));
  q = _mm256_blendv_ps(q, qmax, _mm256_cmp_ps(qmax, q, _CMP_LT_OQ));
  return _mm256_mul_ps(_mm256_sub_ps(q, zero), scale);
}

AVX2 void minmax_avx2(const float* x, int64_t n, float* mn, float* mx) {
  __m256 lo = _mm256_set1_ps(*mn), hi = _mm256_set1_ps(*mx);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 v = _mm256_loadu_ps(x + i);
    lo = _mm256_min_ps(v, lo);  // v < lo ? v : lo, the order std::min(lo, v) has; NaN in v is dropped
    hi = _mm256_max_ps(v, hi);
  }
  alignas(32) float a[8], b[8];
  _mm256_store_ps(a, lo);
  _mm256_store_ps(b, hi);
  float l = *mn, h = *mx;
  for (int k = 0; k < 8; ++k) {
    l = std::min(l, a[k]);
    h = std::max(h, b[k]);
  }
  for (; i < n; ++i) {
    l = std::min(l, x[i]);
    h = std::max(h, x[i]);
  }
  *mn = l;
  *mx = h;
}

AVX2 double sq_err_avx2(const float* x, int64_t n, const IntQdq& p, double acc) {
  const __m256 sc = _mm256_set1_ps(p.scale), ze = _mm256_set1_ps(p.zero), qn = _mm256_set1_ps(p.qmin),
               qx = _mm256_set1_ps(p.qmax);
  alignas(32) double sq[8];
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 v = _mm256_loadu_ps(x + i);
    const __m256 q = qdq8(v, sc, ze, qn, qx);
    const __m256d dlo =
        _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(q)), _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
    const __m256d dhi = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(q, 1)),
                                      _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
    _mm256_store_pd(sq, _mm256_mul_pd(dlo, dlo));
    _mm256_store_pd(sq + 4, _mm256_mul_pd(dhi, dhi));
    for (int k = 0; k < 8; ++k) acc += sq[k];  // element order, as the scalar does
  }
  for (; i < n; ++i) {
    const double d = static_cast<double>(qdq1(x[i], p)) - x[i];
    acc += d * d;
  }
  return acc;
}

AVX2 void qdq_avx2(const float* x, int64_t n, const IntQdq& p, float* out) {
  const __m256 sc = _mm256_set1_ps(p.scale), ze = _mm256_set1_ps(p.zero), qn = _mm256_set1_ps(p.qmin),
               qx = _mm256_set1_ps(p.qmax);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, qdq8(_mm256_loadu_ps(x + i), sc, ze, qn, qx));
  for (; i < n; ++i) out[i] = qdq1(x[i], p);
}

// the clamped value is integral and within [qmin, qmax] (at most 8 bits), so the truncating conversion is
// exact
AVX2 void quantize_avx2(const float* x, int64_t n, const IntQdq& p, int16_t* codes) {
  const __m256 sc = _mm256_set1_ps(p.scale), ze = _mm256_set1_ps(p.zero), qn = _mm256_set1_ps(p.qmin),
               qx = _mm256_set1_ps(p.qmax);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 q = _mm256_div_ps(_mm256_loadu_ps(x + i), sc);
    q = _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    q = _mm256_add_ps(q, ze);
    q = _mm256_blendv_ps(q, qn, _mm256_cmp_ps(q, qn, _CMP_LT_OQ));
    q = _mm256_blendv_ps(q, qx, _mm256_cmp_ps(qx, q, _CMP_LT_OQ));
    const __m256i i32 = _mm256_cvttps_epi32(q);
    // packs works within 128-bit lanes: lanes 0,1 | 2,3 of the 64-bit view hold the eight values in order
    // after a permute
    const __m256i packed = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32, i32), 0xD8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(codes + i), _mm256_castsi256_si128(packed));
  }
  quantize_scalar(x + i, n - i, p, codes + i);
}

AVX2 void dequantize_avx2(const int16_t* codes, int64_t n, const IntQdq& p, float* out) {
  const __m256 sc = _mm256_set1_ps(p.scale), ze = _mm256_set1_ps(p.zero);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i c16 = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i)));
    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(c16), ze), sc));
  }
  dequantize_scalar(codes + i, n - i, p, out + i);
}

const Set kAvx2{"avx2", minmax_avx2, sq_err_avx2, qdq_avx2, quantize_avx2, dequantize_avx2};
bool avx2_runnable() { return __builtin_cpu_supports("avx2"); }
#endif

#ifdef MQDE_KERNELS_NEON
// ---------------------------------------------------------------- NEON (4 floats)
inline float32x4_t qdq4(float32x4_t v, float32x4_t scale, float32x4_t zero, float32x4_t qmin,
                        float32x4_t qmax) {
  float32x4_t q = vdivq_f32(v, scale);
  q = vrndnq_f32(q);  // ties to even
  q = vaddq_f32(q, zero);
  q = vbslq_f32(vcltq_f32(q, qmin), qmin,
                q);  // std::clamp spelled out (vmaxq/vminq give the same bits here: both pass a NaN through)
  q = vbslq_f32(vcltq_f32(qmax, q), qmax, q);
  return vmulq_f32(vsubq_f32(q, zero), scale);
}

void minmax_neon(const float* x, int64_t n, float* mn, float* mx) {
  float32x4_t lo = vdupq_n_f32(*mn), hi = vdupq_n_f32(*mx);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t v = vld1q_f32(x + i);
    lo = vbslq_f32(vcltq_f32(v, lo), v, lo);
    hi = vbslq_f32(vcgtq_f32(v, hi), v, hi);
  }
  float a[4], b[4];
  vst1q_f32(a, lo);
  vst1q_f32(b, hi);
  float l = *mn, h = *mx;
  for (int k = 0; k < 4; ++k) {
    l = std::min(l, a[k]);
    h = std::max(h, b[k]);
  }
  for (; i < n; ++i) {
    l = std::min(l, x[i]);
    h = std::max(h, x[i]);
  }
  *mn = l;
  *mx = h;
}

double sq_err_neon(const float* x, int64_t n, const IntQdq& p, double acc) {
  const float32x4_t sc = vdupq_n_f32(p.scale), ze = vdupq_n_f32(p.zero), qn = vdupq_n_f32(p.qmin),
                    qx = vdupq_n_f32(p.qmax);
  double sq[4];
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t v = vld1q_f32(x + i);
    const float32x4_t q = qdq4(v, sc, ze, qn, qx);
    const float64x2_t dlo = vsubq_f64(vcvt_f64_f32(vget_low_f32(q)), vcvt_f64_f32(vget_low_f32(v)));
    const float64x2_t dhi = vsubq_f64(vcvt_high_f64_f32(q), vcvt_high_f64_f32(v));
    vst1q_f64(sq, vmulq_f64(dlo, dlo));
    vst1q_f64(sq + 2, vmulq_f64(dhi, dhi));
    for (int k = 0; k < 4; ++k) acc += sq[k];
  }
  for (; i < n; ++i) {
    const double d = static_cast<double>(qdq1(x[i], p)) - x[i];
    acc += d * d;
  }
  return acc;
}

void qdq_neon(const float* x, int64_t n, const IntQdq& p, float* out) {
  const float32x4_t sc = vdupq_n_f32(p.scale), ze = vdupq_n_f32(p.zero), qn = vdupq_n_f32(p.qmin),
                    qx = vdupq_n_f32(p.qmax);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) vst1q_f32(out + i, qdq4(vld1q_f32(x + i), sc, ze, qn, qx));
  for (; i < n; ++i) out[i] = qdq1(x[i], p);
}

void quantize_neon(const float* x, int64_t n, const IntQdq& p, int16_t* codes) {
  const float32x4_t sc = vdupq_n_f32(p.scale), ze = vdupq_n_f32(p.zero), qn = vdupq_n_f32(p.qmin),
                    qx = vdupq_n_f32(p.qmax);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t q = vrndnq_f32(vdivq_f32(vld1q_f32(x + i), sc));
    q = vaddq_f32(q, ze);
    q = vbslq_f32(vcltq_f32(q, qn), qn, q);
    q = vbslq_f32(vcltq_f32(qx, q), qx, q);
    vst1_s16(codes + i, vmovn_s32(vcvtq_s32_f32(q)));
  }
  quantize_scalar(x + i, n - i, p, codes + i);
}

void dequantize_neon(const int16_t* codes, int64_t n, const IntQdq& p, float* out) {
  const float32x4_t sc = vdupq_n_f32(p.scale), ze = vdupq_n_f32(p.zero);
  int64_t i = 0;
  for (; i + 4 <= n; i += 4)
    vst1q_f32(out + i, vmulq_f32(vsubq_f32(vcvtq_f32_s32(vmovl_s16(vld1_s16(codes + i))), ze), sc));
  dequantize_scalar(codes + i, n - i, p, out + i);
}

const Set kNeon{"neon", minmax_neon, sq_err_neon, qdq_neon, quantize_neon, dequantize_neon};
#endif

std::atomic<const Set*> g_forced{nullptr};

const Set& pick_default() {
  if (const char* e = std::getenv("MQDE_QUANT_KERNELS"); e && *e)
    if (const Set* s = find(e)) return *s;
  const auto all = available();
  return *all.back();  // the widest (listed last)
}

}  // namespace

const Set& scalar() { return kScalar; }

std::vector<const Set*> available() {
  std::vector<const Set*> v{&kScalar};
#ifdef MQDE_KERNELS_AVX2
  if (avx2_runnable()) v.push_back(&kAvx2);
#endif
#ifdef MQDE_KERNELS_NEON
  v.push_back(&kNeon);
#endif
  return v;
}

const Set* find(std::string_view name) {
  for (const Set* s : available())
    if (name == s->name) return s;
  return nullptr;
}

const Set& active() {
  if (const Set* f = g_forced.load(std::memory_order_acquire)) return *f;
  static const Set& dflt = pick_default();
  return dflt;
}

void force(const Set* s) { g_forced.store(s, std::memory_order_release); }

}  // namespace mqde::quant::kernels
