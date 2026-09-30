#include "mqde/quant/pack.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>

namespace mqde::quant::pack {
namespace {

uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float u2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

Result<void> check_int(int64_t out, int64_t in, int64_t group_size, int bits) {
  if (bits < 1 || bits > 8) return make_error(Errc::kInvalidArgument, "bits must be in [1, 8], got {}", bits);
  if (out <= 0 || in <= 0) return make_error(Errc::kInvalidArgument, "empty weight [{}, {}]", out, in);
  if (group_size <= 0 || in % group_size != 0)
    return make_error(Errc::kInvalidArgument, "in_features {} is not divisible by group_size {}", in,
                      group_size);
  return {};
}

}  // namespace

// ------------------------------------------------------------------ fp16
// Branch-free RNE conversion from the FP16 library (Marat Dukhan, MIT), which ggml uses verbatim.
uint16_t fp32_to_fp16(float f) {
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
  float base = (std::fabs(f) * scale_to_inf) * scale_to_zero;
  const uint32_t w = f2u(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & 0x80000000u;
  uint32_t bias = shl1_w & 0xFF000000u;
  if (bias < 0x71000000u) bias = 0x71000000u;
  base = u2f((bias >> 1) + 0x07800000u) + base;
  const uint32_t bits = f2u(base);
  const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
  const uint32_t mantissa_bits = bits & 0x00000FFFu;
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

float fp16_to_fp32(uint16_t h) {
  const uint32_t w = static_cast<uint32_t>(h) << 16;
  const uint32_t sign = w & 0x80000000u;
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = 0xE0u << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized = u2f((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask = 126u << 23;
  const float magic_bias = 0.5f;
  const float denormalized = u2f((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = 1u << 27;
  const uint32_t result = sign | (two_w < denormalized_cutoff ? f2u(denormalized) : f2u(normalized));
  return u2f(result);
}

// ------------------------------------------------------------------ bit stream
std::vector<uint32_t> pack_bits(std::span<const uint8_t> codes, int bits) {
  std::vector<uint32_t> w((codes.size() * bits + 31) / 32, 0);
  const uint32_t mask = (1u << bits) - 1u;
  for (size_t i = 0; i < codes.size(); ++i) {
    const uint64_t bit = i * static_cast<uint64_t>(bits);
    const size_t word = bit / 32;
    const unsigned shift = bit % 32;
    const uint32_t c = codes[i] & mask;
    w[word] |= c << shift;
    if (shift + bits > 32) w[word + 1] |= c >> (32 - shift);
  }
  return w;
}

std::vector<uint8_t> unpack_bits(std::span<const uint32_t> words, int bits, size_t count) {
  std::vector<uint8_t> c(count);
  const uint32_t mask = (1u << bits) - 1u;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t bit = i * static_cast<uint64_t>(bits);
    const size_t word = bit / 32;
    const unsigned shift = bit % 32;
    uint32_t v = words[word] >> shift;
    if (shift + bits > 32) v |= words[word + 1] << (32 - shift);
    c[i] = static_cast<uint8_t>(v & mask);
  }
  return c;
}

// ------------------------------------------------------------------ GPTQ
Result<IntPacked> pack_gptq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                            int64_t in, int64_t group_size, int bits, bool v1) {
  if (auto ok = check_int(out, in, group_size, bits); !ok) return std::unexpected(ok.error());
  if ((in * bits) % 32 != 0 || (out * bits) % 32 != 0)
    return make_error(Errc::kInvalidArgument,
                      "GPTQ packing needs in*bits and out*bits divisible by 32 ([{}, {}], {} bit)", out, in,
                      bits);
  const int64_t groups = in / group_size;
  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
  IntPacked p;
  p.qweight_rows = in * bits / 32;
  p.qweight_cols = out;
  p.qweight.assign(p.qweight_rows * out, 0);
  for (int64_t o = 0; o < out; ++o) {
    const auto w = pack_bits({codes + o * in, static_cast<size_t>(in)}, bits);
    for (int64_t r = 0; r < p.qweight_rows; ++r) p.qweight[r * out + o] = static_cast<int32_t>(w[r]);
  }
  p.qzeros_rows = groups;
  p.qzeros_cols = out * bits / 32;
  p.qzeros.resize(groups * p.qzeros_cols);
  p.scales.resize(groups * out);
  std::vector<uint8_t> zrow(out);
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t o = 0; o < out; ++o) {
      const uint8_t z = zeros[o * groups + g];
      zrow[o] = v1 ? static_cast<uint8_t>((z - 1) & mask) : z;
      p.scales[g * out + o] = fp32_to_fp16(scales[o * groups + g]);
    }
    const auto w = pack_bits(zrow, bits);
    for (int64_t c = 0; c < p.qzeros_cols; ++c) p.qzeros[g * p.qzeros_cols + c] = static_cast<int32_t>(w[c]);
  }
  return p;
}

Result<void> unpack_gptq(const IntPacked& p, int64_t out, int64_t in, int64_t group_size, int bits, bool v1,
                         uint8_t* codes, uint8_t* zeros) {
  if (auto ok = check_int(out, in, group_size, bits); !ok) return ok;
  const int64_t groups = in / group_size;
  if (p.qweight_rows != in * bits / 32 || p.qweight_cols != out || p.qzeros_rows != groups)
    return make_error(Errc::kInvalidArgument, "GPTQ tensor shapes do not match [{}, {}] g{} {}bit", out, in,
                      group_size, bits);
  const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
  std::vector<uint32_t> col(p.qweight_rows);
  for (int64_t o = 0; o < out; ++o) {
    for (int64_t r = 0; r < p.qweight_rows; ++r) col[r] = static_cast<uint32_t>(p.qweight[r * out + o]);
    const auto c = unpack_bits(col, bits, in);
    std::copy(c.begin(), c.end(), codes + o * in);
  }
  for (int64_t g = 0; g < groups; ++g) {
    const auto* w = reinterpret_cast<const uint32_t*>(p.qzeros.data() + g * p.qzeros_cols);
    const auto z = unpack_bits({w, static_cast<size_t>(p.qzeros_cols)}, bits, out);
    for (int64_t o = 0; o < out; ++o)
      zeros[o * groups + g] = v1 ? static_cast<uint8_t>((z[o] + 1) & mask) : z[o];
  }
  return {};
}

// ------------------------------------------------------------------ AWQ
namespace {
constexpr std::array<int, 8> kAwqOrder = {0, 2, 4, 6, 1, 3, 5, 7};

void awq_interleave(std::vector<uint8_t>& v, int bits) {
  if (bits != 4) return;
  std::array<uint8_t, 8> t{};
  for (size_t b = 0; b + 8 <= v.size(); b += 8) {
    for (int k = 0; k < 8; ++k) t[k] = v[b + kAwqOrder[k]];
    std::copy(t.begin(), t.end(), v.begin() + b);
  }
}
void awq_deinterleave(std::vector<uint8_t>& v, int bits) {
  if (bits != 4) return;
  std::array<uint8_t, 8> t{};
  for (size_t b = 0; b + 8 <= v.size(); b += 8) {
    for (int k = 0; k < 8; ++k) t[kAwqOrder[k]] = v[b + k];
    std::copy(t.begin(), t.end(), v.begin() + b);
  }
}
}  // namespace

Result<IntPacked> pack_awq(const uint8_t* codes, const uint8_t* zeros, const float* scales, int64_t out,
                           int64_t in, int64_t group_size, int bits) {
  if (auto ok = check_int(out, in, group_size, bits); !ok) return std::unexpected(ok.error());
  if ((out * bits) % 32 != 0)
    return make_error(Errc::kInvalidArgument, "AWQ packing needs out*bits divisible by 32 (out={}, {} bit)",
                      out, bits);
  const int64_t groups = in / group_size;
  IntPacked p;
  p.qweight_rows = in;
  p.qweight_cols = out * bits / 32;
  p.qweight.resize(in * p.qweight_cols);
  std::vector<uint8_t> row(out);
  for (int64_t i = 0; i < in; ++i) {
    for (int64_t o = 0; o < out; ++o) row[o] = codes[o * in + i];
    awq_interleave(row, bits);
    const auto w = pack_bits(row, bits);
    for (int64_t c = 0; c < p.qweight_cols; ++c)
      p.qweight[i * p.qweight_cols + c] = static_cast<int32_t>(w[c]);
  }
  p.qzeros_rows = groups;
  p.qzeros_cols = p.qweight_cols;
  p.qzeros.resize(groups * p.qzeros_cols);
  p.scales.resize(groups * out);
  for (int64_t g = 0; g < groups; ++g) {
    for (int64_t o = 0; o < out; ++o) {
      row[o] = zeros[o * groups + g];
      p.scales[g * out + o] = fp32_to_fp16(scales[o * groups + g]);
    }
    awq_interleave(row, bits);
    const auto w = pack_bits(row, bits);
    for (int64_t c = 0; c < p.qzeros_cols; ++c) p.qzeros[g * p.qzeros_cols + c] = static_cast<int32_t>(w[c]);
  }
  return p;
}

Result<void> unpack_awq(const IntPacked& p, int64_t out, int64_t in, int64_t group_size, int bits,
                        uint8_t* codes, uint8_t* zeros) {
  if (auto ok = check_int(out, in, group_size, bits); !ok) return ok;
  const int64_t groups = in / group_size;
  if (p.qweight_rows != in || p.qweight_cols != out * bits / 32 || p.qzeros_rows != groups)
    return make_error(Errc::kInvalidArgument, "AWQ tensor shapes do not match [{}, {}] g{} {}bit", out, in,
                      group_size, bits);
  for (int64_t i = 0; i < in; ++i) {
    const auto* w = reinterpret_cast<const uint32_t*>(p.qweight.data() + i * p.qweight_cols);
    auto row = unpack_bits({w, static_cast<size_t>(p.qweight_cols)}, bits, out);
    awq_deinterleave(row, bits);
    for (int64_t o = 0; o < out; ++o) codes[o * in + i] = row[o];
  }
  for (int64_t g = 0; g < groups; ++g) {
    const auto* w = reinterpret_cast<const uint32_t*>(p.qzeros.data() + g * p.qzeros_cols);
    auto row = unpack_bits({w, static_cast<size_t>(p.qzeros_cols)}, bits, out);
    awq_deinterleave(row, bits);
    for (int64_t o = 0; o < out; ++o) zeros[o * groups + g] = row[o];
  }
  return {};
}

// ------------------------------------------------------------------ GGUF (ports of ggml-quants.c *_ref)
namespace {
constexpr int kQK = 32;    // QK4_0 / QK8_0
constexpr int kQKK = 256;  // QK_K

struct BlockQ4_0 {
  uint16_t d;
  uint8_t qs[kQK / 2];
};
struct BlockQ8_0 {
  uint16_t d;
  int8_t qs[kQK];
};
struct BlockQ2_K {
  uint8_t scales[kQKK / 16];
  uint8_t qs[kQKK / 4];
  uint16_t d;
  uint16_t dmin;
};
struct BlockQ4_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[kQKK / 2];
};
struct BlockQ5_0 {
  uint16_t d;
  uint8_t qh[4];  // 5th bit of each quant
  uint8_t qs[kQK / 2];
};
struct BlockIQ4_NL {
  uint16_t d;
  uint8_t qs[kQK / 2];
};
static_assert(sizeof(BlockQ4_0) == 18 && sizeof(BlockQ8_0) == 34 && sizeof(BlockQ2_K) == 84 &&
              sizeof(BlockQ4_K) == 144 && sizeof(BlockQ5_0) == 22 && sizeof(BlockIQ4_NL) == 18);

// ggml-common.h kvalues_iq4nl: the non-linear 4-bit codebook.
constexpr int8_t kIQ4NLValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                     1,    13,   25,  38,  53,  69,  89,  113};
constexpr float kGroupMaxEps = 1e-15f;  // ggml-quants.c GROUP_MAX_EPS

inline int nearest_int(float fval) {
  const float val = fval + 12582912.f;
  int i;
  std::memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, uint8_t* L, float* the_min,
                       uint8_t* Laux, float rmin, float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min) min = x[i];
    if (x[i] > max) max = x[i];
    const float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_error = 0;
  for (int i = 0; i < n; ++i) {
    const int l = nearest_int(iscale * (x[i] - min));
    L[i] = static_cast<uint8_t>(std::max(0, std::min(nmax, l)));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? std::fabs(diff) : diff * diff;
    best_error += weights[i] * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = std::max(0, std::min(nmax, l));
      Laux[i] = static_cast<uint8_t>(l);
      const float w = weights[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    const float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float cur_error = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? std::fabs(diff) : diff * diff;
        cur_error += weights[i] * diff;
      }
      if (cur_error < best_error) {
        for (int i = 0; i < n; ++i) L[i] = Laux[i];
        best_error = cur_error;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

void quantize_q4_0(const float* x, BlockQ4_0* y, int64_t k) {
  const int64_t nb = k / kQK;
  for (int64_t i = 0; i < nb; i++) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < kQK; j++) {
      const float v = x[i * kQK + j];
      if (amax < std::fabs(v)) {
        amax = std::fabs(v);
        max = v;
      }
    }
    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    for (int j = 0; j < kQK / 2; ++j) {
      const float x0 = x[i * kQK + 0 + j] * id;
      const float x1 = x[i * kQK + kQK / 2 + j] * id;
      const uint8_t xi0 = static_cast<uint8_t>(std::min<int>(15, static_cast<int8_t>(x0 + 8.5f)));
      const uint8_t xi1 = static_cast<uint8_t>(std::min<int>(15, static_cast<int8_t>(x1 + 8.5f)));
      y[i].qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
    }
  }
}

void dequantize_q4_0(const BlockQ4_0* x, float* y, int64_t k) {
  const int64_t nb = k / kQK;
  for (int64_t i = 0; i < nb; i++) {
    const float d = fp16_to_fp32(x[i].d);
    for (int j = 0; j < kQK / 2; ++j) {
      const int x0 = (x[i].qs[j] & 0x0F) - 8;
      const int x1 = (x[i].qs[j] >> 4) - 8;
      y[i * kQK + j + 0] = x0 * d;
      y[i * kQK + j + kQK / 2] = x1 * d;
    }
  }
}

// quantize_row_q5_0_ref
void quantize_q5_0(const float* x, BlockQ5_0* y, int64_t k) {
  const int64_t nb = k / kQK;
  for (int64_t i = 0; i < nb; i++) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < kQK; j++) {
      const float v = x[i * kQK + j];
      if (amax < std::fabs(v)) {
        amax = std::fabs(v);
        max = v;
      }
    }
    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    uint32_t qh = 0;
    for (int j = 0; j < kQK / 2; ++j) {
      const float x0 = x[i * kQK + 0 + j] * id;
      const float x1 = x[i * kQK + kQK / 2 + j] * id;
      const auto xi0 = static_cast<uint8_t>(std::min<int>(31, static_cast<int8_t>(x0 + 16.5f)));
      const auto xi1 = static_cast<uint8_t>(std::min<int>(31, static_cast<int8_t>(x1 + 16.5f)));
      y[i].qs[j] = static_cast<uint8_t>((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
      qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
      qh |= ((xi1 & 0x10u) >> 4) << (j + kQK / 2);
    }
    std::memcpy(y[i].qh, &qh, sizeof(qh));
  }
}

void dequantize_q5_0(const BlockQ5_0* x, float* y, int64_t k) {
  const int64_t nb = k / kQK;
  for (int64_t i = 0; i < nb; i++) {
    const float d = fp16_to_fp32(x[i].d);
    uint32_t qh;
    std::memcpy(&qh, x[i].qh, sizeof(qh));
    for (int j = 0; j < kQK / 2; ++j) {
      const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
      const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
      const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = ((x[i].qs[j] >> 4) | xh_1) - 16;
      y[i * kQK + j + 0] = static_cast<float>(x0) * d;
      y[i * kQK + j + kQK / 2] = static_cast<float>(x1) * d;
    }
  }
}

inline int best_index_int8(int n, const int8_t* val, float x) {
  if (x <= val[0]) return 0;
  if (x >= val[n - 1]) return n - 1;
  int ml = 0, mu = n - 1;
  while (mu - ml > 1) {
    const int mav = (ml + mu) / 2;
    if (x < val[mav])
      mu = mav;
    else
      ml = mav;
  }
  return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

// quantize_row_iq4_nl_impl(QK4_NL, 32, ..., quant_weights = NULL, ntry) for one 32-element block. ntry = 7 is
// what llama-quantize runs (quantize_iq4_nl via ggml_quantize_chunk); ntry = -1 is quantize_row_iq4_nl_ref.
void quantize_iq4_nl_block(const float* x, BlockIQ4_NL* y, int ntry) {
  const int8_t* values = kIQ4NLValues;
  uint8_t L[kQK];
  float weight[kQK];
  std::memset(y->qs, 0, sizeof(y->qs));
  y->d = fp32_to_fp16(0.f);
  for (int j = 0; j < kQK; ++j) weight[j] = x[j] * x[j];
  float amax = 0, max = 0;
  for (int j = 0; j < kQK; ++j) {
    const float ax = std::fabs(x[j]);
    if (ax > amax) {
      amax = ax;
      max = x[j];
    }
  }
  float scale = 0;
  if (amax >= kGroupMaxEps) {
    float d = ntry > 0 ? -max / values[0] : max / values[0];
    float id = 1 / d;
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < kQK; ++j) {
      const float al = id * x[j];
      const int l = best_index_int8(16, values, al);
      L[j] = static_cast<uint8_t>(l);
      const float q = values[l];
      const float w = weight[j];
      sumqx += w * q * x[j];
      sumq2 += w * q * q;
    }
    d = sumq2 > 0 ? sumqx / sumq2 : 0.f;
    float best = d * sumqx;
    for (int itry = -ntry; itry <= ntry; ++itry) {
      id = (static_cast<float>(itry) + values[0]) / max;
      sumqx = sumq2 = 0;
      for (int j = 0; j < kQK; ++j) {
        const float al = id * x[j];
        const int l = best_index_int8(16, values, al);
        const float q = values[l];
        const float w = weight[j];
        sumqx += w * q * x[j];
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
        d = sumqx / sumq2;
        best = d * sumqx;
      }
    }
    scale = d;
  } else {
    // ggml leaves L uninitialised for an all-zero block with ntry <= 0; the packed nibbles then come from
    // stack garbage. With ntry > 0 they are recomputed below (all map to the code nearest 0). Initialise
    // so both paths are deterministic.
    for (int j = 0; j < kQK; ++j) L[j] = static_cast<uint8_t>(best_index_int8(16, values, 0.f));
  }
  y->d = fp32_to_fp16(scale);
  if (ntry > 0) {
    const float id = scale ? 1 / scale : 0;
    for (int j = 0; j < kQK; ++j) L[j] = static_cast<uint8_t>(best_index_int8(16, values, id * x[j]));
  }
  for (int j = 0; j < 16; ++j) y->qs[j] = static_cast<uint8_t>(L[j] | (L[16 + j] << 4));
}

void quantize_iq4_nl(const float* x, BlockIQ4_NL* y, int64_t k, int ntry) {
  for (int64_t i = 0; i < k / kQK; ++i) quantize_iq4_nl_block(x + i * kQK, y + i, ntry);
}

void dequantize_iq4_nl(const BlockIQ4_NL* x, float* y, int64_t k) {
  for (int64_t i = 0; i < k / kQK; i++) {
    const float d = fp16_to_fp32(x[i].d);
    for (int j = 0; j < kQK / 2; ++j) {
      y[i * kQK + j] = d * kIQ4NLValues[x[i].qs[j] & 0xf];
      y[i * kQK + j + kQK / 2] = d * kIQ4NLValues[x[i].qs[j] >> 4];
    }
  }
}

void quantize_q8_0(const float* x, BlockQ8_0* y, int64_t k) {
  const int64_t nb = k / kQK;
  for (int64_t i = 0; i < nb; i++) {
    float amax = 0.0f;
    for (int j = 0; j < kQK; j++) amax = std::max(amax, std::fabs(x[i * kQK + j]));
    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    for (int j = 0; j < kQK; ++j) y[i].qs[j] = static_cast<int8_t>(std::round(x[i * kQK + j] * id));
  }
}

void dequantize_q8_0(const BlockQ8_0* x, float* y, int64_t k) {
  const int64_t nb = k / kQK;
  for (int64_t i = 0; i < nb; i++) {
    const float d = fp16_to_fp32(x[i].d);
    for (int j = 0; j < kQK; ++j) y[i * kQK + j] = x[i].qs[j] * d;
  }
}

void quantize_q2_K(const float* x, BlockQ2_K* y, int64_t k) {
  const int64_t nb = k / kQKK;
  uint8_t L[kQKK];
  uint8_t Laux[16];
  float weights[16];
  float mins[kQKK / 16];
  float scales[kQKK / 16];
  const float q4scale = 15.f;
  for (int64_t i = 0; i < nb; i++) {
    float max_scale = 0, max_min = 0;
    for (int j = 0; j < kQKK / 16; ++j) {
      for (int l = 0; l < 16; ++l) weights[l] = std::fabs(x[16 * j + l]);
      scales[j] =
          make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
      if (scales[j] > max_scale) max_scale = scales[j];
      if (mins[j] > max_min) max_min = mins[j];
    }
    if (max_scale > 0) {
      const float iscale = q4scale / max_scale;
      for (int j = 0; j < kQKK / 16; ++j)
        y[i].scales[j] = static_cast<uint8_t>(nearest_int(iscale * scales[j]));
      y[i].d = fp32_to_fp16(max_scale / q4scale);
    } else {
      for (int j = 0; j < kQKK / 16; ++j) y[i].scales[j] = 0;
      y[i].d = fp32_to_fp16(0.f);
    }
    if (max_min > 0) {
      const float iscale = q4scale / max_min;
      for (int j = 0; j < kQKK / 16; ++j)
        y[i].scales[j] |= static_cast<uint8_t>(nearest_int(iscale * mins[j]) << 4);
      y[i].dmin = fp32_to_fp16(max_min / q4scale);
    } else {
      y[i].dmin = fp32_to_fp16(0.f);
    }
    for (int j = 0; j < kQKK / 16; ++j) {
      const float d = fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF);
      if (!d) continue;
      const float dm = fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4);
      for (int ii = 0; ii < 16; ++ii) {
        const int l = nearest_int((x[16 * j + ii] + dm) / d);
        L[16 * j + ii] = static_cast<uint8_t>(std::max(0, std::min(3, l)));
      }
    }
    for (int j = 0; j < kQKK; j += 128)
      for (int l = 0; l < 32; ++l)
        y[i].qs[j / 4 + l] = static_cast<uint8_t>(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) |
                                                  (L[j + l + 96] << 6));
    x += kQKK;
  }
}

void dequantize_q2_K(const BlockQ2_K* x, float* y, int64_t k) {
  const int64_t nb = k / kQKK;
  for (int64_t i = 0; i < nb; i++) {
    const float d = fp16_to_fp32(x[i].d);
    const float min = fp16_to_fp32(x[i].dmin);
    const uint8_t* q = x[i].qs;
    int is = 0;
    for (int n = 0; n < kQKK; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = x[i].scales[is++];
        float dl = d * (sc & 0xF), ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<int8_t>((q[l] >> shift) & 3) - ml;
        sc = x[i].scales[is++];
        dl = d * (sc & 0xF);
        ml = min * (sc >> 4);
        for (int l = 0; l < 16; ++l) *y++ = dl * static_cast<int8_t>((q[l + 16] >> shift) & 3) - ml;
        shift += 2;
      }
      q += 32;
    }
  }
}

void quantize_q4_K(const float* x, BlockQ4_K* y, int64_t k) {
  const int64_t nb = k / kQKK;
  uint8_t L[kQKK];
  uint8_t Laux[32];
  float weights[32];
  float mins[kQKK / 32];
  float scales[kQKK / 32];
  for (int64_t i = 0; i < nb; i++) {
    float max_scale = 0, max_min = 0;
    for (int j = 0; j < kQKK / 32; ++j) {
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
      const float av_x = std::sqrt(sum_x2 / 32);
      for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
      scales[j] =
          make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20, false);
      if (scales[j] > max_scale) max_scale = scales[j];
      if (mins[j] > max_min) max_min = mins[j];
    }
    const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < kQKK / 32; ++j) {
      uint8_t ls = static_cast<uint8_t>(nearest_int(inv_scale * scales[j]));
      uint8_t lm = static_cast<uint8_t>(nearest_int(inv_min * mins[j]));
      ls = std::min<uint8_t>(63, ls);
      lm = std::min<uint8_t>(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
        y[i].scales[j - 4] |= static_cast<uint8_t>((ls >> 4) << 6);
        y[i].scales[j - 0] |= static_cast<uint8_t>((lm >> 4) << 6);
      }
    }
    y[i].d = fp32_to_fp16(max_scale / 63.f);
    y[i].dmin = fp32_to_fp16(max_min / 63.f);
    uint8_t sc, m;
    for (int j = 0; j < kQKK / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = fp16_to_fp32(y[i].d) * sc;
      if (!d) continue;
      const float dm = fp16_to_fp32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        const int l = nearest_int((x[32 * j + ii] + dm) / d);
        L[32 * j + ii] = static_cast<uint8_t>(std::max(0, std::min(15, l)));
      }
    }
    uint8_t* q = y[i].qs;
    for (int j = 0; j < kQKK; j += 64) {
      for (int l = 0; l < 32; ++l) q[l] = static_cast<uint8_t>(L[j + l] | (L[j + l + 32] << 4));
      q += 32;
    }
    x += kQKK;
  }
}

void dequantize_q4_K(const BlockQ4_K* x, float* y, int64_t k) {
  const int64_t nb = k / kQKK;
  for (int64_t i = 0; i < nb; i++) {
    const uint8_t* q = x[i].qs;
    const float d = fp16_to_fp32(x[i].d);
    const float min = fp16_to_fp32(x[i].dmin);
    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < kQKK; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc, m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc, m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

}  // namespace

GgufTraits gguf_traits(GgufType t) {
  switch (t) {
    case GgufType::kQ4_0:
      return {kQK, sizeof(BlockQ4_0), "Q4_0"};
    case GgufType::kQ8_0:
      return {kQK, sizeof(BlockQ8_0), "Q8_0"};
    case GgufType::kQ2_K:
      return {kQKK, sizeof(BlockQ2_K), "Q2_K"};
    case GgufType::kQ4_K:
      return {kQKK, sizeof(BlockQ4_K), "Q4_K"};
    case GgufType::kQ5_0:
      return {kQK, sizeof(BlockQ5_0), "Q5_0"};
    case GgufType::kIQ4_NL:
      return {kQK, sizeof(BlockIQ4_NL), "IQ4_NL"};
  }
  return {0, 0, "?"};
}

Result<GgufType> gguf_type_from_name(std::string_view name) {
  std::string s(name);
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "q4_0") return GgufType::kQ4_0;
  if (s == "q8_0") return GgufType::kQ8_0;
  if (s == "q2_k") return GgufType::kQ2_K;
  if (s == "q4_k") return GgufType::kQ4_K;
  if (s == "q5_0") return GgufType::kQ5_0;
  if (s == "iq4_nl") return GgufType::kIQ4_NL;
  return make_error(Errc::kInvalidArgument,
                    "unsupported GGUF type '{}' (q4_0 | q5_0 | q8_0 | iq4_nl | q2_k | q4_k)", name);
}

Result<std::vector<uint8_t>> gguf_quantize(GgufType t, const float* x, int64_t n) {
  const auto tr = gguf_traits(t);
  if (tr.block_elems == 0)
    return make_error(Errc::kInvalidArgument, "unknown GGUF type {}", static_cast<uint32_t>(t));
  if (n <= 0 || n % tr.block_elems != 0)
    return make_error(Errc::kInvalidArgument, "{}: element count {} is not a multiple of {}", tr.name, n,
                      tr.block_elems);
  std::vector<uint8_t> out(n / tr.block_elems * tr.block_bytes);
  switch (t) {
    case GgufType::kQ4_0:
      quantize_q4_0(x, reinterpret_cast<BlockQ4_0*>(out.data()), n);
      break;
    case GgufType::kQ8_0:
      quantize_q8_0(x, reinterpret_cast<BlockQ8_0*>(out.data()), n);
      break;
    case GgufType::kQ2_K:
      quantize_q2_K(x, reinterpret_cast<BlockQ2_K*>(out.data()), n);
      break;
    case GgufType::kQ4_K:
      quantize_q4_K(x, reinterpret_cast<BlockQ4_K*>(out.data()), n);
      break;
    case GgufType::kQ5_0:
      quantize_q5_0(x, reinterpret_cast<BlockQ5_0*>(out.data()), n);
      break;
    case GgufType::kIQ4_NL:
      quantize_iq4_nl(x, reinterpret_cast<BlockIQ4_NL*>(out.data()), n, /*ntry=*/7);
      break;
  }
  return out;
}

Result<void> gguf_dequantize(GgufType t, const uint8_t* data, int64_t n, float* out) {
  const auto tr = gguf_traits(t);
  if (tr.block_elems == 0)
    return make_error(Errc::kInvalidArgument, "unknown GGUF type {}", static_cast<uint32_t>(t));
  if (n <= 0 || n % tr.block_elems != 0)
    return make_error(Errc::kInvalidArgument, "{}: element count {} is not a multiple of {}", tr.name, n,
                      tr.block_elems);
  switch (t) {
    case GgufType::kQ4_0:
      dequantize_q4_0(reinterpret_cast<const BlockQ4_0*>(data), out, n);
      break;
    case GgufType::kQ8_0:
      dequantize_q8_0(reinterpret_cast<const BlockQ8_0*>(data), out, n);
      break;
    case GgufType::kQ2_K:
      dequantize_q2_K(reinterpret_cast<const BlockQ2_K*>(data), out, n);
      break;
    case GgufType::kQ4_K:
      dequantize_q4_K(reinterpret_cast<const BlockQ4_K*>(data), out, n);
      break;
    case GgufType::kQ5_0:
      dequantize_q5_0(reinterpret_cast<const BlockQ5_0*>(data), out, n);
      break;
    case GgufType::kIQ4_NL:
      dequantize_iq4_nl(reinterpret_cast<const BlockIQ4_NL*>(data), out, n);
      break;
  }
  return {};
}

}  // namespace mqde::quant::pack
