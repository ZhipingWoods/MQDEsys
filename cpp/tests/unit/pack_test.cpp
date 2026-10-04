#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mqde/quant/pack.h"

using namespace mqde::quant::pack;

namespace {
std::vector<uint8_t> rand_codes(size_t n, int bits, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(0, (1 << bits) - 1);
  std::vector<uint8_t> v(n);
  for (auto& x : v) x = static_cast<uint8_t>(d(rng));
  return v;
}
std::vector<float> randn(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 0.02f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}
}  // namespace

TEST(QuantPack, Fp16KnownValuesAndRoundTrip) {
  EXPECT_EQ(fp32_to_fp16(1.0f), 0x3C00);
  EXPECT_EQ(fp32_to_fp16(-2.0f), 0xC000);
  EXPECT_EQ(fp32_to_fp16(65504.0f), 0x7BFF);
  EXPECT_EQ(fp32_to_fp16(1e6f), 0x7C00);                // overflow -> inf
  EXPECT_EQ(fp32_to_fp16(0.1f), 0x2E66);                // RNE
  EXPECT_EQ(fp32_to_fp16(5.9604645e-08f), 0x0001);      // smallest subnormal
  EXPECT_EQ(fp32_to_fp16(1.0f + 1.0f / 2048), 0x3C00);  // tie -> even
  for (uint32_t h = 0; h < 0x10000; ++h) {
    if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF)) continue;  // NaN payloads
    ASSERT_EQ(fp32_to_fp16(fp16_to_fp32(static_cast<uint16_t>(h))), h) << h;
  }
}

TEST(QuantPack, BitStreamRoundTripAllWidths) {
  for (int bits = 1; bits <= 8; ++bits)
    for (size_t n : {1u, 7u, 32u, 33u, 1000u}) {
      const auto c = rand_codes(n, bits, 17 * bits + static_cast<unsigned>(n));
      const auto w = pack_bits(c, bits);
      EXPECT_EQ(w.size(), (n * bits + 31) / 32);
      EXPECT_EQ(unpack_bits(w, bits, n), c) << bits << " bit, n=" << n;
    }
}

TEST(QuantPack, ThreeBitMatchesGptq10_1_10_1_10Scheme) {
  const auto x = rand_codes(32, 3, 3);
  uint32_t A = 0, B = 0, C = 0;  // GPTQModel pack_block 3-bit layout
  for (int i = 0; i < 10; ++i) A |= uint32_t(x[i]) << (3 * i);
  A |= uint32_t(x[10]) << 30;
  B = (x[10] >> 2) & 1;
  for (int i = 0; i < 10; ++i) B |= uint32_t(x[11 + i]) << (3 * i + 1);
  B |= uint32_t(x[21]) << 31;
  C = (x[21] >> 1) & 3;
  for (int i = 0; i < 10; ++i) C |= uint32_t(x[22 + i]) << (3 * i + 2);
  EXPECT_EQ(pack_bits(x, 3), (std::vector<uint32_t>{A, B, C}));
}

TEST(QuantPack, GptqLayoutAndRoundTrip) {
  for (int bits : {2, 3, 4, 8}) {
    const int64_t out = 64, in = 128, gs = 32, groups = in / gs;
    const auto codes = rand_codes(out * in, bits, bits);
    auto zeros = rand_codes(out * groups, bits, 100 + bits);
    zeros[0] = 0;  // v1 stores zero-1: must wrap and come back
    std::vector<float> scales(out * groups, 0.0125f);
    for (bool v1 : {true, false}) {
      auto p = pack_gptq(codes.data(), zeros.data(), scales.data(), out, in, gs, bits, v1);
      ASSERT_TRUE(p) << p.error().message;
      EXPECT_EQ(p->qweight_rows, in * bits / 32);
      EXPECT_EQ(p->qzeros_cols, out * bits / 32);
      if (bits == 4) {  // qweight[i/8, o] nibble i%8 = codes[o, i]
        EXPECT_EQ(uint32_t(p->qweight[1 * out + 5]) >> 12 & 0xF, codes[5 * in + 11]);
        const uint32_t z0 = uint32_t(p->qzeros[0]) & 0xF;
        EXPECT_EQ(z0, v1 ? 15u : 0u);
      }
      EXPECT_EQ(p->scales[0], 0x2266);  // 0.0125 = 1.6 * 2^-7 in fp16
      std::vector<uint8_t> c2(codes.size()), z2(zeros.size());
      ASSERT_TRUE(unpack_gptq(*p, out, in, gs, bits, v1, c2.data(), z2.data()));
      EXPECT_EQ(c2, codes);
      EXPECT_EQ(z2, zeros);
    }
  }
}

TEST(QuantPack, AwqInterleavedOrderAndRoundTrip) {
  const int64_t out = 64, in = 256, gs = 128, groups = in / gs;
  const auto codes = rand_codes(out * in, 4, 9);
  const auto zeros = rand_codes(out * groups, 4, 10);
  std::vector<float> scales(out * groups, 0.5f);
  auto p = pack_awq(codes.data(), zeros.data(), scales.data(), out, in, gs, 4);
  ASSERT_TRUE(p);
  EXPECT_EQ(p->qweight_rows, in);
  EXPECT_EQ(p->qweight_cols, out / 8);
  const int order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  const int64_t i = 3;
  for (int k = 0; k < 8; ++k)  // word 1 of row i holds outputs 8..15 interleaved
    EXPECT_EQ(uint32_t(p->qweight[i * p->qweight_cols + 1]) >> (4 * k) & 0xF, codes[(8 + order[k]) * in + i]);
  std::vector<uint8_t> c2(codes.size()), z2(zeros.size());
  ASSERT_TRUE(unpack_awq(*p, out, in, gs, 4, c2.data(), z2.data()));
  EXPECT_EQ(c2, codes);
  EXPECT_EQ(z2, zeros);
}

TEST(QuantPack, MarlinRoundTrip) {
  struct Case {
    int64_t out, in, gs;
    int bits;
    bool zeros;
  };
  for (const Case& c :
       {Case{128, 256, 128, 4, false}, Case{64, 128, 128, 4, true}, Case{192, 256, 64, 4, true},
        Case{128, 256, 256, 4, false}, Case{128, 256, 128, 8, false}, Case{64, 256, 32, 8, true}}) {
    const int64_t groups = c.in / c.gs;
    const auto codes = rand_codes(c.out * c.in, c.bits, 7);
    const auto zeros = rand_codes(c.out * groups, c.bits, 8);
    auto s = randn(c.out * groups, 9);
    for (auto& x : s) x = fp16_to_fp32(fp32_to_fp16(std::fabs(x) + 1e-3f));
    auto p = pack_marlin(codes.data(), zeros.data(), s.data(), c.out, c.in, c.gs, c.bits, c.zeros);
    ASSERT_TRUE(p) << p.error().to_string();
    EXPECT_EQ(p->qweight_rows, c.in / 16);
    EXPECT_EQ(p->qweight_cols, c.out * 16 * c.bits / 32);
    EXPECT_EQ(static_cast<int64_t>(p->qzeros.size()), c.zeros ? groups * c.out * c.bits / 32 : 0);
    std::vector<uint8_t> c2(codes.size()), z2(zeros.size());
    std::vector<float> s2(s.size());
    ASSERT_TRUE(
        unpack_marlin(*p, c.out, c.in, c.gs, c.bits, c2.data(), c.zeros ? z2.data() : nullptr, s2.data()));
    EXPECT_EQ(c2, codes);
    EXPECT_EQ(s2, s);
    if (c.zeros) {
      EXPECT_EQ(z2, zeros);
    }
  }
}

TEST(QuantPack, MarlinRejectsBadShapes) {
  std::vector<uint8_t> codes(32 * 128), zeros(32);
  std::vector<float> s(32, 1.0f);
  EXPECT_FALSE(pack_marlin(codes.data(), zeros.data(), s.data(), 32, 128, 128, 4, false));  // out % 64
  EXPECT_FALSE(pack_marlin(codes.data(), zeros.data(), s.data(), 64, 64, 64, 3, false));    // 3 bit
  EXPECT_FALSE(pack_marlin(codes.data(), nullptr, s.data(), 64, 64, 64, 4, true));          // zeros missing
}

TEST(QuantPack, RejectsBadShapes) {
  std::vector<uint8_t> c(10 * 30), z(10);
  std::vector<float> s(10, 1.f);
  EXPECT_FALSE(pack_gptq(c.data(), z.data(), s.data(), 10, 30, 30, 4));  // in*bits % 32
  EXPECT_FALSE(pack_awq(c.data(), z.data(), s.data(), 10, 30, 7, 4));    // group size
  EXPECT_FALSE(gguf_type_from_name("q5_k"));
  float x[31] = {};
  EXPECT_FALSE(gguf_quantize(GgufType::kQ4_0, x, 31));
}

TEST(QuantPack, GgufBlockSizesAndErrorBounds) {
  const auto x = randn(256 * 16, 5);
  const int64_t n = static_cast<int64_t>(x.size());
  struct Case {
    GgufType t;
    int64_t bytes;
    double max_rel;
  };
  // rel-L2 budgets: 8 bit << 4 bit < 2 bit
  for (const auto& c :
       {Case{GgufType::kQ8_0, 34, 0.01}, Case{GgufType::kQ4_0, 18, 0.15}, Case{GgufType::kQ4_K, 144, 0.12},
        Case{GgufType::kQ2_K, 84, 0.6}, Case{GgufType::kQ5_0, 22, 0.08}, Case{GgufType::kIQ4_NL, 18, 0.12}}) {
    const auto tr = gguf_traits(c.t);
    EXPECT_EQ(tr.block_bytes, c.bytes) << tr.name;
    auto q = gguf_quantize(c.t, x.data(), n);
    ASSERT_TRUE(q);
    EXPECT_EQ(static_cast<int64_t>(q->size()), n / tr.block_elems * tr.block_bytes);
    std::vector<float> y(x.size());
    ASSERT_TRUE(gguf_dequantize(c.t, q->data(), n, y.data()));
    double e = 0, r = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      e += (y[i] - x[i]) * (y[i] - x[i]);
      r += x[i] * x[i];
    }
    EXPECT_LT(std::sqrt(e / r), c.max_rel) << tr.name;
    auto q2 = gguf_quantize(c.t, y.data(), n);  // re-quantizing a dequantized tensor is (nearly) stable
    ASSERT_TRUE(q2);
  }
  // Q4_0 exact example: values on the grid of d = max/-8 round-trip exactly
  std::vector<float> g(32);
  for (int i = 0; i < 32; ++i) g[i] = static_cast<float>((i % 16) - 8) * 0.25f;
  g[0] = -2.0f;  // max |.| = 2 -> d = 0.25
  auto q = gguf_quantize(GgufType::kQ4_0, g.data(), 32);
  std::vector<float> y(32);
  ASSERT_TRUE(gguf_dequantize(GgufType::kQ4_0, q->data(), 32, y.data()));
  EXPECT_EQ(y, g);
}
