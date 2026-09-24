#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mqde/quant/metrics.h"
#include "mqde/quant/ops.h"

using namespace mqde::quant;

namespace {
QuantSpec int_spec(int bits, bool sym, Granularity g, int64_t group = 0, CalibAlgo calib = CalibAlgo::kMinMax) {
  QuantSpec s;
  s.fmt = NumFormat::kInt;
  s.bits = bits;
  s.symmetric = sym;
  s.gran = g;
  s.group_size = group;
  s.calib = calib;
  return s;
}
std::vector<float> randn(size_t n, unsigned seed, float scale = 1.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}
}  // namespace

TEST(quant_fp8, e4m3_rounding) {
  EXPECT_FLOAT_EQ(fp8_round(0.1f, NumFormat::kFp8E4M3), 0.1015625f);
  EXPECT_FLOAT_EQ(fp8_round(448.0f, NumFormat::kFp8E4M3), 448.0f);
  EXPECT_FLOAT_EQ(fp8_round(1000.0f, NumFormat::kFp8E4M3), 448.0f) << "saturating";
  EXPECT_FLOAT_EQ(fp8_round(-1000.0f, NumFormat::kFp8E4M3), -448.0f);
  EXPECT_FLOAT_EQ(fp8_round(1.0625f, NumFormat::kFp8E4M3), 1.0f) << "ties to even";
  EXPECT_FLOAT_EQ(fp8_round(1.1875f, NumFormat::kFp8E4M3), 1.25f) << "ties to even (odd neighbour)";
  EXPECT_FLOAT_EQ(fp8_round(0.001f, NumFormat::kFp8E4M3), 0.001953125f) << "smallest subnormal";
  EXPECT_FLOAT_EQ(fp8_round(0.0f, NumFormat::kFp8E4M3), 0.0f);
}

TEST(quant_fp8, e5m2_rounding) {
  EXPECT_FLOAT_EQ(fp8_round(57344.0f, NumFormat::kFp8E5M2), 57344.0f);
  EXPECT_FLOAT_EQ(fp8_round(1.1f, NumFormat::kFp8E5M2), 1.0f);
  EXPECT_FLOAT_EQ(fp8_round(1.2f, NumFormat::kFp8E5M2), 1.25f);
}

TEST(quant_int, sym_per_channel_hand_computed) {
  // absmax 7 -> scale exactly 1.0, so x/scale is exact and rounding ties are real ties.
  std::vector<float> x = {-7.0f, 3.5f, 2.5f, 7.0f};
  std::vector<float> out(4);
  QParams qp;
  ASSERT_TRUE(fake_quant(x.data(), 1, 4, int_spec(4, true, Granularity::kPerChannel), out.data(), &qp));
  EXPECT_FLOAT_EQ(qp.scale[0], 1.0f);
  EXPECT_EQ(out, (std::vector<float>{-7.0f, 4.0f, 2.0f, 7.0f})) << "round-half-to-even: 3.5->4, 2.5->2";
}

TEST(quant_int, sym_int4_range_is_minus8_to_7) {
  std::vector<float> x = {-8.0f, 7.0f};  // absmax 8 -> scale 8/7; -8/(8/7) = -7 exactly representable
  QParams qp;
  std::vector<float> out(2);
  ASSERT_TRUE(fake_quant(x.data(), 1, 2, int_spec(4, true, Granularity::kPerTensor), out.data(), &qp));
  const auto s = int_spec(4, true, Granularity::kPerTensor);
  EXPECT_EQ(s.qmin(), -8);
  EXPECT_EQ(s.qmax(), 7);
}

TEST(quant_int, asym_zero_point) {
  std::vector<float> x = {0.0f, 1.0f, 2.0f, 3.0f};  // all non-negative -> min clamps to 0, zero = 0
  QParams qp;
  std::vector<float> out(4);
  ASSERT_TRUE(fake_quant(x.data(), 1, 4, int_spec(2, false, Granularity::kPerChannel), out.data(), &qp));
  EXPECT_FLOAT_EQ(qp.scale[0], 1.0f);
  EXPECT_FLOAT_EQ(qp.zero[0], 0.0f);
  EXPECT_EQ(out, x) << "2-bit asym represents 0..3 exactly";
}

TEST(quant_int, per_group_requires_divisible) {
  std::vector<float> x(3 * 100, 1.0f);
  auto r = compute_qparams(x.data(), 3, 100, int_spec(4, true, Granularity::kPerGroup, 64));
  ASSERT_FALSE(r);
  EXPECT_EQ(r.error().code, mqde::Errc::kInvalidArgument);
}

TEST(quant_int, per_group_layout) {
  auto x = randn(8 * 256, 1);
  auto r = compute_qparams(x.data(), 8, 256, int_spec(4, false, Granularity::kPerGroup, 64));
  ASSERT_TRUE(r);
  EXPECT_EQ(r->groups_per_row, 4);
  EXPECT_EQ(r->scale.size(), 32u);
  EXPECT_EQ(r->zero.size(), 32u);
}

TEST(quant_int, real_path_matches_fake_path_bitwise) {
  for (bool sym : {true, false}) {
    for (int bits : {2, 4, 8}) {
      const auto spec = int_spec(bits, sym, Granularity::kPerGroup, 32, CalibAlgo::kMse);
      auto x = randn(16 * 128, 7 + bits);
      std::vector<float> fake(x.size()), real(x.size());
      std::vector<int16_t> codes(x.size());
      QParams qp;
      ASSERT_TRUE(fake_quant(x.data(), 16, 128, spec, fake.data(), &qp));
      ASSERT_TRUE(quantize_int(x.data(), 16, 128, spec, qp, codes.data()));
      ASSERT_TRUE(dequantize_int(codes.data(), 16, 128, spec, qp, real.data()));
      for (size_t i = 0; i < x.size(); ++i) {
        ASSERT_EQ(fake[i], real[i]) << "bits=" << bits << " sym=" << sym << " i=" << i;
        ASSERT_GE(codes[i], spec.qmin());
        ASSERT_LE(codes[i], spec.qmax());
      }
    }
  }
}

TEST(quant_calib, mse_beats_minmax_with_outliers) {
  auto x = randn(4 * 512, 3, 0.02f);
  for (int r = 0; r < 4; ++r) x[static_cast<size_t>(r * 512 + 7)] = 1.0f;  // one large outlier per row
  std::vector<float> a(x.size()), b(x.size());
  ASSERT_TRUE(fake_quant(x.data(), 4, 512, int_spec(4, true, Granularity::kPerChannel), a.data()));
  ASSERT_TRUE(fake_quant(x.data(), 4, 512, int_spec(4, true, Granularity::kPerChannel, 0, CalibAlgo::kMse), b.data()));
  const auto ma = measure(x.data(), a.data(), static_cast<int64_t>(x.size()));
  const auto mb = measure(x.data(), b.data(), static_cast<int64_t>(x.size()));
  EXPECT_LE(mb.mse, ma.mse);
}

TEST(quant_int, more_bits_less_error) {
  auto x = randn(32 * 128, 11);
  double prev = 1e30;
  for (int bits : {2, 3, 4, 8}) {
    std::vector<float> o(x.size());
    ASSERT_TRUE(fake_quant(x.data(), 32, 128, int_spec(bits, false, Granularity::kPerGroup, 64), o.data()));
    const auto m = measure(x.data(), o.data(), static_cast<int64_t>(x.size()));
    EXPECT_LT(m.mse, prev) << bits;
    prev = m.mse;
  }
}

TEST(quant_fp8, fake_quant_per_channel) {
  QuantSpec s;
  s.fmt = NumFormat::kFp8E4M3;
  s.bits = 8;
  s.gran = Granularity::kPerChannel;
  auto x = randn(4 * 64, 5);
  std::vector<float> o(x.size());
  QParams qp;
  ASSERT_TRUE(fake_quant(x.data(), 4, 64, s, o.data(), &qp));
  const auto m = measure(x.data(), o.data(), static_cast<int64_t>(x.size()));
  EXPECT_GT(m.sqnr_db, 25.0);  // ~3 mantissa bits -> ~30 dB on Gaussian data
  EXPECT_GT(m.cosine, 0.999);
}

TEST(quant_metrics, basics) {
  std::vector<float> a = {1, 2, 3, 4}, b = {1, 2, 3, 4};
  auto m = measure(a.data(), b.data(), 4);
  EXPECT_EQ(m.mse, 0.0);
  EXPECT_TRUE(std::isinf(m.sqnr_db));
  EXPECT_DOUBLE_EQ(m.cosine, 1.0);
  b[3] = 5;
  m = measure(a.data(), b.data(), 4);
  EXPECT_DOUBLE_EQ(m.mse, 0.25);
  EXPECT_DOUBLE_EQ(m.max_abs, 1.0);
  auto s = tensor_stats(a.data(), 4);
  EXPECT_DOUBLE_EQ(s.mean, 2.5);
  EXPECT_DOUBLE_EQ(s.absmax, 4.0);
}
