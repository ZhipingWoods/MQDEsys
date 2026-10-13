// T-0406: the SIMD quantizer kernels return the scalar kernels' bits, on awkward data as well as on weights.
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mqde/quant/ops.h"
#include "quant_kernels.h"

using namespace mqde::quant;
namespace k = mqde::quant::kernels;

namespace {

// bit-equal, with every NaN equal to every other NaN (the payload of a propagated NaN is not part of the
// contract)
bool same(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return true;
  return std::memcmp(&a, &b, sizeof a) == 0;
}
bool same(double a, double b) {
  if (std::isnan(a) && std::isnan(b)) return true;
  return std::memcmp(&a, &b, sizeof a) == 0;
}

std::vector<const k::Set*> simd_sets() {
  std::vector<const k::Set*> v;
  for (const k::Set* s : k::available())
    if (std::string_view(s->name) != "scalar") v.push_back(s);
  return v;
}

// weights-like values plus the values that break naive vector code
std::vector<float> awkward(size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 0.05f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  const float specials[] = {0.0f,
                            -0.0f,
                            std::numeric_limits<float>::denorm_min(),
                            -std::numeric_limits<float>::denorm_min(),
                            std::numeric_limits<float>::min(),
                            std::numeric_limits<float>::max(),
                            -std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN(),
                            0.5f,
                            1.5f,
                            2.5f,
                            -0.5f,
                            -2.5f,
                            7.5f};
  for (size_t i = 0; i < n; i += 3) v[i] = specials[rng() % (sizeof specials / sizeof *specials)];
  return v;
}

std::vector<k::IntQdq> params() {
  std::vector<k::IntQdq> v;
  for (float scale : {0.01f, 0.5f, 1.0f, 3.7f, 1e-30f, 1e30f, std::numeric_limits<float>::denorm_min()})
    for (auto [qmin, qmax, zero] : {std::tuple{-8.0f, 7.0f, 0.0f},
                                    {0.0f, 15.0f, 8.0f},
                                    {0.0f, 3.0f, 1.0f},
                                    {-128.0f, 127.0f, 0.0f},
                                    {0.0f, 255.0f, 128.0f},
                                    {0.0f, 3.0f, 2.0f}})
      v.push_back({scale, zero, qmin, qmax});
  return v;
}

const size_t kLengths[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 63, 127, 128, 129, 1000};

}  // namespace

TEST(quant_kernels, the_scalar_set_is_listed_first_and_found_by_name) {
  const auto all = k::available();
  ASSERT_FALSE(all.empty());
  EXPECT_STREQ(all.front()->name, "scalar");
  EXPECT_EQ(k::find("scalar"), &k::scalar());
  EXPECT_EQ(k::find("no-such-set"), nullptr);
}

TEST(quant_kernels, force_overrides_the_default_and_nullptr_restores_it) {
  const k::Set* dflt = &k::active();
  k::force(&k::scalar());
  EXPECT_EQ(&k::active(), &k::scalar());
  k::force(nullptr);
  EXPECT_EQ(&k::active(), dflt);
}

TEST(quant_kernels, minmax_matches_scalar_bit_for_bit) {
  for (const k::Set* s : simd_sets())
    for (size_t n : kLengths)
      for (unsigned seed = 0; seed < 6; ++seed) {
        const auto x = awkward(n, seed);
        float mn0 = std::numeric_limits<float>::infinity(), mx0 = -mn0, mn1 = mn0, mx1 = mx0;
        k::scalar().minmax(x.data(), static_cast<int64_t>(n), &mn0, &mx0);
        s->minmax(x.data(), static_cast<int64_t>(n), &mn1, &mx1);
        // +0 and -0 compare equal, and the first one met wins in the scalar loop, so the SIMD order may pick
        // the other; that cannot reach the scale or the zero point (checked end to end below), so compare by
        // value here
        EXPECT_TRUE(mn0 == mn1 && mx0 == mx1) << s->name << " n=" << n << " seed=" << seed << ": " << mn0
                                              << "," << mx0 << " vs " << mn1 << "," << mx1;
      }
}

TEST(quant_kernels, minmax_ignores_nan_and_keeps_the_starting_values) {
  for (const k::Set* s : simd_sets()) {
    std::vector<float> x(37, std::numeric_limits<float>::quiet_NaN());
    float mn = -3.0f, mx = 4.0f;
    s->minmax(x.data(), static_cast<int64_t>(x.size()), &mn, &mx);
    EXPECT_EQ(mn, -3.0f) << s->name;
    EXPECT_EQ(mx, 4.0f) << s->name;
    x[20] = 9.0f;
    x[3] = -11.0f;
    s->minmax(x.data(), static_cast<int64_t>(x.size()), &mn, &mx);
    EXPECT_EQ(mn, -11.0f) << s->name;
    EXPECT_EQ(mx, 9.0f) << s->name;
  }
}

TEST(quant_kernels, squared_error_matches_scalar_bit_for_bit) {
  for (const k::Set* s : simd_sets())
    for (const auto& p : params())
      for (size_t n : kLengths) {
        const auto x = awkward(n, 11);
        const double a = k::scalar().sq_err(x.data(), static_cast<int64_t>(n), p, 0.25);
        const double b = s->sq_err(x.data(), static_cast<int64_t>(n), p, 0.25);
        ASSERT_TRUE(same(a, b)) << s->name << " n=" << n << " scale=" << p.scale << " q=[" << p.qmin << ","
                                << p.qmax << "] zero=" << p.zero << ": " << a << " vs " << b;
      }
}

TEST(quant_kernels, squared_error_is_summed_in_element_order_not_per_lane) {
  // 1e16 + 1 + 1 + ... only comes out right in one order; weights of this size make lane-wise partial sums
  // differ
  for (const k::Set* s : simd_sets()) {
    std::vector<float> x(64);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 8 == 0) ? 1.0e8f : 0.3f;
    const k::IntQdq p{1.0f, 0.0f, -8.0f, 7.0f};
    const double a = k::scalar().sq_err(x.data(), 64, p, 0.0);
    const double b = s->sq_err(x.data(), 64, p, 0.0);
    EXPECT_TRUE(same(a, b)) << s->name << ": " << a << " vs " << b;
  }
}

TEST(quant_kernels, qdq_matches_scalar_bit_for_bit_and_in_place) {
  for (const k::Set* s : simd_sets())
    for (const auto& p : params())
      for (size_t n : kLengths) {
        const auto x = awkward(n, 5);
        std::vector<float> a(n), b(n), c = x;
        k::scalar().qdq(x.data(), static_cast<int64_t>(n), p, a.data());
        s->qdq(x.data(), static_cast<int64_t>(n), p, b.data());
        s->qdq(c.data(), static_cast<int64_t>(n), p, c.data());
        for (size_t i = 0; i < n; ++i) {
          ASSERT_TRUE(same(a[i], b[i])) << s->name << " i=" << i << " x=" << x[i] << " scale=" << p.scale
                                        << ": " << a[i] << " vs " << b[i];
          ASSERT_TRUE(same(a[i], c[i])) << s->name << " in place, i=" << i;
        }
      }
}

std::vector<float> finite_awkward(size_t n, unsigned seed) {
  auto v = awkward(n, seed);
  for (auto& x : v)
    if (!std::isfinite(x))
      x = (x > 0 ? 1.0f : -1.0f) * 1.0e6f;  // far outside every range: clamps, stays finite
  return v;
}

TEST(quant_kernels, quantize_matches_scalar_codes_and_stays_in_range) {
  for (const k::Set* s : simd_sets())
    for (const auto& p : params())
      for (size_t n : kLengths) {
        const auto x = finite_awkward(n, 21);
        std::vector<int16_t> a(n), b(n);
        k::scalar().quantize(x.data(), static_cast<int64_t>(n), p, a.data());
        s->quantize(x.data(), static_cast<int64_t>(n), p, b.data());
        for (size_t i = 0; i < n; ++i) {
          ASSERT_EQ(a[i], b[i]) << s->name << " i=" << i << " x=" << x[i] << " scale=" << p.scale;
          ASSERT_GE(b[i], static_cast<int16_t>(p.qmin));
          ASSERT_LE(b[i], static_cast<int16_t>(p.qmax));
        }
      }
}

TEST(quant_kernels, dequantize_matches_scalar_bit_for_bit_over_the_whole_code_range) {
  for (const k::Set* s : simd_sets())
    for (const auto& p : params())
      for (size_t n : kLengths) {
        std::vector<int16_t> codes(n);
        for (size_t i = 0; i < n; ++i)
          codes[i] = static_cast<int16_t>(static_cast<int>(p.qmin) + static_cast<int>(i * 7 % 300));
        std::vector<float> a(n), b(n);
        k::scalar().dequantize(codes.data(), static_cast<int64_t>(n), p, a.data());
        s->dequantize(codes.data(), static_cast<int64_t>(n), p, b.data());
        for (size_t i = 0; i < n; ++i)
          ASSERT_TRUE(same(a[i], b[i])) << s->name << " i=" << i << " code=" << codes[i];
      }
}

TEST(quant_kernels, quantize_then_dequantize_equals_qdq_under_every_set) {
  for (const k::Set* s : k::available())
    for (const auto& p : params()) {
      const auto x = finite_awkward(129, 4);
      std::vector<int16_t> codes(x.size());
      std::vector<float> a(x.size()), b(x.size());
      s->quantize(x.data(), 129, p, codes.data());
      s->dequantize(codes.data(), 129, p, a.data());
      s->qdq(x.data(), 129, p, b.data());
      for (size_t i = 0; i < x.size(); ++i) ASSERT_TRUE(same(a[i], b[i])) << s->name << " i=" << i;
    }
}

TEST(quant_kernels, ties_round_to_even_like_the_scalar) {
  for (const k::Set* s : simd_sets()) {
    const std::vector<float> x{0.5f, 1.5f,        2.5f, 3.5f,  -0.5f, -1.5f, -2.5f, 6.5f,
                               7.5f, 0.49999997f, 8.5f, -8.5f, 1,     1,     1,     1};
    const k::IntQdq p{1.0f, 0.0f, -8.0f, 7.0f};
    std::vector<float> a(x.size()), b(x.size());
    k::scalar().qdq(x.data(), 16, p, a.data());
    s->qdq(x.data(), 16, p, b.data());
    for (size_t i = 0; i < x.size(); ++i) EXPECT_TRUE(same(a[i], b[i])) << s->name << " x=" << x[i];
    EXPECT_EQ(b[0], 0.0f);  // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2
    EXPECT_EQ(b[1], 2.0f);
    EXPECT_EQ(b[2], 2.0f);
  }
}

namespace {
QuantSpec spec(int bits, bool sym, Granularity g, int64_t group, CalibAlgo c) {
  QuantSpec s;
  s.fmt = NumFormat::kInt;
  s.bits = bits;
  s.symmetric = sym;
  s.gran = g;
  s.group_size = group;
  s.calib = c;
  return s;
}
}  // namespace

TEST(quant_kernels, compute_qparams_and_fake_quant_are_identical_under_every_set) {
  const int64_t rows = 37, cols = 256;  // 37 rows: no multiple of anything
  std::mt19937 rng(3);
  std::normal_distribution<float> d(0.0f, 0.02f);
  std::vector<float> w(static_cast<size_t>(rows * cols));
  for (auto& v : w) v = d(rng);
  w[5] = 0.0f;
  w[6] = -0.0f;
  w[300] = 3.0f;  // an outlier
  const std::vector<QuantSpec> specs{spec(4, false, Granularity::kPerGroup, 128, CalibAlgo::kMse),
                                     spec(4, true, Granularity::kPerGroup, 64, CalibAlgo::kMse),
                                     spec(2, false, Granularity::kPerGroup, 32, CalibAlgo::kMse),
                                     spec(8, true, Granularity::kPerChannel, 0, CalibAlgo::kMse),
                                     spec(4, false, Granularity::kPerTensor, 0, CalibAlgo::kMse),
                                     spec(4, false, Granularity::kPerGroup, 128, CalibAlgo::kMinMax),
                                     spec(3, true, Granularity::kPerChannel, 0, CalibAlgo::kMinMax)};
  for (const k::Set* s : simd_sets())
    for (const auto& sp : specs) {
      std::vector<float> o0(w.size()), o1(w.size());
      QParams q0, q1;
      k::force(&k::scalar());
      ASSERT_TRUE(fake_quant(w.data(), rows, cols, sp, o0.data(), &q0));
      k::force(s);
      ASSERT_TRUE(fake_quant(w.data(), rows, cols, sp, o1.data(), &q1));
      k::force(nullptr);
      ASSERT_EQ(q0.scale.size(), q1.scale.size());
      for (size_t i = 0; i < q0.scale.size(); ++i) {
        ASSERT_TRUE(same(q0.scale[i], q1.scale[i])) << s->name << " " << describe(sp) << " scale " << i;
        if (!q0.zero.empty()) {
          ASSERT_TRUE(same(q0.zero[i], q1.zero[i])) << s->name << " " << describe(sp) << " zero " << i;
        }
      }
      for (size_t i = 0; i < w.size(); ++i)
        ASSERT_TRUE(same(o0[i], o1[i])) << s->name << " " << describe(sp) << " out " << i;
    }
}

TEST(quant_kernels, quantize_int_and_dequantize_int_are_identical_under_every_set) {
  const int64_t rows = 19, cols = 192;
  std::mt19937 rng(8);
  std::normal_distribution<float> d(0.0f, 0.03f);
  std::vector<float> w(static_cast<size_t>(rows * cols));
  for (auto& v : w) v = d(rng);
  for (const auto& sp : {spec(4, false, Granularity::kPerGroup, 64, CalibAlgo::kMse),
                         spec(8, true, Granularity::kPerChannel, 0, CalibAlgo::kMinMax),
                         spec(2, false, Granularity::kPerTensor, 0, CalibAlgo::kMinMax)}) {
    k::force(&k::scalar());
    auto qp = compute_qparams(w.data(), rows, cols, sp);
    ASSERT_TRUE(qp);
    std::vector<int16_t> c0(w.size());
    std::vector<float> d0(w.size());
    ASSERT_TRUE(quantize_int(w.data(), rows, cols, sp, *qp, c0.data()));
    ASSERT_TRUE(dequantize_int(c0.data(), rows, cols, sp, *qp, d0.data()));
    for (const k::Set* s : simd_sets()) {
      k::force(s);
      std::vector<int16_t> c1(w.size());
      std::vector<float> d1(w.size());
      ASSERT_TRUE(quantize_int(w.data(), rows, cols, sp, *qp, c1.data()));
      ASSERT_TRUE(dequantize_int(c1.data(), rows, cols, sp, *qp, d1.data()));
      EXPECT_EQ(c0, c1) << s->name << " " << describe(sp);
      for (size_t i = 0; i < w.size(); ++i)
        ASSERT_TRUE(same(d0[i], d1[i])) << s->name << " " << describe(sp) << " i=" << i;
    }
    k::force(nullptr);
  }
}

TEST(quant_kernels, fp8_is_untouched_by_the_choice_of_set) {
  const int64_t rows = 8, cols = 64;
  std::mt19937 rng(9);
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> w(static_cast<size_t>(rows * cols));
  for (auto& v : w) v = d(rng);
  QuantSpec sp;
  sp.fmt = NumFormat::kFp8E4M3;
  sp.bits = 8;
  sp.gran = Granularity::kPerChannel;
  sp.calib = CalibAlgo::kMse;
  for (const k::Set* s : simd_sets()) {
    std::vector<float> a(w.size()), b(w.size());
    k::force(&k::scalar());
    ASSERT_TRUE(fake_quant(w.data(), rows, cols, sp, a.data()));
    k::force(s);
    ASSERT_TRUE(fake_quant(w.data(), rows, cols, sp, b.data()));
    k::force(nullptr);
    for (size_t i = 0; i < w.size(); ++i) ASSERT_TRUE(same(a[i], b[i])) << s->name << " i=" << i;
  }
}
