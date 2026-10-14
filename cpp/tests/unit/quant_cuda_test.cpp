// T-0406: the CUDA quantizer returns the scalar CPU path's bits. Compiled only with MQDE_WITH_CUDA; skipped
// when no GPU is visible.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mqde/quant/ops.h"
#include "quant_cuda.h"
#include "quant_kernels.h"

using namespace mqde::quant;

namespace {

bool same(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return true;
  return std::memcmp(&a, &b, sizeof a) == 0;
}

QuantSpec spec(int bits, bool sym, Granularity g, int64_t group, CalibAlgo c, int grid = 20,
               float maxshrink = 0.8f) {
  QuantSpec s;
  s.fmt = NumFormat::kInt;
  s.bits = bits;
  s.symmetric = sym;
  s.gran = g;
  s.group_size = group;
  s.calib = c;
  s.grid = grid;
  s.maxshrink = maxshrink;
  return s;
}

std::vector<float> weights(size_t n, unsigned seed, bool awkward) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 0.03f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  if (awkward) {
    const float sp[] = {0.0f,
                        -0.0f,
                        std::numeric_limits<float>::denorm_min(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(),
                        0.5f,
                        2.5f,
                        1.0e30f};
    for (size_t i = 0; i < n; i += 11) v[i] = sp[rng() % (sizeof sp / sizeof *sp)];
  }
  return v;
}

// the CPU reference: the scalar kernels, whatever the host would pick
void cpu(const std::vector<float>& w, int64_t rows, int64_t cols, const QuantSpec& s, std::vector<float>* out,
         QParams* qp) {
  kernels::force(&kernels::scalar());
  out->assign(w.size(), 0.0f);
  ASSERT_TRUE(fake_quant(w.data(), rows, cols, s, out->data(), qp));
  kernels::force(nullptr);
}

}  // namespace

#define NEED_GPU() \
  if (!mqde::quant::cuda::available()) GTEST_SKIP() << "no CUDA device"

TEST(quant_cuda, matches_the_scalar_path_bit_for_bit_for_every_granularity_and_calibration) {
  NEED_GPU();
  const int64_t rows = 70, cols = 256;
  struct Case {
    QuantSpec s;
    bool awkward;
  };
  const std::vector<Case> cases{
      {spec(4, false, Granularity::kPerGroup, 128, CalibAlgo::kMse), false},
      {spec(4, true, Granularity::kPerGroup, 64, CalibAlgo::kMse), false},
      {spec(2, false, Granularity::kPerGroup, 32, CalibAlgo::kMse), false},
      {spec(3, true, Granularity::kPerChannel, 0, CalibAlgo::kMse), false},
      {spec(8, false, Granularity::kPerChannel, 0, CalibAlgo::kMinMax), false},
      {spec(4, false, Granularity::kPerTensor, 0, CalibAlgo::kMse), false},
      {spec(4, true, Granularity::kPerTensor, 0, CalibAlgo::kMinMax), false},
      {spec(4, false, Granularity::kPerGroup, 128, CalibAlgo::kMse, 5, 0.5f), false},
      {spec(4, false, Granularity::kPerGroup, 128, CalibAlgo::kMse),
       true},  // NaN / inf / denormals / +-0 in the groups
      {spec(8, true, Granularity::kPerChannel, 0, CalibAlgo::kMinMax), true},
  };
  for (const auto& c : cases) {
    const auto w = weights(static_cast<size_t>(rows * cols), 17, c.awkward);
    std::vector<float> ref, got(w.size());
    QParams qr, qg;
    cpu(w, rows, cols, c.s, &ref, &qr);
    auto r = mqde::quant::cuda::fake_quant(w.data(), rows, cols, c.s, got.data(), &qg);
    ASSERT_TRUE(r) << describe(c.s) << ": " << r.error().message;
    ASSERT_EQ(qr.scale.size(), qg.scale.size()) << describe(c.s);
    ASSERT_EQ(qr.zero.size(), qg.zero.size()) << describe(c.s);
    for (size_t i = 0; i < qr.scale.size(); ++i) {
      ASSERT_TRUE(same(qr.scale[i], qg.scale[i]))
          << describe(c.s) << " scale " << i << ": " << qr.scale[i] << " vs " << qg.scale[i];
      if (!qr.zero.empty()) {
        ASSERT_TRUE(same(qr.zero[i], qg.zero[i])) << describe(c.s) << " zero " << i;
      }
    }
    for (size_t i = 0; i < w.size(); ++i)
      ASSERT_TRUE(same(ref[i], got[i]))
          << describe(c.s) << (c.awkward ? " (awkward)" : "") << " out " << i << " x=" << w[i];
  }
}

TEST(quant_cuda, one_sided_and_all_zero_groups_match_the_scalar_path) {
  NEED_GPU();
  const int64_t rows = 24, cols = 128;
  for (const auto& sp : {spec(4, false, Granularity::kPerGroup, 32, CalibAlgo::kMse),
                         spec(4, true, Granularity::kPerGroup, 32, CalibAlgo::kMse),
                         spec(8, false, Granularity::kPerChannel, 0, CalibAlgo::kMinMax),
                         spec(2, false, Granularity::kPerGroup, 64, CalibAlgo::kMse)})
    for (float offset : {0.5f, -0.5f, 0.0f}) {
      auto w = weights(static_cast<size_t>(rows * cols), 5, false);
      for (auto& v : w)
        v += offset;  // every group positive-only / negative-only: asymmetric ranges are clamped to include
                      // zero
      for (int64_t r = 0; r < 4; ++r)  // whole rows of zeros: scale 0 must become 1
        for (int64_t c = 0; c < cols; ++c) w[static_cast<size_t>(r * cols + c)] = r % 2 ? -0.0f : 0.0f;
      std::vector<float> ref, got(w.size());
      QParams qr, qg;
      cpu(w, rows, cols, sp, &ref, &qr);
      auto r = mqde::quant::cuda::fake_quant(w.data(), rows, cols, sp, got.data(), &qg);
      ASSERT_TRUE(r) << r.error().message;
      for (size_t i = 0; i < qr.scale.size(); ++i) {
        ASSERT_TRUE(same(qr.scale[i], qg.scale[i])) << describe(sp) << " offset " << offset << " scale " << i;
        if (!qr.zero.empty()) {
          ASSERT_TRUE(same(qr.zero[i], qg.zero[i])) << describe(sp) << " offset " << offset << " zero " << i;
        }
      }
      for (size_t i = 0; i < w.size(); ++i)
        ASSERT_TRUE(same(ref[i], got[i])) << describe(sp) << " offset " << offset << " out " << i;
    }
}

namespace {
// The MSE winner of one asymmetric group, with the squared error summed in T. Local copy of the candidate
// parameters (ops.cpp params_from_range): this is only used to FIND groups on which a float sum and a double
// sum pick different winners.
template <class T>
int winner(const float* x, int n, const QuantSpec& s) {
  float mn = INFINITY, mx = -INFINITY;
  for (int i = 0; i < n; ++i) {
    mn = std::min(mn, x[i]);
    mx = std::max(mx, x[i]);
  }
  const float qmin = static_cast<float>(s.qmin()), qmax = static_cast<float>(s.qmax());
  std::vector<float> q(static_cast<size_t>(n));
  T best_err = std::numeric_limits<T>::infinity();
  int best = 0;
  for (int i = 0; i <= s.grid; ++i) {
    const float shrink = 1.0f - static_cast<float>(i) * (1.0f - s.maxshrink) / static_cast<float>(s.grid);
    float lo = std::min(mn * shrink, 0.0f), hi = std::max(mx * shrink, 0.0f);
    float scale = (hi - lo) / (qmax - qmin);
    if (!(scale > 0.0f)) scale = 1.0f;
    const float zero = std::clamp(std::nearbyint(qmin - lo / scale), qmin, qmax);
    const kernels::IntQdq p{scale, zero, qmin, qmax};
    kernels::scalar().qdq(x, n, p, q.data());
    T err = 0;
    for (int j = 0; j < n; ++j) {
      const T d = static_cast<T>(static_cast<double>(q[static_cast<size_t>(j)]) - x[j]);
      err += d * d;
    }
    if (err < best_err) {
      best_err = err;
      best = i;
    }
  }
  return best;
}
}  // namespace

TEST(quant_cuda, groups_where_a_float_error_sum_would_pick_another_scale_still_match) {
  NEED_GPU();
  // the squared error must be summed in double, in element order: only a near-tie between two candidates
  // shows it
  const int n = 32;
  // a fine grid puts neighbouring candidates within a few float ulps of each other in error, which is where
  // the sums disagree
  const auto sp = spec(4, false, Granularity::kPerGroup, n, CalibAlgo::kMse, 1000);
  std::mt19937 rng(99);
  std::normal_distribution<float> d(0.0f, 0.02f);
  std::vector<float> w;
  std::vector<float> g(n);
  for (int tries = 0; tries < 4000 && w.size() < static_cast<size_t>(n) * 64; ++tries) {
    for (auto& v : g) v = d(rng);
    if (winner<double>(g.data(), n, sp) != winner<float>(g.data(), n, sp))
      w.insert(w.end(), g.begin(), g.end());
  }
  ASSERT_GE(w.size(), static_cast<size_t>(n) * 2)
      << "found no group on which float and double sums disagree: the search is too small";
  const int64_t rows = static_cast<int64_t>(w.size()) / n;
  std::vector<float> ref, got(w.size());
  QParams qr, qg;
  cpu(w, rows, n, sp, &ref, &qr);
  ASSERT_TRUE(mqde::quant::cuda::fake_quant(w.data(), rows, n, sp, got.data(), &qg));
  for (size_t i = 0; i < qr.scale.size(); ++i)
    ASSERT_TRUE(same(qr.scale[i], qg.scale[i])) << "group " << i << " of " << rows;
  for (size_t i = 0; i < w.size(); ++i) ASSERT_TRUE(same(ref[i], got[i])) << i;
}

namespace {
QuantSpec fp8_spec(NumFormat f, Granularity g, int64_t group, CalibAlgo c) {
  QuantSpec s;
  s.fmt = f;
  s.bits = 8;
  s.gran = g;
  s.group_size = group;
  s.calib = c;
  return s;
}
}  // namespace

TEST(quant_cuda, fp8_matches_the_scalar_path_bit_for_bit) {
  NEED_GPU();
  const int64_t rows = 40, cols = 192;
  for (NumFormat f : {NumFormat::kFp8E4M3, NumFormat::kFp8E5M2})
    for (auto [g, group] : {std::pair{Granularity::kPerChannel, int64_t{0}},
                            {Granularity::kPerGroup, int64_t{64}},
                            {Granularity::kPerTensor, int64_t{0}}})
      for (CalibAlgo c : {CalibAlgo::kMinMax, CalibAlgo::kMse})
        for (bool awkward : {false, true}) {
          const auto sp = fp8_spec(f, g, group, c);
          auto w = weights(static_cast<size_t>(rows * cols), 23, awkward);
          // the range of FP8 is where the rounding differs from integers: subnormals, ties, saturation
          const float fm = sp.fp8_max();
          const float vals[] = {fm,      -fm,     fm * 1.01f, -fm * 3.0f, 0.001953125f, 0.0009765625f,
                                1.0625f, 1.1875f, 17.0f,      18.0f,      1e-8f};
          for (size_t i = 3; i < w.size(); i += 17)
            w[i] = vals[i % (sizeof vals / sizeof *vals)] * (c == CalibAlgo::kMse ? 1.0f : 0.01f);
          std::vector<float> ref, got(w.size());
          QParams qr, qg;
          cpu(w, rows, cols, sp, &ref, &qr);
          auto r = mqde::quant::cuda::fake_quant(w.data(), rows, cols, sp, got.data(), &qg);
          ASSERT_TRUE(r) << describe(sp) << ": " << r.error().message;
          ASSERT_EQ(qr.scale.size(), qg.scale.size());
          EXPECT_TRUE(qg.zero.empty()) << "FP8 has no zero point";
          for (size_t i = 0; i < qr.scale.size(); ++i)
            ASSERT_TRUE(same(qr.scale[i], qg.scale[i]))
                << describe(sp) << (awkward ? " awkward" : "") << " scale " << i;
          for (size_t i = 0; i < w.size(); ++i)
            ASSERT_TRUE(same(ref[i], got[i])) << describe(sp) << (awkward ? " awkward" : "") << " out " << i
                                              << " x=" << w[i] << ": " << ref[i] << " vs " << got[i];
        }
}

TEST(quant_cuda, works_in_place_and_a_group_of_only_nan_gets_scale_one) {
  NEED_GPU();
  const int64_t rows = 4, cols = 64;
  auto w = weights(static_cast<size_t>(rows * cols), 3, false);
  for (int64_t c = 0; c < 32; ++c)
    w[static_cast<size_t>(c)] = std::numeric_limits<float>::quiet_NaN();  // row 0, group 0
  const auto s = spec(4, false, Granularity::kPerGroup, 32, CalibAlgo::kMse);
  std::vector<float> ref;
  QParams qr, qg;
  cpu(w, rows, cols, s, &ref, &qr);
  auto in_place = w;
  ASSERT_TRUE(mqde::quant::cuda::fake_quant(in_place.data(), rows, cols, s, in_place.data(), &qg));
  EXPECT_EQ(qg.scale[0], 1.0f);
  EXPECT_EQ(qg.zero[0], 0.0f);
  for (size_t i = 0; i < w.size(); ++i) ASSERT_TRUE(same(ref[i], in_place[i])) << i;
}

TEST(quant_cuda, bad_input_is_an_error_not_a_crash) {
  NEED_GPU();
  std::vector<float> w(10, 1.0f), out(10);
  EXPECT_FALSE(mqde::quant::cuda::fake_quant(
      w.data(), 1, 10, spec(4, false, Granularity::kPerGroup, 4, CalibAlgo::kMinMax), out.data()));
  EXPECT_FALSE(mqde::quant::cuda::fake_quant(
      w.data(), 0, 10, spec(4, false, Granularity::kPerChannel, 0, CalibAlgo::kMinMax), out.data()));
  QuantSpec off;
  EXPECT_FALSE(mqde::quant::cuda::fake_quant(w.data(), 1, 10, off, out.data()));
}
