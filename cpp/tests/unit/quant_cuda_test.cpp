// T-0406: the CUDA quantizer returns the scalar CPU path's bits. Compiled only with MQDE_WITH_CUDA; skipped
// when no GPU is visible.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mqde/quant/ops.h"
#include "mqde/quant/pack.h"
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

TEST(quant_cuda, pack_bits_matches_the_cpu_packer_for_every_width_and_awkward_lengths) {
  NEED_GPU();
  namespace pack = mqde::quant::pack;
  std::mt19937 rng(77);
  for (int bits = 1; bits <= 8; ++bits)
    for (size_t n : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{10}, size_t{31}, size_t{32},
                     size_t{33}, size_t{63}, size_t{64}, size_t{65}, size_t{1000}, size_t{100003}}) {
      std::vector<uint8_t> codes(n);
      for (auto& c : codes) c = static_cast<uint8_t>(rng());  // out of range for bits < 8: both must mask
      const auto ref = pack::pack_bits(codes, bits);
      auto got = mqde::quant::cuda::pack_bits(codes, bits);
      ASSERT_TRUE(got) << got.error().message;
      ASSERT_EQ(ref, *got) << bits << " bits, n=" << n;
      auto back = mqde::quant::cuda::unpack_bits(ref, bits, n);
      ASSERT_TRUE(back) << back.error().message;
      ASSERT_EQ(pack::unpack_bits(ref, bits, n), *back) << "unpack, " << bits << " bits, n=" << n;
    }
}

TEST(quant_cuda, pack_bits_rejects_bad_arguments) {
  NEED_GPU();
  std::vector<uint8_t> codes(8, 1);
  EXPECT_FALSE(mqde::quant::cuda::pack_bits(codes, 0));
  EXPECT_FALSE(mqde::quant::cuda::pack_bits(codes, 9));
  EXPECT_FALSE(mqde::quant::cuda::unpack_bits(std::vector<uint32_t>(1), 4, 9))
      << "one word holds 8 four-bit codes, not 9";
  EXPECT_TRUE(mqde::quant::cuda::unpack_bits(std::vector<uint32_t>(1), 4, 8));
}

TEST(quant_cuda, pack_bits_indexes_past_32_bits) {
  NEED_GPU();
  // 8-bit codes i % 251: word w is the four bytes 4w..4w+3, so every word is known without a CPU reference.
  // 600M codes put the last bit offset (4.8e9) above 2^32. The period is prime on purpose: a power of two
  // would repeat with the 2^32-bit wrap-around itself, and an offset truncated to 32 bits would read the same
  // values and go unnoticed.
  const size_t n = 600u * 1000 * 1000;
  std::vector<uint8_t> codes(n);
  for (size_t i = 0; i < n; ++i) codes[i] = static_cast<uint8_t>(i % 251);
  auto got = mqde::quant::cuda::pack_bits(codes, 8);
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->size(), n / 4);
  size_t bad = 0;
  for (size_t w = 0; w < got->size(); ++w) {
    const uint32_t b0 = (4 * w) % 251, b1 = (4 * w + 1) % 251, b2 = (4 * w + 2) % 251, b3 = (4 * w + 3) % 251;
    bad += (*got)[w] != (b0 | b1 << 8 | b2 << 16 | b3 << 24);
  }
  EXPECT_EQ(bad, 0u);
  auto back = mqde::quant::cuda::unpack_bits(*got, 8, n);
  ASSERT_TRUE(back) << back.error().message;
  EXPECT_EQ(*back, codes);
}

TEST(quant_cuda, pack_gptq_matches_the_cpu_packer_field_by_field) {
  NEED_GPU();
  namespace pack = mqde::quant::pack;
  std::mt19937 rng(5);
  struct Shape {
    int64_t out, in, group;
  };
  // in*bits and out*bits must be divisible by 32: 3 bit needs 32 | in and 32 | out, 8 bit only 4 | in
  for (int bits : {2, 3, 4, 8})
    for (Shape sh :
         {Shape{32, 128, 32}, Shape{64, 256, 128}, Shape{96, 96, 32}, Shape{32, 32, 32}, Shape{128, 512, 64}})
      for (bool v1 : {true, false}) {
        const int64_t groups = sh.in / sh.group;
        std::vector<uint8_t> codes(static_cast<size_t>(sh.out * sh.in)),
            zeros(static_cast<size_t>(sh.out * groups));
        std::vector<float> scales(zeros.size());
        const uint32_t mask = (1u << bits) - 1u;
        for (auto& c : codes) c = static_cast<uint8_t>(rng() & mask);
        for (auto& z : zeros) z = static_cast<uint8_t>(rng() & mask);  // includes 0: v1 stores (0 - 1) & mask
        std::normal_distribution<float> d(0.0f, 0.02f);
        for (auto& v : scales) v = d(rng);
        auto ref =
            pack::pack_gptq(codes.data(), zeros.data(), scales.data(), sh.out, sh.in, sh.group, bits, v1);
        auto got = mqde::quant::cuda::pack_gptq(codes.data(), zeros.data(), scales.data(), sh.out, sh.in,
                                                sh.group, bits, v1);
        ASSERT_TRUE(ref) << ref.error().message;
        ASSERT_TRUE(got) << got.error().message;
        const std::string what = std::to_string(bits) + " bit [" + std::to_string(sh.out) + "," +
                                 std::to_string(sh.in) + "] g" + std::to_string(sh.group) +
                                 (v1 ? " v1" : " v2");
        EXPECT_EQ(ref->qweight, got->qweight) << what;
        EXPECT_EQ(ref->qzeros, got->qzeros) << what;
        EXPECT_EQ(ref->scales, got->scales) << what;
        EXPECT_EQ(ref->qweight_rows, got->qweight_rows) << what;
        EXPECT_EQ(ref->qweight_cols, got->qweight_cols) << what;
        EXPECT_EQ(ref->qzeros_rows, got->qzeros_rows) << what;
        EXPECT_EQ(ref->qzeros_cols, got->qzeros_cols) << what;
      }
}

TEST(quant_cuda, pack_gptq_converts_scales_to_fp16_exactly_like_the_cpu_for_every_kind_of_value) {
  NEED_GPU();
  namespace pack = mqde::quant::pack;
  // one scale per (group, output): 32 outputs x 1 group per row of the weight, enough rows to carry the
  // values below
  std::vector<float> vals{0.0f,
                          -0.0f,
                          1.0f,
                          -1.0f,
                          65504.0f,
                          65519.99f,
                          65520.0f,
                          1e6f,
                          -1e6f,
                          6.1035156e-05f,
                          5.9604645e-08f,
                          2.9802322e-08f,
                          2.9802323e-08f,
                          1e-10f,
                          1.00048828125f,
                          1.0014648437f,
                          1.0009765625f,
                          3.14159265f,
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::denorm_min(),
                          std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::min()};
  std::mt19937 rng(11);
  for (int i = 0; i < 8000; ++i)
    vals.push_back(std::bit_cast<float>(static_cast<uint32_t>(rng())));  // every exponent, finite or not
  const int64_t out = 32, in = 32, group = 32;
  const size_t per = static_cast<size_t>(out);  // groups = 1
  std::vector<uint8_t> codes(static_cast<size_t>(out * in), 1), zeros(per, 1);
  for (size_t base = 0; base < vals.size(); base += per) {
    std::vector<float> scales(per, 1.0f);
    for (size_t k = 0; k < per && base + k < vals.size(); ++k) scales[k] = vals[base + k];
    auto ref = pack::pack_gptq(codes.data(), zeros.data(), scales.data(), out, in, group, 4, true);
    auto got =
        mqde::quant::cuda::pack_gptq(codes.data(), zeros.data(), scales.data(), out, in, group, 4, true);
    ASSERT_TRUE(ref && got);
    for (size_t k = 0; k < per; ++k)
      ASSERT_EQ(ref->scales[k], got->scales[k])
          << "scale " << scales[k] << " (bits 0x" << std::hex << std::bit_cast<uint32_t>(scales[k]) << ")";
  }
}

TEST(quant_cuda, pack_awq_matches_the_cpu_packer_field_by_field) {
  NEED_GPU();
  namespace pack = mqde::quant::pack;
  std::mt19937 rng(6);
  struct Shape {
    int64_t out, in, group;
  };
  // 4 bit is the interleaved layout; the other widths are the plain stream. out * bits must be a multiple of
  // 32; `in` is free.
  for (int bits : {2, 3, 4, 8})
    for (Shape sh : {Shape{32, 128, 32}, Shape{64, 96, 32}, Shape{96, 100, 50}, Shape{32, 7, 7},
                     Shape{128, 64, 64}, Shape{40, 64, 32}}) {
      const bool ok = (sh.out * bits) % 32 == 0;
      const int64_t groups = sh.in / sh.group;
      std::vector<uint8_t> codes(static_cast<size_t>(sh.out * sh.in)),
          zeros(static_cast<size_t>(sh.out * groups));
      std::vector<float> scales(zeros.size());
      const uint32_t mask = (1u << bits) - 1u;
      for (auto& c : codes) c = static_cast<uint8_t>(rng() & mask);
      for (auto& z : zeros) z = static_cast<uint8_t>(rng() & mask);
      std::normal_distribution<float> d(0.0f, 0.02f);
      for (auto& v : scales) v = d(rng);
      auto ref = pack::pack_awq(codes.data(), zeros.data(), scales.data(), sh.out, sh.in, sh.group, bits);
      auto got = mqde::quant::cuda::pack_awq(codes.data(), zeros.data(), scales.data(), sh.out, sh.in,
                                             sh.group, bits);
      const std::string what = std::to_string(bits) + " bit [" + std::to_string(sh.out) + "," +
                               std::to_string(sh.in) + "] g" + std::to_string(sh.group);
      ASSERT_EQ(static_cast<bool>(ref), static_cast<bool>(got))
          << what << ": the two packers disagree on whether the shape is valid";
      EXPECT_EQ(ok, static_cast<bool>(got)) << what;
      if (!ref) continue;
      EXPECT_EQ(ref->qweight, got->qweight) << what;
      EXPECT_EQ(ref->qzeros, got->qzeros) << what;
      EXPECT_EQ(ref->scales, got->scales) << what;
      EXPECT_EQ(ref->qweight_rows, got->qweight_rows) << what;
      EXPECT_EQ(ref->qweight_cols, got->qweight_cols) << what;
      EXPECT_EQ(ref->qzeros_rows, got->qzeros_rows) << what;
      EXPECT_EQ(ref->qzeros_cols, got->qzeros_cols) << what;
    }
}

TEST(quant_cuda, pack_gptq_rejects_the_shapes_the_cpu_packer_rejects) {
  NEED_GPU();
  std::vector<uint8_t> codes(64 * 64, 0), zeros(64 * 2, 0);
  std::vector<float> scales(64 * 2, 1.0f);
  auto bad = [&](int64_t out, int64_t in, int64_t group, int bits) {
    return mqde::quant::cuda::pack_gptq(codes.data(), zeros.data(), scales.data(), out, in, group, bits,
                                        true);
  };
  EXPECT_FALSE(bad(64, 64, 32, 0));
  EXPECT_FALSE(bad(64, 64, 32, 9));
  EXPECT_FALSE(bad(0, 64, 32, 4));
  EXPECT_FALSE(bad(64, 64, 48, 4)) << "64 is not divisible by 48";
  EXPECT_TRUE(bad(64, 64, 32, 3)) << "64 * 3 = 192 is a multiple of 32";
  EXPECT_FALSE(bad(40, 64, 32, 3)) << "40 * 3 = 120 is not a multiple of 32";
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

TEST(quant_cuda, gguf_q8_0_matches_the_cpu_packer_bit_for_bit) {
  NEED_GPU();
  namespace pack = mqde::quant::pack;
  std::mt19937 rng(13);
  // a mix of ordinary blocks, an all-zero block (d == 0, id must be 0 too: the CPU's `d ? 1/d : 0`), a
  // block whose amax is exactly representable (no rounding ambiguity at the boundary codes +-127), and
  // the max.magnitude block right after it (scale carried over from a previous iteration must not leak)
  for (size_t nb : {size_t{1}, size_t{2}, size_t{3}, size_t{10}, size_t{1000}}) {
    std::vector<float> x(nb * 32);
    std::normal_distribution<float> d(0.0f, 3.0f);
    for (auto& v : x) v = d(rng);
    for (size_t i = 0; i * 32 < x.size(); i += 7)  // every 7th block: all zero
      std::fill(x.begin() + static_cast<long>(i * 32), x.begin() + static_cast<long>(i * 32 + 32), 0.0f);
    const auto ref = pack::gguf_quantize(pack::GgufType::kQ8_0, x.data(), static_cast<int64_t>(x.size()));
    auto got = mqde::quant::cuda::gguf_quantize_q8_0(x.data(), static_cast<int64_t>(x.size()));
    ASSERT_TRUE(ref) << ref.error().message;
    ASSERT_TRUE(got) << got.error().message;
    EXPECT_EQ(*ref, *got) << "nb=" << nb;

    std::vector<float> ref_deq(x.size()), got_deq(x.size());
    ASSERT_TRUE(pack::gguf_dequantize(pack::GgufType::kQ8_0, ref->data(), static_cast<int64_t>(x.size()),
                                      ref_deq.data()));
    ASSERT_TRUE(mqde::quant::cuda::gguf_dequantize_q8_0(got->data(), static_cast<int64_t>(x.size()),
                                                        got_deq.data()));
    EXPECT_EQ(ref_deq, got_deq) << "dequantize, nb=" << nb;
    // round-tripping the CPU's own quantize/dequantize is the real end-to-end contract, not just
    // agreement between the two packers
    std::vector<float> cpu_deq(x.size());
    ASSERT_TRUE(pack::gguf_dequantize(pack::GgufType::kQ8_0, ref->data(), static_cast<int64_t>(x.size()),
                                      cpu_deq.data()));
    EXPECT_EQ(cpu_deq, got_deq) << "cpu-quantized, gpu-dequantized, nb=" << nb;
  }
}

TEST(quant_cuda, gguf_q8_0_rejects_lengths_that_are_not_a_multiple_of_32) {
  NEED_GPU();
  std::vector<float> x(31, 1.0f);
  EXPECT_FALSE(mqde::quant::cuda::gguf_quantize_q8_0(x.data(), 31));
  EXPECT_FALSE(mqde::quant::cuda::gguf_quantize_q8_0(x.data(), 0));
  std::vector<uint8_t> packed(34);
  std::vector<float> out(31);
  EXPECT_FALSE(mqde::quant::cuda::gguf_dequantize_q8_0(packed.data(), 31, out.data()));
}
