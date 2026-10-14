// T-0406 benchmark: MSE calibration + fake quantization of one 4096 x 4096 weight (per-group W4, group 128,
// grid 20), single thread, once per kernel set this CPU can run. Also checks that every set returns the
// scalar's bits.
//
//   build/<preset>/bench/quant_bench [rows cols group]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "mqde/quant/ops.h"
#include "quant_kernels.h"
#ifdef MQDE_QUANT_CUDA
#include "quant_cuda.h"
#endif

using namespace mqde::quant;
namespace k = mqde::quant::kernels;

int main(int argc, char** argv) {
  const int64_t rows = argc > 1 ? std::atoll(argv[1]) : 4096, cols = argc > 2 ? std::atoll(argv[2]) : 4096;
  const int64_t group = argc > 3 ? std::atoll(argv[3]) : 128;
  std::mt19937 rng(1);
  std::normal_distribution<float> d(0.0f, 0.02f);
  std::vector<float> w(static_cast<size_t>(rows * cols));
  for (auto& v : w) v = d(rng);

  QuantSpec s;
  s.fmt = NumFormat::kInt;
  s.bits = 4;
  s.symmetric = false;
  s.gran = Granularity::kPerGroup;
  s.group_size = group;
  s.calib = CalibAlgo::kMse;

  std::printf("%lld x %lld, W4 per-group g%lld, MSE grid %d, 1 thread\n", static_cast<long long>(rows),
              static_cast<long long>(cols), static_cast<long long>(group), s.grid);
  std::vector<float> ref(w.size()), out(w.size());
  double base = 0;
  int bad = 0;
  for (const k::Set* set : k::available()) {
    k::force(set);
    const auto t0 = std::chrono::steady_clock::now();
    QParams qp;
    auto r = fake_quant(w.data(), rows, cols, s, out.data(), &qp);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!r) {
      std::printf("%-7s failed: %s\n", set->name, r.error().message.c_str());
      return 2;
    }
    const bool scalar = set == &k::scalar();
    if (scalar) {
      ref = out;
      base = sec;
    }
    const bool same = scalar || std::memcmp(ref.data(), out.data(), out.size() * sizeof(float)) == 0;
    bad += !same;
    std::printf("%-7s %7.3f s  %s%s\n", set->name, sec, scalar ? "(reference)" : "",
                scalar ? ""
                : same ? "bit-identical"
                       : "DIFFERS");
    if (!scalar && base > 0) std::printf("        %.1fx faster than scalar\n", base / sec);
  }
  k::force(nullptr);
#ifdef MQDE_QUANT_CUDA
  if (mqde::quant::cuda::available()) {
    std::vector<float> g(w.size());
    for (int rep = 0; rep < 2; ++rep) {  // the first call pays for context creation
      const auto t0 = std::chrono::steady_clock::now();
      auto r = mqde::quant::cuda::fake_quant(w.data(), rows, cols, s, g.data());
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (!r) {
        std::printf("cuda    failed: %s\n", r.error().message.c_str());
        return 2;
      }
      const bool same = std::memcmp(ref.data(), g.data(), g.size() * sizeof(float)) == 0;
      bad += !same;
      std::printf("cuda    %7.3f s total, %.1f ms kernels  %s%s\n", sec, mqde::quant::cuda::last_kernel_ms(),
                  same ? "bit-identical" : "DIFFERS",
                  rep == 0 ? "  (first call: includes context creation)" : "");
    }
  } else {
    std::printf("cuda    no device\n");
  }
#endif
  return bad ? 1 : 0;
}
