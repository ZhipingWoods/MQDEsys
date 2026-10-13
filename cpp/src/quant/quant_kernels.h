#pragma once
// Inner loops of the integer quantizer (min/max scan, MSE error, quantize-dequantize) behind a
// runtime-dispatched table.
//
// The scalar set is the reference and is what ops.cpp always did; the SIMD sets (AVX2, NEON) must return the
// same BITS, not merely close values: the float operations are the same IEEE operations in the same order per
// element, and the squared error is still summed in double one element after the other (a vector reduction
// would change the rounding and with it which MSE candidate wins). Internal header: not part of the public
// API.

#include <cstdint>
#include <string_view>
#include <vector>

namespace mqde::quant::kernels {

// One group's integer quantizer: q = clamp(rne(v / scale) + zero, qmin, qmax); out = (q - zero) * scale
struct IntQdq {
  float scale, zero, qmin, qmax;
};

struct Set {
  const char* name;
  // Fold x[0..n) into *mn / *mx as `mn = std::min(mn, v); mx = std::max(mx, v)` would (NaN never replaces a
  // value).
  void (*minmax)(const float* x, int64_t n, float* mn, float* mx);
  // acc + sum over i of (double(qdq(x[i])) - x[i])^2, added in index order.
  double (*sq_err)(const float* x, int64_t n, const IntQdq& p, double acc);
  void (*qdq)(const float* x, int64_t n, const IntQdq& p, float* out);  // `out` may alias `x`
};

const Set& scalar();
// Every set this CPU can run, scalar first.
std::vector<const Set*> available();
const Set* find(std::string_view name);  // nullptr when unknown or not runnable here
// What the quantizer uses: the widest available set, unless MQDE_QUANT_KERNELS names another one (scalar,
// avx2, neon) or force() says so.
const Set& active();
void force(const Set* s);  // nullptr: back to the default. For tests and benchmarks.

}  // namespace mqde::quant::kernels
