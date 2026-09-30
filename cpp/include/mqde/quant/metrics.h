#pragma once
// Error metrics between a reference tensor and its quantized approximation, plus distribution
// statistics that explain *why* a tensor quantizes badly (outliers, heavy tails).

#include <cstdint>

#include "mqde/common/config.h"

namespace mqde::quant {

struct ErrorMetrics {
  double mse = 0;
  double rmse = 0;
  double max_abs = 0;
  double sqnr_db = 0;  // 10*log10(signal power / noise power); +inf when exact
  double cosine = 1;
  double rel_l2 = 0;  // ||ref - approx|| / ||ref||
};

struct TensorStats {
  double min = 0, max = 0, absmax = 0, mean = 0, std = 0;
  double kurtosis = 0;       // excess kurtosis; 0 for Gaussian
  double outlier_ratio = 0;  // fraction with |x - mean| > 6 std
};

ErrorMetrics measure(const float* ref, const float* approx, int64_t n);
TensorStats tensor_stats(const float* x, int64_t n);

Json to_json(const ErrorMetrics& m);
Json to_json(const TensorStats& s);

}  // namespace mqde::quant
