#include "mqde/quant/quant_spec.h"

namespace mqde::quant {

int32_t QuantSpec::qmin() const {
  if (fmt != NumFormat::kInt) return 0;
  return symmetric ? -(1 << (bits - 1)) : 0;
}

int32_t QuantSpec::qmax() const {
  if (fmt != NumFormat::kInt) return 0;
  return symmetric ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
}

float QuantSpec::fp8_max() const {
  switch (fmt) {
    case NumFormat::kFp8E4M3: return 448.0f;
    case NumFormat::kFp8E5M2: return 57344.0f;
    default: return 0.0f;
  }
}

Result<QuantSpec> parse_quant_spec(const Json& j, std::string_view where) {
  QuantSpec s;
  if (j.is_null()) return s;
  const auto fmt = get_or<std::string>(j, "format", "int");
  if (fmt == "int") s.fmt = NumFormat::kInt;
  else if (fmt == "fp8_e4m3" || fmt == "fp8") s.fmt = NumFormat::kFp8E4M3;
  else if (fmt == "fp8_e5m2") s.fmt = NumFormat::kFp8E5M2;
  else if (fmt == "none" || fmt == "fp16" || fmt == "bf16") s.fmt = NumFormat::kNone;
  else return make_error(Errc::kInvalidArgument, "{}.format: unknown format '{}'", where, fmt);

  if (s.fmt == NumFormat::kNone) return s;
  s.bits = static_cast<int>(get_or<int64_t>(j, "bits", s.fmt == NumFormat::kInt ? 8 : 8));
  if (s.fmt == NumFormat::kInt && (s.bits < 2 || s.bits > 8))
    return make_error(Errc::kInvalidArgument, "{}.bits: integer bits must be in [2, 8], got {}", where, s.bits);
  if (s.fmt != NumFormat::kInt && s.bits != 8)
    return make_error(Errc::kInvalidArgument, "{}.bits: fp8 formats are 8-bit, got {}", where, s.bits);
  s.symmetric = get_or<bool>(j, "symmetric", true);
  if (s.fmt != NumFormat::kInt && !s.symmetric)
    return make_error(Errc::kInvalidArgument, "{}.symmetric: fp8 quantization is symmetric only", where);

  const auto gran = get_or<std::string>(j, "granularity", "per_channel");
  if (gran == "per_tensor") s.gran = Granularity::kPerTensor;
  else if (gran == "per_channel" || gran == "per_token") s.gran = Granularity::kPerChannel;
  else if (gran == "per_group") s.gran = Granularity::kPerGroup;
  else return make_error(Errc::kInvalidArgument, "{}.granularity: unknown '{}'", where, gran);
  s.group_size = get_or<int64_t>(j, "group_size", 0);
  if (s.gran == Granularity::kPerGroup && s.group_size <= 0)
    return make_error(Errc::kInvalidArgument, "{}.group_size: per_group requires group_size > 0", where);

  const auto calib = get_or<std::string>(j, "calib", "minmax");
  if (calib == "minmax") s.calib = CalibAlgo::kMinMax;
  else if (calib == "mse") s.calib = CalibAlgo::kMse;
  else return make_error(Errc::kInvalidArgument, "{}.calib: unknown calibration '{}'", where, calib);
  s.maxshrink = static_cast<float>(get_or<double>(j, "maxshrink", 0.8));
  s.grid = static_cast<int>(get_or<int64_t>(j, "grid", 20));
  if (!(s.maxshrink > 0.0f && s.maxshrink <= 1.0f) || s.grid < 1)
    return make_error(Errc::kInvalidArgument, "{}: maxshrink must be in (0,1] and grid >= 1", where);
  return s;
}

Json to_json(const QuantSpec& s) {
  if (!s.enabled()) return {{"format", "none"}};
  Json j;
  j["format"] = s.fmt == NumFormat::kInt ? "int" : (s.fmt == NumFormat::kFp8E4M3 ? "fp8_e4m3" : "fp8_e5m2");
  j["bits"] = s.bits;
  j["symmetric"] = s.symmetric;
  j["granularity"] = s.gran == Granularity::kPerTensor ? "per_tensor"
                     : s.gran == Granularity::kPerGroup ? "per_group" : "per_channel";
  if (s.gran == Granularity::kPerGroup) j["group_size"] = s.group_size;
  j["calib"] = s.calib == CalibAlgo::kMse ? "mse" : "minmax";
  return j;
}

std::string describe(const QuantSpec& s) {
  if (!s.enabled()) return "FP16/BF16";
  std::string out = s.fmt == NumFormat::kInt ? fmt::format("INT{} {}", s.bits, s.symmetric ? "sym" : "asym")
                    : s.fmt == NumFormat::kFp8E4M3 ? "FP8-E4M3" : "FP8-E5M2";
  out += s.gran == Granularity::kPerTensor ? " per-tensor"
         : s.gran == Granularity::kPerGroup ? fmt::format(" g{}", s.group_size) : " per-channel";
  if (s.calib == CalibAlgo::kMse) out += " mse";
  return out;
}

}  // namespace mqde::quant
