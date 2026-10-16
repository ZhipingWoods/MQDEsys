#include "mqde/modelspec/safetensors.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>

#include "mqde/common/config.h"

namespace mqde::modelspec {
namespace fs = std::filesystem;

double dtype_bytes(std::string_view dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "F8_E4M3" || dtype == "F8_E5M2" || dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
  return 0;
}

Result<std::vector<TensorInfo>> read_safetensors_header(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return make_error(Errc::kNotFound, "cannot open '{}'", file.string());
  std::array<unsigned char, 8> len_buf{};
  if (!in.read(reinterpret_cast<char*>(len_buf.data()), 8))
    return make_error(Errc::kParse, "'{}' is too short to be a safetensors file", file.string());
  uint64_t len = 0;
  for (int i = 7; i >= 0; --i) len = (len << 8) | len_buf[static_cast<size_t>(i)];
  if (len == 0 || len > kMaxSafetensorsHeader)
    return make_error(Errc::kParse, "'{}': invalid header length {}", file.string(), len);
  std::string header(len, '\0');
  if (!in.read(header.data(), static_cast<std::streamsize>(len)))
    return make_error(Errc::kParse, "'{}': truncated header", file.string());
  return parse_safetensors_header(header, file.filename().string());
}

Result<std::vector<TensorInfo>> parse_safetensors_header(const std::string& header,
                                                         const std::string& file_name) {
  const uint64_t len = header.size();
  Json j;
  try {
    j = Json::parse(header);
  } catch (const std::exception& e) {
    return make_error(Errc::kParse, "'{}': header is not valid JSON: {}", file_name, e.what());
  }
  std::vector<TensorInfo> out;
  out.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    TensorInfo t;
    t.name = it.key();
    t.file = file_name;
    try {
      t.dtype = it->at("dtype").get<std::string>();
      t.shape = it->at("shape").get<std::vector<int64_t>>();
      auto offs = it->at("data_offsets").get<std::vector<uint64_t>>();
      if (offs.size() != 2 || offs[1] < offs[0])
        return make_error(Errc::kParse, "'{}': bad data_offsets for {}", file_name, t.name);
      t.nbytes = offs[1] - offs[0];
      t.data_offset = 8 + len + offs[0];
    } catch (const std::exception& e) {
      return make_error(Errc::kParse, "'{}': bad entry for {}: {}", file_name, t.name, e.what());
    }
    t.numel = 1;
    for (auto d : t.shape) t.numel *= d;
    out.push_back(std::move(t));
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
  return out;
}

Result<std::vector<TensorInfo>> read_model_tensors(const fs::path& model_dir) {
  if (fs::is_regular_file(model_dir)) return read_safetensors_header(model_dir);
  if (!fs::is_directory(model_dir))
    return make_error(Errc::kNotFound, "model path '{}' does not exist", model_dir.string());

  std::set<fs::path> shards;
  for (const auto& e : fs::directory_iterator(model_dir)) {
    const auto name = e.path().filename().string();
    if (name.ends_with(".safetensors.index.json")) {
      MQDE_ASSIGN_OR_RETURN(Json idx, load_structured_file(e.path()));
      // Keep the map alive: iterating items() of the temporary returned by value() dangles.
      const Json weight_map = idx.value("weight_map", Json::object());
      for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
        if (!it->is_string())
          return make_error(Errc::kParse, "'{}': weight_map entry for '{}' is not a file name", name,
                            it.key());
        shards.insert(model_dir / it->get<std::string>());
      }
    }
  }
  if (shards.empty()) {
    for (const auto& e : fs::directory_iterator(model_dir))
      if (e.path().extension() == ".safetensors") shards.insert(e.path());
  }
  if (shards.empty())
    return make_error(Errc::kNotFound, "no *.safetensors found in '{}'", model_dir.string());

  std::vector<TensorInfo> all;
  for (const auto& s : shards) {
    auto r = read_safetensors_header(s);
    if (!r) return std::unexpected(r.error());
    all.insert(all.end(), r->begin(), r->end());
  }
  std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
  return all;
}

namespace {
float bf16_to_f32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float f16_to_f32(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
  uint32_t u;
  if (exp == 0) {
    if (man == 0)
      u = sign;
    else {  // subnormal
      float f = std::ldexp(static_cast<float>(man), -24);
      return sign ? -f : f;
    }
  } else if (exp == 31)
    u = sign | 0x7f800000u | (man << 13);
  else
    u = sign | ((exp + 112) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float f8_to_f32(uint8_t v, bool e4m3) {
  const int mbits = e4m3 ? 3 : 2, ebits = e4m3 ? 4 : 5, bias = e4m3 ? 7 : 15;
  const int sign = v >> 7, exp = (v >> mbits) & ((1 << ebits) - 1), man = v & ((1 << mbits) - 1);
  if (e4m3 && exp == 15 && man == 7) return std::nanf("");
  if (!e4m3 && exp == 31) return man ? std::nanf("") : (sign ? -INFINITY : INFINITY);
  const float m =
      exp == 0 ? static_cast<float>(man) / (1 << mbits) : 1.0f + static_cast<float>(man) / (1 << mbits);
  const float f = std::ldexp(m, (exp == 0 ? 1 : exp) - bias);
  return sign ? -f : f;
}
}  // namespace

Result<std::vector<float>> read_tensor_f32(const fs::path& model_dir, const TensorInfo& t,
                                           int64_t max_elems) {
  const double eb = dtype_bytes(t.dtype);
  if (!(t.dtype == "BF16" || t.dtype == "F16" || t.dtype == "F32" || t.dtype == "F8_E4M3" ||
        t.dtype == "F8_E5M2"))
    return make_error(Errc::kUnsupported, "dtype {} is packed or non-float; no value view", t.dtype);
  std::ifstream in(model_dir / t.file, std::ios::binary);
  if (!in) return make_error(Errc::kNotFound, "cannot open shard '{}'", t.file);
  const int64_t n = t.numel;
  const int64_t stride = std::max<int64_t>(1, n / std::max<int64_t>(1, max_elems));
  const int64_t out_n = (n + stride - 1) / stride;
  std::vector<float> out;
  out.reserve(static_cast<size_t>(out_n));
  const size_t esz = static_cast<size_t>(eb);
  // Read in chunks of contiguous elements; strided sampling picks every `stride`-th value.
  constexpr int64_t kChunk = 1 << 20;
  std::vector<char> buf;
  for (int64_t start = 0; start < n; start += kChunk * stride) {
    const int64_t span = std::min<int64_t>(kChunk * stride, n - start);
    buf.resize(static_cast<size_t>(span) * esz);
    in.seekg(static_cast<std::streamoff>(t.data_offset + static_cast<uint64_t>(start) * esz));
    if (!in.read(buf.data(), static_cast<std::streamsize>(buf.size())))
      return make_error(Errc::kIo, "short read on '{}'", t.name);
    for (int64_t i = 0; i < span; i += stride) {
      const char* p = buf.data() + static_cast<size_t>(i) * esz;
      float v = 0;
      if (t.dtype == "F32")
        std::memcpy(&v, p, 4);
      else if (t.dtype == "BF16") {
        uint16_t h;
        std::memcpy(&h, p, 2);
        v = bf16_to_f32(h);
      } else if (t.dtype == "F16") {
        uint16_t h;
        std::memcpy(&h, p, 2);
        v = f16_to_f32(h);
      } else
        v = f8_to_f32(static_cast<uint8_t>(*p), t.dtype == "F8_E4M3");
      out.push_back(v);
    }
  }
  return out;
}

Result<std::pair<std::vector<float>, int64_t>> read_tensor_rows_f32(const fs::path& model_dir,
                                                                     const TensorInfo& t, int64_t max_elems) {
  if (t.shape.size() != 2)
    return make_error(Errc::kUnsupported, "'{}' is not 2-D (row sampling needs rows and columns)", t.name);
  const double eb = dtype_bytes(t.dtype);
  if (!(t.dtype == "BF16" || t.dtype == "F16" || t.dtype == "F32" || t.dtype == "F8_E4M3" ||
        t.dtype == "F8_E5M2"))
    return make_error(Errc::kUnsupported, "dtype {} is packed or non-float; no value view", t.dtype);
  const int64_t rows_total = t.shape[0], cols = t.shape[1];
  if (rows_total <= 0 || cols <= 0) return make_error(Errc::kInvalidArgument, "'{}' has an empty dimension", t.name);
  const int64_t rows_wanted = std::clamp<int64_t>(max_elems / cols, 1, rows_total);
  std::ifstream in(model_dir / t.file, std::ios::binary);
  if (!in) return make_error(Errc::kNotFound, "cannot open shard '{}'", t.file);
  const size_t esz = static_cast<size_t>(eb);
  std::vector<float> out;
  out.reserve(static_cast<size_t>(rows_wanted * cols));
  std::vector<char> row_buf(static_cast<size_t>(cols) * esz);
  for (int64_t i = 0; i < rows_wanted; ++i) {
    // evenly spaced row indices across the whole height; rows_wanted == rows_total makes this 0,1,2,...,
    // i.e. every row, identically to what a direct full read would give
    const int64_t row = i * rows_total / rows_wanted;
    in.seekg(static_cast<std::streamoff>(t.data_offset + static_cast<uint64_t>(row * cols) * esz));
    if (!in.read(row_buf.data(), static_cast<std::streamsize>(row_buf.size())))
      return make_error(Errc::kIo, "short read on '{}'", t.name);
    for (int64_t c = 0; c < cols; ++c) {
      const char* p = row_buf.data() + static_cast<size_t>(c) * esz;
      float v = 0;
      if (t.dtype == "F32")
        std::memcpy(&v, p, 4);
      else if (t.dtype == "BF16") {
        uint16_t h;
        std::memcpy(&h, p, 2);
        v = bf16_to_f32(h);
      } else if (t.dtype == "F16") {
        uint16_t h;
        std::memcpy(&h, p, 2);
        v = f16_to_f32(h);
      } else
        v = f8_to_f32(static_cast<uint8_t>(*p), t.dtype == "F8_E4M3");
      out.push_back(v);
    }
  }
  return std::make_pair(std::move(out), rows_wanted);
}

}  // namespace mqde::modelspec
