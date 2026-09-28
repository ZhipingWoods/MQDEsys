#include "mqde/modelspec/safetensors.h"

#include <algorithm>
#include <array>
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
  constexpr uint64_t kMaxHeader = 100ull << 20;  // same limit as the reference implementation
  if (len == 0 || len > kMaxHeader)
    return make_error(Errc::kParse, "'{}': invalid header length {}", file.string(), len);
  std::string header(len, '\0');
  if (!in.read(header.data(), static_cast<std::streamsize>(len)))
    return make_error(Errc::kParse, "'{}': truncated header", file.string());

  Json j;
  try {
    j = Json::parse(header);
  } catch (const std::exception& e) {
    return make_error(Errc::kParse, "'{}': header is not valid JSON: {}", file.string(), e.what());
  }
  std::vector<TensorInfo> out;
  out.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    TensorInfo t;
    t.name = it.key();
    t.file = file.filename().string();
    try {
      t.dtype = it->at("dtype").get<std::string>();
      t.shape = it->at("shape").get<std::vector<int64_t>>();
      auto offs = it->at("data_offsets").get<std::vector<uint64_t>>();
      if (offs.size() != 2 || offs[1] < offs[0])
        return make_error(Errc::kParse, "'{}': bad data_offsets for {}", file.string(), t.name);
      t.nbytes = offs[1] - offs[0];
    } catch (const std::exception& e) {
      return make_error(Errc::kParse, "'{}': bad entry for {}: {}", file.string(), t.name, e.what());
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
  if (!fs::is_directory(model_dir)) return make_error(Errc::kNotFound, "model path '{}' does not exist", model_dir.string());

  std::set<fs::path> shards;
  for (const auto& e : fs::directory_iterator(model_dir)) {
    const auto name = e.path().filename().string();
    if (name.ends_with(".safetensors.index.json")) {
      MQDE_ASSIGN_OR_RETURN(Json idx, load_structured_file(e.path()));
      // Keep the map alive: iterating items() of the temporary returned by value() dangles.
      const Json weight_map = idx.value("weight_map", Json::object());
      for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
        if (!it->is_string())
          return make_error(Errc::kParse, "'{}': weight_map entry for '{}' is not a file name", name, it.key());
        shards.insert(model_dir / it->get<std::string>());
      }
    }
  }
  if (shards.empty()) {
    for (const auto& e : fs::directory_iterator(model_dir))
      if (e.path().extension() == ".safetensors") shards.insert(e.path());
  }
  if (shards.empty()) return make_error(Errc::kNotFound, "no *.safetensors found in '{}'", model_dir.string());

  std::vector<TensorInfo> all;
  for (const auto& s : shards) {
    auto r = read_safetensors_header(s);
    if (!r) return std::unexpected(r.error());
    all.insert(all.end(), r->begin(), r->end());
  }
  std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
  return all;
}

}  // namespace mqde::modelspec
