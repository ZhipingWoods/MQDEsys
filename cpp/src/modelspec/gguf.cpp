#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>

// Propagate the error of a Result<void> (the common header only has the value-binding form).
#define GG_TRY(expr)                                         \
  do {                                                       \
    auto gg_r_ = (expr);                                     \
    if (!gg_r_) return std::unexpected(gg_r_.error());       \
  } while (0)

namespace mqde::modelspec {
namespace fs = std::filesystem;
namespace {

// ggml_type id -> (name, elements per block, bytes per block): gguf-py's GGML_QUANT_SIZES (QK_K = 256)
struct TypeInfo {
  uint32_t id;
  const char* name;
  int64_t blck;
  uint64_t bytes;
};
constexpr TypeInfo kTypes[] = {
    {0, "F32", 1, 4},       {1, "F16", 1, 2},       {2, "Q4_0", 32, 18},     {3, "Q4_1", 32, 20},    {6, "Q5_0", 32, 22},
    {7, "Q5_1", 32, 24},    {8, "Q8_0", 32, 34},    {9, "Q8_1", 32, 36},     {10, "Q2_K", 256, 84},  {11, "Q3_K", 256, 110},
    {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210},  {15, "Q8_K", 256, 292}, {16, "IQ2_XXS", 256, 66},
    {17, "IQ2_XS", 256, 74}, {18, "IQ3_XXS", 256, 98}, {19, "IQ1_S", 256, 50}, {20, "IQ4_NL", 32, 18}, {21, "IQ3_S", 256, 110},
    {22, "IQ2_S", 256, 82}, {23, "IQ4_XS", 256, 136}, {24, "I8", 1, 1},       {25, "I16", 1, 2},      {26, "I32", 1, 4},
    {27, "I64", 1, 8},      {28, "F64", 1, 8},      {29, "IQ1_M", 256, 56},  {30, "BF16", 1, 2},     {34, "TQ1_0", 256, 54},
    {35, "TQ2_0", 256, 66}, {39, "MXFP4", 32, 17},  {40, "NVFP4", 64, 36},   {41, "Q1_0", 128, 18},  {42, "Q2_0", 64, 18},
};

const TypeInfo* find_type(uint32_t id) {
  for (const auto& t : kTypes)
    if (t.id == id) return &t;
  return nullptr;
}

constexpr uint64_t kMaxTensors = 1u << 24, kMaxKv = 1u << 22, kMaxDims = 8, kHead = 8;

// Bounds-checked sequential reader; every failure carries the file and what was being read.
class Reader {
 public:
  Reader(std::ifstream& in, uint64_t size, std::string file) : in_(in), size_(size), file_(std::move(file)) {}
  uint64_t pos() const { return pos_; }
  Result<void> read(void* dst, uint64_t n, const char* what) {
    if (n > size_ - std::min(pos_, size_)) return std::unexpected(trunc(what));
    in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!in_) return std::unexpected(trunc(what));
    pos_ += n;
    return {};
  }
  template <class T>
  Result<T> num(const char* what) {
    T v{};
    GG_TRY(read(&v, sizeof v, what));
    return v;
  }
  Result<std::string> str(const char* what) {
    MQDE_ASSIGN_OR_RETURN(uint64_t n, num<uint64_t>(what));
    if (n > size_ - std::min(pos_, size_)) return std::unexpected(trunc(what));
    std::string s(n, '\0');
    if (n) GG_TRY(read(s.data(), n, what));
    return s;
  }
  Result<void> skip(uint64_t n, const char* what) {
    if (n > size_ - std::min(pos_, size_)) return std::unexpected(trunc(what));
    in_.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    pos_ += n;
    if (!in_) return std::unexpected(trunc(what));
    return {};
  }
  Error trunc_error(const char* what) const { return Error{Errc::kParse, fmt::format("'{}': truncated GGUF file while reading {}", file_, what)}; }

 private:
  Error trunc(const char* what) const { return trunc_error(what); }
  std::ifstream& in_;
  uint64_t size_, pos_ = 0;
  std::string file_;
};

int64_t scalar_size(uint32_t t) {
  switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
  }
}
const char* scalar_name(uint32_t t) {
  static const char* n[] = {"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "bool", "string", "array", "uint64", "int64", "float64"};
  return t < 13 ? n[t] : "unknown";
}

Result<Json> scalar(Reader& r, uint32_t t, const char* what) {
  switch (t) {
    case 0: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<uint8_t>(what)); return Json(v); }
    case 1: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<int8_t>(what)); return Json(v); }
    case 2: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<uint16_t>(what)); return Json(v); }
    case 3: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<int16_t>(what)); return Json(v); }
    case 4: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<uint32_t>(what)); return Json(v); }
    case 5: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<int32_t>(what)); return Json(v); }
    case 6: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<float>(what)); return Json(v); }
    case 7: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<uint8_t>(what)); return Json(v != 0); }
    case 8: { MQDE_ASSIGN_OR_RETURN(auto v, r.str(what)); return Json(v); }
    case 10: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<uint64_t>(what)); return Json(v); }
    case 11: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<int64_t>(what)); return Json(v); }
    case 12: { MQDE_ASSIGN_OR_RETURN(auto v, r.num<double>(what)); return Json(v); }
    default: return make_error(Errc::kParse, "unknown GGUF metadata value type {} in {}", t, what);
  }
}

Result<Json> value(Reader& r, uint32_t t, const std::string& key) {
  if (t != 9) return scalar(r, t, key.c_str());
  MQDE_ASSIGN_OR_RETURN(uint32_t et, r.num<uint32_t>(key.c_str()));
  MQDE_ASSIGN_OR_RETURN(uint64_t n, r.num<uint64_t>(key.c_str()));
  if (et == 9) return make_error(Errc::kParse, "metadata '{}': nested arrays are not part of GGUF", key);
  Json head = Json::array();
  if (const int64_t sz = scalar_size(et); sz > 0) {  // fixed-size elements: read the head, seek past the rest
    for (uint64_t i = 0; i < std::min<uint64_t>(n, kHead); ++i) { MQDE_ASSIGN_OR_RETURN(Json v, scalar(r, et, key.c_str())); head.push_back(v); }
    if (n > kHead) {
      if (n - kHead > UINT64_MAX / static_cast<uint64_t>(sz)) return std::unexpected(r.trunc_error(key.c_str()));
      GG_TRY(r.skip((n - kHead) * static_cast<uint64_t>(sz), key.c_str()));
    }
  } else if (et == 8) {  // strings: each is length-prefixed, so walk them without keeping them
    for (uint64_t i = 0; i < n; ++i) {
      MQDE_ASSIGN_OR_RETURN(uint64_t len, r.num<uint64_t>(key.c_str()));
      if (i < kHead) {
        std::string s(len > (1u << 20) ? 0 : len, '\0');
        if (len > (1u << 20)) { GG_TRY(r.skip(len, key.c_str())); s = "(long)"; }
        else if (len) { GG_TRY(r.read(s.data(), len, key.c_str())); }
        head.push_back(s);
      } else {
        GG_TRY(r.skip(len, key.c_str()));
      }
    }
  } else {
    return make_error(Errc::kParse, "metadata '{}': unknown array element type {}", key, et);
  }
  return Json{{"array_of", scalar_name(et)}, {"length", n}, {"head", head}};
}

}  // namespace

Result<GgufFile> read_gguf(const fs::path& file) {
  std::error_code ec;
  const uint64_t size = fs::file_size(file, ec);
  if (ec) return make_error(Errc::kNotFound, "cannot open '{}'", file.string());
  std::ifstream in(file, std::ios::binary);
  if (!in) return make_error(Errc::kNotFound, "cannot open '{}'", file.string());
  Reader r(in, size, file.string());
  GgufFile g;
  g.file_bytes = size;
  char magic[4];
  GG_TRY(r.read(magic, 4, "the magic"));
  if (std::memcmp(magic, "GGUF", 4) != 0) return make_error(Errc::kParse, "'{}': not a GGUF file (magic is not 'GGUF')", file.string());
  MQDE_ASSIGN_OR_RETURN(g.version, r.num<uint32_t>("the version"));
  if (g.version > 0xffff)
    return make_error(Errc::kUnsupported, "'{}': a big-endian GGUF file (version field reads {:#x}); only little-endian is supported", file.string(), g.version);
  if (g.version < 2 || g.version > 3)
    return make_error(Errc::kUnsupported, "'{}': GGUF version {} is not supported (2 and 3 are)", file.string(), g.version);
  MQDE_ASSIGN_OR_RETURN(uint64_t n_tensors, r.num<uint64_t>("the tensor count"));
  MQDE_ASSIGN_OR_RETURN(uint64_t n_kv, r.num<uint64_t>("the metadata count"));
  if (n_tensors > kMaxTensors) return make_error(Errc::kParse, "'{}': tensor count {} is not credible", file.string(), n_tensors);
  if (n_kv > kMaxKv) return make_error(Errc::kParse, "'{}': metadata count {} is not credible", file.string(), n_kv);
  for (uint64_t i = 0; i < n_kv; ++i) {
    MQDE_ASSIGN_OR_RETURN(std::string key, r.str("a metadata key"));
    MQDE_ASSIGN_OR_RETURN(uint32_t t, r.num<uint32_t>(key.c_str()));
    MQDE_ASSIGN_OR_RETURN(Json v, value(r, t, key));
    g.metadata[key] = std::move(v);
  }
  if (g.metadata.contains("general.alignment") && g.metadata["general.alignment"].is_number_unsigned()) {
    g.alignment = g.metadata["general.alignment"].get<uint64_t>();
    if (g.alignment == 0 || (g.alignment & (g.alignment - 1)) != 0)
      return make_error(Errc::kParse, "'{}': general.alignment {} is not a power of two", file.string(), g.alignment);
  }
  struct Raw { GgufTensor t; uint64_t rel = 0; };
  std::vector<Raw> raw;
  raw.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    Raw x;
    MQDE_ASSIGN_OR_RETURN(x.t.name, r.str("a tensor name"));
    MQDE_ASSIGN_OR_RETURN(uint32_t nd, r.num<uint32_t>(x.t.name.c_str()));
    if (nd == 0 || nd > kMaxDims) return make_error(Errc::kParse, "'{}': tensor '{}' has {} dimensions (1..{} expected)", file.string(), x.t.name, nd, kMaxDims);
    std::vector<int64_t> ne;
    for (uint32_t d = 0; d < nd; ++d) {
      MQDE_ASSIGN_OR_RETURN(uint64_t v, r.num<uint64_t>(x.t.name.c_str()));
      if (v == 0 || v > (1ull << 40)) return make_error(Errc::kParse, "'{}': tensor '{}' has a dimension of {}", file.string(), x.t.name, v);
      ne.push_back(static_cast<int64_t>(v));
    }
    MQDE_ASSIGN_OR_RETURN(uint32_t ty, r.num<uint32_t>(x.t.name.c_str()));
    const TypeInfo* ti = find_type(ty);
    if (!ti) return make_error(Errc::kUnsupported, "'{}': tensor '{}' has unknown tensor type {}", file.string(), x.t.name, ty);
    MQDE_ASSIGN_OR_RETURN(x.rel, r.num<uint64_t>(x.t.name.c_str()));
    if (ne[0] % ti->blck != 0)
      return make_error(Errc::kParse, "'{}': tensor '{}' ({}): row length {} is not a multiple of the block size {}", file.string(), x.t.name, ti->name, ne[0], ti->blck);
    x.t.type = ti->name;
    x.t.numel = 1;
    for (auto d : ne) x.t.numel *= d;
    x.t.nbytes = static_cast<uint64_t>(x.t.numel / ti->blck) * ti->bytes;
    x.t.shape.assign(ne.rbegin(), ne.rend());
    raw.push_back(std::move(x));
  }
  g.data_offset = (r.pos() + g.alignment - 1) / g.alignment * g.alignment;
  for (auto& x : raw) {
    x.t.offset = g.data_offset + x.rel;
    if (x.t.offset > size || x.t.nbytes > size - x.t.offset)
      return make_error(Errc::kParse, "'{}': tensor '{}' ends beyond the end of the file ({} + {} > {} bytes): truncated or still downloading?",
                        file.string(), x.t.name, x.t.offset, x.t.nbytes, size);
    g.tensors.push_back(std::move(x.t));
  }
  return g;
}

Json gguf_summary(const GgufFile& g, bool with_tensors) {
  static const std::regex kBlk(R"(^blk\.(\d+)\.)");
  const auto meta_str = [&](const char* k) -> std::string {
    return g.metadata.contains(k) && g.metadata[k].is_string() ? g.metadata[k].get<std::string>() : std::string();
  };
  Json out = {{"version", g.version}, {"alignment", g.alignment}, {"file_bytes", g.file_bytes}, {"data_offset", g.data_offset},
              {"n_tensors", g.tensors.size()}, {"metadata", g.metadata}};
  const std::string arch = meta_str("general.architecture");
  out["architecture"] = arch;
  out["name"] = meta_str("general.name");
  if (const auto key = arch + ".block_count"; !arch.empty() && g.metadata.contains(key) && g.metadata[key].is_number_integer())
    out["block_count"] = g.metadata[key];
  struct Agg { int64_t tensors = 0, params = 0; uint64_t bytes = 0; };
  std::map<std::string, Agg> types;
  struct Blk { int64_t tensors = 0, params = 0; uint64_t bytes = 0; std::map<std::string, int64_t> types; };
  std::map<int, Blk> blocks;
  Json global = Json::array(), ts = Json::array();
  int64_t params = 0;
  uint64_t bytes = 0;
  for (const auto& t : g.tensors) {
    params += t.numel;
    bytes += t.nbytes;
    auto& a = types[t.type];
    a.tensors++, a.params += t.numel, a.bytes += t.nbytes;
    std::smatch m;
    if (std::regex_search(t.name, m, kBlk)) {
      auto& b = blocks[std::stoi(m[1])];
      b.tensors++, b.params += t.numel, b.bytes += t.nbytes, b.types[t.type]++;
    } else {
      global.push_back(t.name);
    }
    if (with_tensors) ts.push_back({{"name", t.name}, {"type", t.type}, {"shape", t.shape}, {"numel", t.numel}, {"nbytes", t.nbytes}, {"offset", t.offset}});
  }
  out["n_params"] = params;
  out["tensor_bytes"] = bytes;
  Json qt = Json::object();
  for (const auto& [k, a] : types) qt[k] = {{"tensors", a.tensors}, {"params", a.params}, {"bytes", a.bytes}};
  out["quant_types"] = qt;
  Json bl = Json::array();
  for (const auto& [i, b] : blocks) bl.push_back({{"index", i}, {"tensors", b.tensors}, {"params", b.params}, {"bytes", b.bytes}, {"types", b.types}});
  out["blocks"] = bl;
  out["global"] = global;
  if (with_tensors) out["tensors"] = ts;
  return out;
}

}  // namespace mqde::modelspec
