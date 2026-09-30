#pragma once
// Reads safetensors headers only (8-byte little-endian length + JSON), never the tensor payload,
// so a multi-GB checkpoint can be inspected in milliseconds.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mqde/common/result.h"

namespace mqde::modelspec {

struct TensorInfo {
  std::string name;
  std::string dtype;  // safetensors dtype string: "BF16", "F32", "F8_E4M3", ...
  std::vector<int64_t> shape;
  int64_t numel = 0;
  uint64_t nbytes = 0;
  std::string file;          // shard file name
  uint64_t data_offset = 0;  // absolute byte offset of the payload inside `file`
};

// Reads one tensor's payload and converts it to float32 (BF16/F16/F32/F8_E4M3/F8_E5M2).
// At most `max_elems` evenly-strided elements are returned so huge embeddings stay cheap.
Result<std::vector<float>> read_tensor_f32(const std::filesystem::path& model_dir, const TensorInfo& t,
                                           int64_t max_elems = 4'000'000);

// Bytes per element for a safetensors dtype string; 0 for unknown.
double dtype_bytes(std::string_view dtype);

Result<std::vector<TensorInfo>> read_safetensors_header(const std::filesystem::path& file);

// Parses the JSON header (the `header_len` bytes after the 8-byte length) of shard `file_name`; shared by the
// local reader and the Hub's HTTP Range reader.
Result<std::vector<TensorInfo>> parse_safetensors_header(const std::string& header,
                                                         const std::string& file_name);

// Header length limit (same as the reference implementation).
inline constexpr uint64_t kMaxSafetensorsHeader = 100ull << 20;

// Reads every shard of a model directory: model.safetensors, or *.safetensors.index.json + shards,
// or any *.safetensors files present.
Result<std::vector<TensorInfo>> read_model_tensors(const std::filesystem::path& model_dir);

}  // namespace mqde::modelspec
