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
  std::string file;  // shard file name
};

// Bytes per element for a safetensors dtype string; 0 for unknown.
double dtype_bytes(std::string_view dtype);

Result<std::vector<TensorInfo>> read_safetensors_header(const std::filesystem::path& file);

// Reads every shard of a model directory: model.safetensors, or *.safetensors.index.json + shards,
// or any *.safetensors files present.
Result<std::vector<TensorInfo>> read_model_tensors(const std::filesystem::path& model_dir);

}  // namespace mqde::modelspec
