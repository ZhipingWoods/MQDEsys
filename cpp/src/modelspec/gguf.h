#pragma once
// Reads a GGUF file's header only (magic, metadata, tensor table); the tensor payload is never read, so a multi-GB model is
// inspected in milliseconds. Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md (versions 2 and 3, little-endian).
// Block sizes / type ids are those of gguf-py's GGML_QUANT_SIZES; the unit tests pin them.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mqde/common/config.h"
#include "mqde/common/result.h"

namespace mqde::modelspec {

struct GgufTensor {
  std::string name;
  std::string type;            // "F16", "Q4_K", ...
  std::vector<int64_t> shape;  // row-major (torch) order: GGUF stores the fastest dimension first, this is reversed
  int64_t numel = 0;
  uint64_t nbytes = 0;
  uint64_t offset = 0;  // absolute byte offset of the payload in the file
};

struct GgufFile {
  uint32_t version = 0;
  uint64_t alignment = 32;
  uint64_t data_offset = 0;
  uint64_t file_bytes = 0;
  // Scalars as JSON values; an array becomes {"array_of": "string", "length": N, "head": [first few]} so a 150k-entry
  // vocabulary stays a few bytes.
  Json metadata = Json::object();
  std::vector<GgufTensor> tensors;
};

Result<GgufFile> read_gguf(const std::filesystem::path& file);

// {architecture, name, ..., quant_types: {type: {tensors, params, bytes}}, blocks: [{index, tensors, params, bytes, types}],
//  global: [tensor names outside blk.N], tensors: [...]}. `with_tensors=false` leaves out the per-tensor list.
Json gguf_summary(const GgufFile& g, bool with_tensors = true);

}  // namespace mqde::modelspec
