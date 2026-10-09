#pragma once
// Streaming SHA-256 (FIPS 180-4) and content hashing for files and byte strings. No external dependency.

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "mqde/common/result.h"

namespace mqde {

class Sha256 {
 public:
  void update(const void* data, size_t n);
  void update(std::string_view s) { update(s.data(), s.size()); }
  // Finishes the hash and returns 64 lowercase hex characters. The object must not be updated afterwards.
  std::string hex();

 private:
  void block(const uint8_t* p);
  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf_[64] = {};
  size_t fill_ = 0;
  uint64_t total_ = 0;
};

std::string sha256_hex(std::string_view data);
// Streams the file in 1 MiB chunks, so a multi-GB checkpoint is hashed without being loaded.
Result<std::string> sha256_file(const std::filesystem::path& p);
// Hash of a directory tree: sha256 over the sorted lines "<relative path>\t<file sha256>\n" of every regular
// file. Equal content gives an equal hash whatever the file system order or the directory's own name.
Result<std::string> content_hash(const std::filesystem::path& file_or_dir);

}  // namespace mqde
