// T-0103: SHA-256 against standard vectors and against the system's own sha256 tool, and directory content
// hashes.
#include <gtest/gtest.h>

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#include "mqde/common/hash.h"

namespace fs = std::filesystem;
using mqde::content_hash;
using mqde::sha256_file;
using mqde::sha256_hex;

namespace {
fs::path tmp(const char* n) {
  auto d = fs::temp_directory_path() / (std::string("mqde_hash_") + n + "_" + std::to_string(::getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}
void write(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << s;
}
std::string system_sha256(const fs::path& p) {
  for (const char* cmd : {"sha256sum '%s' 2>/dev/null", "shasum -a 256 '%s' 2>/dev/null"}) {
    char line[512];
    std::snprintf(line, sizeof line, cmd, p.c_str());
    if (FILE* f = ::popen(line, "r")) {
      char out[128] = {};
      const bool ok = std::fgets(out, sizeof out, f) != nullptr;
      ::pclose(f);
      if (ok && std::strlen(out) >= 64) return std::string(out, 64);
    }
  }
  return "";
}
}  // namespace

TEST(common_hash, standard_vectors) {
  EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(sha256_hex(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(common_hash, streaming_in_any_chunking_equals_one_shot) {
  std::mt19937 g(1);
  std::string data(100003, '\0');
  for (auto& c : data) c = static_cast<char>(g());
  const auto want = sha256_hex(data);
  for (size_t chunk : {1u, 7u, 63u, 64u, 65u, 4096u}) {
    mqde::Sha256 h;
    for (size_t i = 0; i < data.size(); i += chunk)
      h.update(data.data() + i, std::min(chunk, data.size() - i));
    EXPECT_EQ(h.hex(), want) << "chunk " << chunk;
  }
}

TEST(common_hash, files_match_the_system_sha256_tool_including_sizes_around_block_edges) {
  const auto d = tmp("file");
  std::mt19937 g(2);
  int compared = 0;
  for (size_t n : {size_t{0}, size_t{1}, size_t{55}, size_t{56}, size_t{63}, size_t{64}, size_t{65},
                   size_t{1} << 20, (size_t{1} << 20) + 17, size_t{3} << 20}) {
    std::string s(n, '\0');
    for (auto& c : s) c = static_cast<char>(g());
    write(d / "f", s);
    auto h = sha256_file(d / "f");
    ASSERT_TRUE(h);
    EXPECT_EQ(*h, sha256_hex(s)) << n;
    const auto sys = system_sha256(d / "f");
    if (!sys.empty()) {
      EXPECT_EQ(*h, sys) << "size " << n;
      ++compared;
    }
  }
  if (compared == 0) GTEST_SKIP() << "no sha256sum / shasum on this machine; vectors above still passed";
}

TEST(common_hash, a_missing_file_is_an_error) {
  EXPECT_FALSE(sha256_file("/definitely/not/here"));
  auto c = content_hash("/definitely/not/here");
  ASSERT_FALSE(c);
  EXPECT_EQ(c.error().code, mqde::Errc::kNotFound);
}

TEST(common_hash, directory_hash_ignores_the_directory_name_and_creation_order_but_sees_every_change) {
  const auto a = tmp("dir_a"), b = tmp("dir_b");
  write(a / "x.txt", "one");
  write(a / "sub" / "y.txt", "two");
  write(b / "sub" / "y.txt", "two");  // created in the opposite order
  write(b / "x.txt", "one");
  auto ha = content_hash(a), hb = content_hash(b);
  ASSERT_TRUE(ha && hb);
  EXPECT_EQ(*ha, *hb);
  write(b / "x.txt", "ONE");
  EXPECT_NE(*content_hash(b), *ha) << "content change";
  write(b / "x.txt", "one");
  fs::rename(b / "x.txt", b / "z.txt");
  EXPECT_NE(*content_hash(b), *ha) << "rename";
  fs::rename(b / "z.txt", b / "x.txt");
  EXPECT_EQ(*content_hash(b), *ha);
  write(b / "extra", "");
  EXPECT_NE(*content_hash(b), *ha) << "an extra empty file";
  EXPECT_EQ(*content_hash(a / "x.txt"), sha256_hex("one")) << "a file path hashes the file";
}
