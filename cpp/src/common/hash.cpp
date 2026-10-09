#include "mqde/common/hash.h"

#include <algorithm>
#include <fstream>
#include <vector>

#include <fmt/format.h>

namespace mqde {

namespace {
constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
}  // namespace

void Sha256::block(const uint8_t* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
           p[4 * i + 3];
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
  for (int i = 0; i < 64; ++i) {
    const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
    const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
  }
  h_[0] += a, h_[1] += b, h_[2] += c, h_[3] += d, h_[4] += e, h_[5] += f, h_[6] += g, h_[7] += hh;
}

void Sha256::update(const void* data, size_t n) {
  total_ += n;
  const auto* p = static_cast<const uint8_t*>(data);
  while (n > 0) {
    const size_t take = std::min(n, sizeof(buf_) - fill_);
    std::copy(p, p + take, buf_ + fill_);
    fill_ += take, p += take, n -= take;
    if (fill_ == sizeof(buf_)) block(buf_), fill_ = 0;
  }
}

std::string Sha256::hex() {
  const uint64_t bits = total_ * 8;
  const uint8_t one = 0x80, zero = 0;
  update(&one, 1);
  while (fill_ != 56) update(&zero, 1);
  uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
  update(len, 8);
  return fmt::format("{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}", h_[0], h_[1], h_[2], h_[3], h_[4],
                     h_[5], h_[6], h_[7]);
}

std::string sha256_hex(std::string_view data) {
  Sha256 h;
  h.update(data);
  return h.hex();
}

Result<std::string> sha256_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return make_error(Errc::kIo, "cannot read {}", p.string());
  Sha256 h;
  std::vector<char> buf(1 << 20);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    h.update(buf.data(), static_cast<size_t>(in.gcount()));
  }
  return h.hex();
}

Result<std::string> content_hash(const std::filesystem::path& p) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::is_regular_file(p, ec)) return sha256_file(p);
  if (!fs::is_directory(p, ec))
    return make_error(Errc::kNotFound, "{}: no such file or directory", p.string());
  std::vector<std::string> lines;
  for (const auto& e :
       fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied, ec)) {
    if (!e.is_regular_file(ec)) continue;
    auto h = sha256_file(e.path());
    if (!h) return std::unexpected(h.error());
    lines.push_back(fmt::format("{}\t{}\n", fs::relative(e.path(), p).generic_string(), *h));
  }
  std::sort(lines.begin(), lines.end());
  Sha256 total;
  for (const auto& l : lines) total.update(l);
  return total.hex();
}

}  // namespace mqde
