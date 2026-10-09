#include "mqde/storage/cas.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <set>

#include <fmt/format.h>

#include "mqde/common/hash.h"

namespace mqde::storage {

namespace fs = std::filesystem;

namespace {

double now_s() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}

uint64_t tree_bytes(const fs::path& p) {
  std::error_code ec;
  if (fs::is_regular_file(p, ec)) return static_cast<uint64_t>(fs::file_size(p, ec));
  uint64_t n = 0;
  for (const auto& e : fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied, ec))
    if (e.is_regular_file(ec)) n += static_cast<uint64_t>(e.file_size(ec));
  return n;
}

bool valid_hash(const std::string& h) {
  return h.size() == 64 && std::all_of(h.begin(), h.end(), [](unsigned char c) {
           return std::isdigit(c) || (c >= 'a' && c <= 'f');
         });
}

std::string unique_name() {
  static std::atomic<uint64_t> n{0};
  return fmt::format("{}-{}-{}", ::getpid(), std::chrono::steady_clock::now().time_since_epoch().count(),
                     n++);
}

}  // namespace

fs::path Cas::object_path(const std::string& hash) const {
  return root_ / "objects" / hash.substr(0, 2) / hash;
}

Result<std::unique_ptr<Cas>> Cas::open(const fs::path& root, Db& db) {
  std::error_code ec;
  fs::create_directories(root / "objects", ec);
  fs::create_directories(root / "tmp", ec);
  if (ec) return make_error(Errc::kIo, "cannot create cache directory {}: {}", root.string(), ec.message());
  std::unique_ptr<Cas> c(new Cas(root, db));
  if (auto r = c->recover(); !r) return std::unexpected(r.error());
  return c;
}

Result<void> Cas::recover() {
  std::error_code ec;
  // 1. unfinished writes
  for (const auto& e : fs::directory_iterator(root_ / "tmp", ec)) {
    fs::remove_all(e.path(), ec);
    ++recovery_.tmp_removed;
  }
  // 2. objects on disk: adopt the ones without metadata, remove the ones whose content is not their address
  auto rows = db_.query("SELECT hash FROM cache_refs");
  if (!rows) return std::unexpected(rows.error());
  std::set<std::string> known;
  for (const auto& r : *rows) known.insert(as_text(r[0]));
  for (const auto& shard : fs::directory_iterator(root_ / "objects", ec)) {
    if (!shard.is_directory(ec)) continue;
    for (const auto& obj : fs::directory_iterator(shard.path(), ec)) {
      const std::string h = obj.path().filename().string();
      if (!valid_hash(h) || h.substr(0, 2) != shard.path().filename().string()) {
        fs::remove_all(obj.path(), ec);
        ++recovery_.objects_removed;
        continue;
      }
      if (known.count(h)) continue;
      auto actual = content_hash(obj.path());
      if (actual && *actual == h) {
        const bool dir = fs::is_directory(obj.path(), ec);
        auto ins = db_.execute(
            "INSERT OR IGNORE INTO cache_refs(hash, kind, bytes, pins, created_at, last_access) "
            "VALUES(?,?,?,0,?,?)",
            {h, std::string(dir ? "dir" : "file"), static_cast<int64_t>(tree_bytes(obj.path())), now_s(),
             now_s()});
        if (!ins) return std::unexpected(ins.error());
        ++recovery_.objects_adopted;
      } else {
        fs::remove_all(obj.path(), ec);
        ++recovery_.objects_removed;
      }
    }
  }
  // 3. metadata whose object is gone
  for (const auto& h : known)
    if (!fs::exists(object_path(h), ec)) {
      if (auto d = db_.execute("DELETE FROM cache_refs WHERE hash=?", {h}); !d)
        return std::unexpected(d.error());
      ++recovery_.rows_dropped;
    }
  return {};
}

Result<ArtifactRef> Cas::put(const fs::path& src) {
  std::error_code ec;
  const bool is_dir = fs::is_directory(src, ec);
  if (!is_dir && !fs::is_regular_file(src, ec))
    return make_error(Errc::kNotFound, "{}: no such file or directory", src.string());
  auto h = content_hash(src);
  if (!h) return std::unexpected(h.error());
  const std::string hash = *h;
  ArtifactRef ref{hash, is_dir ? "dir" : "file", 0, object_path(hash), false};

  std::lock_guard lk(put_mu_);
  auto have = db_.query("SELECT bytes FROM cache_refs WHERE hash=?", {hash});
  if (!have) return std::unexpected(have.error());
  if (!have->empty() && fs::exists(ref.path, ec)) {  // same content again: nothing is copied or stored twice
    ref.bytes = static_cast<uint64_t>(as_int((*have)[0][0]));
    ref.existed = true;
    db_.execute("UPDATE cache_refs SET last_access=? WHERE hash=?", {now_s(), hash});
    return ref;
  }
  // Copy into tmp/, then one rename into objects/: atomic on the same file system.
  const fs::path tmp = root_ / "tmp" / unique_name();
  fs::copy(src, tmp, fs::copy_options::recursive | fs::copy_options::copy_symlinks, ec);
  if (ec) {
    fs::remove_all(tmp, ec);
    return make_error(Errc::kIo, "copy of {} into the cache failed: {}", src.string(), ec.message());
  }
  ref.bytes = tree_bytes(tmp);
  fs::create_directories(ref.path.parent_path(), ec);
  if (fs::exists(ref.path, ec))
    fs::remove_all(ref.path, ec);  // a stale object without metadata (e.g. after a manual delete of the row)
  fs::rename(tmp, ref.path, ec);
  if (ec) {
    fs::remove_all(tmp, ec);
    return make_error(Errc::kIo, "cannot move the object into place: {}", ec.message());
  }
  auto ins = db_.execute(
      "INSERT OR REPLACE INTO cache_refs(hash, kind, bytes, pins, created_at, last_access) "
      "VALUES(?,?,?,COALESCE((SELECT pins FROM cache_refs WHERE hash=?),0),?,?)",
      {hash, ref.kind, static_cast<int64_t>(ref.bytes), hash, now_s(), now_s()});
  if (!ins) return std::unexpected(ins.error());
  return ref;
}

Result<ArtifactRef> Cas::get(const std::string& hash) {
  if (!valid_hash(hash)) return make_error(Errc::kInvalidArgument, "'{}' is not a sha256 address", hash);
  auto r = db_.query("SELECT kind, bytes FROM cache_refs WHERE hash=?", {hash});
  if (!r) return std::unexpected(r.error());
  std::error_code ec;
  if (r->empty() || !fs::exists(object_path(hash), ec))
    return make_error(Errc::kNotFound, "no cached object {}", hash);
  db_.execute("UPDATE cache_refs SET last_access=? WHERE hash=?", {now_s(), hash});
  return ArtifactRef{hash, as_text((*r)[0][0]), static_cast<uint64_t>(as_int((*r)[0][1])), object_path(hash),
                     true};
}

bool Cas::exists(const std::string& hash) {
  if (!valid_hash(hash)) return false;
  auto r = db_.query("SELECT 1 FROM cache_refs WHERE hash=?", {hash});
  std::error_code ec;
  return r && !r->empty() && fs::exists(object_path(hash), ec);
}

Result<void> Cas::pin(const std::string& hash) {
  auto n = db_.execute("UPDATE cache_refs SET pins = pins + 1 WHERE hash=?", {hash});
  if (!n) return std::unexpected(n.error());
  if (*n == 0) return make_error(Errc::kNotFound, "cannot pin {}: not in the cache", hash);
  return {};
}

Result<void> Cas::unpin(const std::string& hash) {
  auto n = db_.execute("UPDATE cache_refs SET pins = MAX(pins - 1, 0) WHERE hash=?", {hash});
  if (!n) return std::unexpected(n.error());
  if (*n == 0) return make_error(Errc::kNotFound, "cannot unpin {}: not in the cache", hash);
  return {};
}

Result<int> Cas::pins(const std::string& hash) {
  auto r = db_.query("SELECT pins FROM cache_refs WHERE hash=?", {hash});
  if (!r) return std::unexpected(r.error());
  if (r->empty()) return make_error(Errc::kNotFound, "{} is not in the cache", hash);
  return static_cast<int>(as_int((*r)[0][0]));
}

Result<uint64_t> Cas::total_bytes() {
  auto r = db_.query("SELECT COALESCE(SUM(bytes),0) FROM cache_refs");
  if (!r) return std::unexpected(r.error());
  return static_cast<uint64_t>(as_int((*r)[0][0]));
}

Result<GcResult> Cas::gc(uint64_t max_bytes) {
  GcResult out;
  auto total = total_bytes();
  if (!total) return std::unexpected(total.error());
  uint64_t size = *total;
  auto rows = db_.query("SELECT hash, bytes, pins FROM cache_refs ORDER BY last_access ASC, created_at ASC");
  if (!rows) return std::unexpected(rows.error());
  std::error_code ec;
  for (const auto& r : *rows) {
    if (as_int(r[2]) > 0) {
      ++out.pinned_kept;
      continue;
    }
    if (size <= max_bytes) continue;
    const std::string h = as_text(r[0]);
    // Re-check the pin count in the delete itself: a pin that arrived since the SELECT must win.
    auto del = db_.execute("DELETE FROM cache_refs WHERE hash=? AND pins=0", {h});
    if (!del) return std::unexpected(del.error());
    if (*del == 0) {
      ++out.pinned_kept;
      continue;
    }
    fs::remove_all(object_path(h), ec);
    const auto b = static_cast<uint64_t>(as_int(r[1]));
    size -= std::min(size, b);
    out.freed_bytes += b;
    ++out.removed;
  }
  out.remaining_bytes = size;
  return out;
}

Result<bool> Cas::verify(const std::string& hash) {
  auto g = get(hash);
  if (!g) return std::unexpected(g.error());
  auto actual = content_hash(g->path);
  if (!actual) return std::unexpected(actual.error());
  return *actual == hash;
}

}  // namespace mqde::storage
