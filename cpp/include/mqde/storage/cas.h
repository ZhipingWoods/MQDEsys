#pragma once
// Content-addressed artifact cache (T-0105).
//
// Layout:  <root>/objects/<hash[0:2]>/<hash>   (a file, or a directory tree)      <root>/tmp/   (writes in
// progress) The address of a file is its sha256; the address of a directory is mqde::content_hash(dir) (a
// hash over the sorted list of "relative path + file hash"). Writes go to tmp/ first and are renamed into
// place, so a reader never sees a half-written object and a crash leaves only tmp/ debris, which open()
// removes. Metadata (size, pins, last access) lives in table cache_refs.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "mqde/common/result.h"
#include "mqde/storage/db.h"

namespace mqde::storage {

struct ArtifactRef {
  std::string hash;
  std::string kind;  // "file" | "dir"
  uint64_t bytes = 0;
  std::filesystem::path path;  // where the object lives now
  bool existed = false;        // put(): the content was already in the cache, nothing was copied
};

struct GcResult {
  uint64_t freed_bytes = 0;
  int removed = 0;
  uint64_t remaining_bytes = 0;
  int pinned_kept = 0;
};

struct RecoveryReport {
  int tmp_removed = 0;      // unfinished writes
  int rows_dropped = 0;     // metadata whose object is missing
  int objects_adopted = 0;  // finished objects that had no metadata (crash between rename and insert)
  int objects_removed = 0;  // objects whose content does not match their address
};

class Cas {
 public:
  // Opens (creating) the cache under `root` using `db` for metadata, and repairs what an interrupted run left
  // behind.
  static Result<std::unique_ptr<Cas>> open(const std::filesystem::path& root, Db& db);

  Result<ArtifactRef> put(const std::filesystem::path& file_or_dir);
  Result<ArtifactRef> get(const std::string& hash);  // kNotFound when absent; updates the last-access time
  bool exists(const std::string& hash);
  Result<void> pin(const std::string& hash);    // reference count + 1: gc never removes a pinned object
  Result<void> unpin(const std::string& hash);  // reference count - 1 (not below 0)
  Result<int> pins(const std::string& hash);
  // Removes unpinned objects, least recently used first, until the cache holds at most max_bytes.
  Result<GcResult> gc(uint64_t max_bytes);
  Result<uint64_t> total_bytes();
  // Re-hashes an object and compares it with its address.
  Result<bool> verify(const std::string& hash);

  const RecoveryReport& recovery() const { return recovery_; }
  const std::filesystem::path& root() const { return root_; }

 private:
  Cas(std::filesystem::path root, Db& db) : root_(std::move(root)), db_(db) {}
  std::filesystem::path object_path(const std::string& hash) const;
  Result<void> recover();

  std::filesystem::path root_;
  Db& db_;
  RecoveryReport recovery_;
  std::mutex put_mu_;
};

}  // namespace mqde::storage
