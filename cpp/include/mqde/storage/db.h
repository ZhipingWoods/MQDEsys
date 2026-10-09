#pragma once
// SQLite metadata store (T-0104): WAL mode, versioned migrations, transactions that retry on SQLITE_BUSY.
//
// One Db object is one connection, safe to share between threads (calls are serialised). Several Db objects
// on the same file (several processes, or one per worker thread) coexist: writers wait on each other with a
// busy timeout plus retry.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

#include "mqde/common/result.h"

struct sqlite3;

namespace mqde::storage {

using Value = std::variant<std::nullptr_t, int64_t, double, std::string>;
using Row = std::vector<Value>;

inline int64_t as_int(const Value& v) {
  return std::holds_alternative<int64_t>(v) ? std::get<int64_t>(v) : 0;
}
inline double as_real(const Value& v) {
  return std::holds_alternative<double>(v) ? std::get<double>(v) : static_cast<double>(as_int(v));
}
inline std::string as_text(const Value& v) {
  return std::holds_alternative<std::string>(v) ? std::get<std::string>(v) : std::string();
}
inline bool is_null(const Value& v) { return std::holds_alternative<std::nullptr_t>(v); }

struct DbOptions {
  std::filesystem::path migrations_dir;  // empty: <data dir>/migrations
  int busy_timeout_ms = 5000;
  int retries = 50;  // extra attempts when SQLite still reports BUSY after the timeout
  bool read_only = false;
};

class Db {
 public:
  static Result<std::unique_ptr<Db>> open(const std::filesystem::path& file, const DbOptions& opts = {});
  ~Db();
  Db(const Db&) = delete;
  Db& operator=(const Db&) = delete;

  // Statements with `?` placeholders. execute returns the number of changed rows.
  Result<int64_t> execute(const std::string& sql, const std::vector<Value>& params = {});
  Result<std::vector<Row>> query(const std::string& sql, const std::vector<Value>& params = {});
  int64_t last_insert_rowid();

  // BEGIN IMMEDIATE ... COMMIT around `fn`; ROLLBACK if fn returns an error or throws. Nested calls join the
  // outer transaction.
  Result<void> transaction(const std::function<Result<void>(Db&)>& fn);

  int schema_version();
  std::string journal_mode();
  const std::filesystem::path& path() const { return path_; }

 private:
  Db() = default;
  Result<void> run_migrations(const std::filesystem::path& dir);
  Result<int64_t> execute_locked(const std::string& sql, const std::vector<Value>& params);
  Result<std::vector<Row>> query_locked(const std::string& sql, const std::vector<Value>& params);
  template <class F>
  auto with_retry(F&& f) -> decltype(f());

  sqlite3* db_ = nullptr;
  std::filesystem::path path_;
  DbOptions opts_;
  std::recursive_mutex mu_;
  int depth_ = 0;
};

}  // namespace mqde::storage
