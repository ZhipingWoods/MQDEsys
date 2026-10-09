#include "mqde/storage/db.h"

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <regex>
#include <thread>

#include <fmt/format.h>

#include "mqde/common/config.h"
#include "mqde/common/hash.h"

namespace mqde::storage {

namespace fs = std::filesystem;

namespace {

double now_s() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}

Error sql_error(sqlite3* db, int rc, const std::string& sql) {
  std::string head = sql.substr(0, 60);
  std::replace(head.begin(), head.end(), '\n', ' ');
  return Error{Errc::kIo, fmt::format("sqlite: {} (code {}) while running \"{}\"",
                                      db ? sqlite3_errmsg(db) : sqlite3_errstr(rc), rc, head)};
}

struct Stmt {
  sqlite3_stmt* s = nullptr;
  ~Stmt() {
    if (s) sqlite3_finalize(s);
  }
};

int bind_all(sqlite3_stmt* s, const std::vector<Value>& params) {
  for (size_t i = 0; i < params.size(); ++i) {
    const int idx = static_cast<int>(i) + 1;
    int rc = SQLITE_OK;
    if (std::holds_alternative<std::nullptr_t>(params[i]))
      rc = sqlite3_bind_null(s, idx);
    else if (std::holds_alternative<int64_t>(params[i]))
      rc = sqlite3_bind_int64(s, idx, std::get<int64_t>(params[i]));
    else if (std::holds_alternative<double>(params[i]))
      rc = sqlite3_bind_double(s, idx, std::get<double>(params[i]));
    else {
      const auto& t = std::get<std::string>(params[i]);
      rc = sqlite3_bind_text(s, idx, t.data(), static_cast<int>(t.size()), SQLITE_TRANSIENT);
    }
    if (rc != SQLITE_OK) return rc;
  }
  return SQLITE_OK;
}

}  // namespace

template <class F>
auto Db::with_retry(F&& f) -> decltype(f()) {
  for (int attempt = 0;; ++attempt) {
    auto r = f();
    const bool is_busy = !r && r.error().message.find("(code 5)") != std::string::npos;
    const bool is_locked = !r && r.error().message.find("(code 6)") != std::string::npos;
    if (!(is_busy || is_locked) || attempt >= opts_.retries) return r;
    std::this_thread::sleep_for(std::chrono::milliseconds(2 + attempt * 2));
  }
}

Result<std::unique_ptr<Db>> Db::open(const fs::path& file, const DbOptions& opts) {
  std::unique_ptr<Db> d(new Db());
  d->path_ = file;
  d->opts_ = opts;
  if (!file.parent_path().empty()) {
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
  }
  const int flags = opts.read_only ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
  int rc = sqlite3_open_v2(file.string().c_str(), &d->db_, flags | SQLITE_OPEN_FULLMUTEX, nullptr);
  if (rc != SQLITE_OK) {
    auto e = sql_error(d->db_, rc, "open " + file.string());
    return std::unexpected(e);
  }
  sqlite3_busy_timeout(d->db_, opts.busy_timeout_ms);
  if (!opts.read_only) {
    // WAL: readers never block the writer and the writer never blocks readers. `synchronous=NORMAL` is the
    // WAL default pairing.
    auto r = d->with_retry([&] { return d->execute_locked("PRAGMA journal_mode=WAL", {}); });
    if (!r) return std::unexpected(r.error());
    if (auto q = d->query_locked("PRAGMA journal_mode", {}); q && !q->empty() && as_text((*q)[0][0]) != "wal")
      return make_error(Errc::kIo, "sqlite: could not enable WAL on {} (got '{}')", file.string(),
                        as_text((*q)[0][0]));
    d->execute_locked("PRAGMA synchronous=NORMAL", {});
  }
  d->execute_locked("PRAGMA foreign_keys=ON", {});
  fs::path mig = opts.migrations_dir.empty() ? fs::path(data_dir()) / "migrations" : opts.migrations_dir;
  if (auto m = d->run_migrations(mig); !m) return std::unexpected(m.error());
  return d;
}

Db::~Db() {
  if (db_) sqlite3_close(db_);
}

Result<int64_t> Db::execute_locked(const std::string& sql, const std::vector<Value>& params) {
  Stmt st;
  int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &st.s, nullptr);
  if (rc != SQLITE_OK) return std::unexpected(sql_error(db_, rc, sql));
  if (rc = bind_all(st.s, params); rc != SQLITE_OK) return std::unexpected(sql_error(db_, rc, sql));
  while ((rc = sqlite3_step(st.s)) == SQLITE_ROW) {
  }
  if (rc != SQLITE_DONE) return std::unexpected(sql_error(db_, rc, sql));
  return static_cast<int64_t>(sqlite3_changes(db_));
}

Result<std::vector<Row>> Db::query_locked(const std::string& sql, const std::vector<Value>& params) {
  Stmt st;
  int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &st.s, nullptr);
  if (rc != SQLITE_OK) return std::unexpected(sql_error(db_, rc, sql));
  if (rc = bind_all(st.s, params); rc != SQLITE_OK) return std::unexpected(sql_error(db_, rc, sql));
  std::vector<Row> rows;
  const int n = sqlite3_column_count(st.s);
  while ((rc = sqlite3_step(st.s)) == SQLITE_ROW) {
    Row r;
    for (int c = 0; c < n; ++c) {
      switch (sqlite3_column_type(st.s, c)) {
        case SQLITE_INTEGER:
          r.emplace_back(static_cast<int64_t>(sqlite3_column_int64(st.s, c)));
          break;
        case SQLITE_FLOAT:
          r.emplace_back(sqlite3_column_double(st.s, c));
          break;
        case SQLITE_NULL:
          r.emplace_back(nullptr);
          break;
        default: {
          const auto* t = reinterpret_cast<const char*>(sqlite3_column_text(st.s, c));
          r.emplace_back(std::string(t ? t : "", static_cast<size_t>(sqlite3_column_bytes(st.s, c))));
        }
      }
    }
    rows.push_back(std::move(r));
  }
  if (rc != SQLITE_DONE) return std::unexpected(sql_error(db_, rc, sql));
  return rows;
}

Result<int64_t> Db::execute(const std::string& sql, const std::vector<Value>& params) {
  std::lock_guard lk(mu_);
  return with_retry([&] { return execute_locked(sql, params); });
}

Result<std::vector<Row>> Db::query(const std::string& sql, const std::vector<Value>& params) {
  std::lock_guard lk(mu_);
  return with_retry([&] { return query_locked(sql, params); });
}

int64_t Db::last_insert_rowid() {
  std::lock_guard lk(mu_);
  return sqlite3_last_insert_rowid(db_);
}

Result<void> Db::transaction(const std::function<Result<void>(Db&)>& fn) {
  std::lock_guard lk(mu_);
  if (depth_ > 0) return fn(*this);  // joins the enclosing transaction
  auto begin = with_retry([&] { return execute_locked("BEGIN IMMEDIATE", {}); });
  if (!begin) return std::unexpected(begin.error());
  ++depth_;
  Result<void> r;
  try {
    r = fn(*this);
  } catch (const std::exception& e) {
    r = make_error(Errc::kInternal, "exception inside a transaction: {}", e.what());
  }
  --depth_;
  if (!r) {
    execute_locked("ROLLBACK", {});
    return r;
  }
  auto commit = with_retry([&] { return execute_locked("COMMIT", {}); });
  if (!commit) {
    execute_locked("ROLLBACK", {});
    return std::unexpected(commit.error());
  }
  return {};
}

int Db::schema_version() {
  auto r = query("SELECT COALESCE(MAX(version), 0) FROM schema_migrations");
  return r && !r->empty() ? static_cast<int>(as_int((*r)[0][0])) : 0;
}

std::string Db::journal_mode() {
  auto r = query("PRAGMA journal_mode");
  return r && !r->empty() ? as_text((*r)[0][0]) : "";
}

Result<void> Db::run_migrations(const fs::path& dir) {
  struct Mig {
    int version;
    std::string name, sql, checksum;
  };
  std::vector<Mig> migs;
  std::error_code ec;
  if (!fs::is_directory(dir, ec))
    return make_error(Errc::kNotFound, "migrations directory '{}' not found", dir.string());
  static const std::regex re(R"(^(\d{4})_[A-Za-z0-9_]+\.sql$)");
  for (const auto& e : fs::directory_iterator(dir)) {
    std::smatch m;
    const std::string fn = e.path().filename().string();
    if (!std::regex_match(fn, m, re)) continue;
    std::ifstream in(e.path(), std::ios::binary);
    std::string sql((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    migs.push_back({std::stoi(m[1]), fn, sql, sha256_hex(sql)});
  }
  std::sort(migs.begin(), migs.end(), [](const Mig& a, const Mig& b) { return a.version < b.version; });
  for (size_t i = 1; i < migs.size(); ++i)
    if (migs[i].version == migs[i - 1].version)
      return make_error(Errc::kInvalidArgument, "two migrations have version {}", migs[i].version);

  if (opts_.read_only) {
    auto q = query("SELECT COUNT(*) FROM sqlite_master WHERE name='schema_migrations'");
    if (!q || as_int((*q)[0][0]) == 0)
      return make_error(Errc::kNotFound, "{} has no schema (open it read-write once first)", path_.string());
  } else {
    auto r = execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
        "checksum TEXT NOT NULL, applied_at REAL NOT NULL)");
    if (!r) return std::unexpected(r.error());
  }
  auto applied = query("SELECT version, name, checksum FROM schema_migrations ORDER BY version");
  if (!applied) return std::unexpected(applied.error());
  for (const auto& row : *applied) {
    const int v = static_cast<int>(as_int(row[0]));
    const auto it = std::find_if(migs.begin(), migs.end(), [&](const Mig& m) { return m.version == v; });
    if (it == migs.end())
      return make_error(
          Errc::kInvalidArgument,
          "{} was migrated to version {} ('{}') by a newer build; this build only knows up to {}",
          path_.string(), v, as_text(row[1]), migs.empty() ? 0 : migs.back().version);
    if (it->checksum != as_text(row[2]))
      return make_error(
          Errc::kInvalidArgument,
          "migration {} was modified after it was applied (checksum differs); add a new migration instead",
          it->name);
  }
  if (opts_.read_only) return {};
  for (const auto& m : migs) {
    // Re-check inside the transaction: another process may have applied it between our read and our BEGIN
    // IMMEDIATE.
    auto r = transaction([&](Db&) -> Result<void> {
      auto have =
          query_locked("SELECT COUNT(*) FROM schema_migrations WHERE version=?", {int64_t{m.version}});
      if (!have) return std::unexpected(have.error());
      if (as_int((*have)[0][0]) > 0) return {};
      char* err = nullptr;
      const int rc = sqlite3_exec(db_, m.sql.c_str(), nullptr, nullptr, &err);
      if (rc != SQLITE_OK) {
        Error e{Errc::kIo, fmt::format("migration {} failed: {}", m.name, err ? err : sqlite3_errmsg(db_))};
        sqlite3_free(err);
        return std::unexpected(e);
      }
      auto ins =
          execute_locked("INSERT INTO schema_migrations(version, name, checksum, applied_at) VALUES(?,?,?,?)",
                         {int64_t{m.version}, m.name, m.checksum, now_s()});
      if (!ins) return std::unexpected(ins.error());
      return {};
    });
    if (!r) return r;
  }
  return {};
}

}  // namespace mqde::storage
