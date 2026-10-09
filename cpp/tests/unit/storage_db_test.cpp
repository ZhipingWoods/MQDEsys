// T-0104: SQLite store: migrations, WAL, transactions, and concurrent writers.
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

#include "mqde/storage/db.h"

namespace fs = std::filesystem;
using namespace mqde::storage;

namespace {

fs::path tmpdir(const char* n) {
  auto d = fs::temp_directory_path() / (std::string("mqde_db_") + n + "_" + std::to_string(::getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}
void write(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << s;
}
std::unique_ptr<Db> open_ok(const fs::path& f, DbOptions o = {}) {
  auto r = Db::open(f, o);
  EXPECT_TRUE(r) << (r ? "" : r.error().message);
  return r ? std::move(*r) : nullptr;
}
int64_t count(Db& d, const std::string& table) {
  auto r = d.query("SELECT COUNT(*) FROM " + table);
  return r ? as_int((*r)[0][0]) : -1;
}
void add_job(Db& d, const std::string& id) {
  auto r =
      d.execute("INSERT INTO jobs(id, state, spec_json, created_at) VALUES(?, 'QUEUED', '{}', 1.0)", {id});
  ASSERT_TRUE(r) << r.error().message;
}

}  // namespace

TEST(storage_db, an_empty_file_is_migrated_to_the_latest_version_in_wal_mode) {
  auto d = open_ok(tmpdir("fresh") / "m.db");
  ASSERT_TRUE(d);
  EXPECT_EQ(d->schema_version(), 1);
  EXPECT_EQ(d->journal_mode(), "wal");
  for (const char* t :
       {"jobs", "stages", "events", "artifacts", "hw_profiles", "bench_history", "calibration", "cache_refs"})
    EXPECT_GE(count(*d, t), 0) << t << " exists";
}

TEST(storage_db, running_it_again_is_idempotent_and_keeps_the_original_apply_time) {
  const auto f = tmpdir("again") / "m.db";
  double t1 = 0;
  {
    auto d = open_ok(f);
    add_job(*d, "j1");
    t1 = as_real((*d->query("SELECT applied_at FROM schema_migrations WHERE version=1"))[0][0]);
  }
  auto d2 = open_ok(f);
  EXPECT_EQ(d2->schema_version(), 1);
  EXPECT_EQ(count(*d2, "schema_migrations"), 1);
  EXPECT_EQ(count(*d2, "jobs"), 1) << "data survives a reopen";
  EXPECT_EQ(as_real((*d2->query("SELECT applied_at FROM schema_migrations WHERE version=1"))[0][0]), t1);
}

TEST(storage_db, a_migration_edited_after_it_was_applied_is_refused_by_name) {
  const auto dir = tmpdir("edit");
  write(dir / "mig" / "0001_a.sql", "CREATE TABLE a(x INTEGER);");
  DbOptions o;
  o.migrations_dir = dir / "mig";
  ASSERT_TRUE(open_ok(dir / "m.db", o));
  write(dir / "mig" / "0001_a.sql", "CREATE TABLE a(x INTEGER, y INTEGER);");
  auto r = Db::open(dir / "m.db", o);
  ASSERT_FALSE(r);
  EXPECT_NE(r.error().message.find("0001_a.sql was modified after it was applied"), std::string::npos)
      << r.error().message;
}

TEST(storage_db, a_database_from_a_newer_build_is_refused) {
  const auto dir = tmpdir("newer");
  write(dir / "new" / "0001_a.sql", "CREATE TABLE a(x INTEGER);");
  write(dir / "new" / "0002_b.sql", "CREATE TABLE b(x INTEGER);");
  write(dir / "old" / "0001_a.sql", "CREATE TABLE a(x INTEGER);");
  DbOptions n, o;
  n.migrations_dir = dir / "new";
  o.migrations_dir = dir / "old";
  ASSERT_TRUE(open_ok(dir / "m.db", n));
  auto r = Db::open(dir / "m.db", o);
  ASSERT_FALSE(r);
  EXPECT_NE(r.error().message.find("newer build"), std::string::npos) << r.error().message;
}

TEST(storage_db, a_failing_migration_leaves_no_trace_of_itself) {
  const auto dir = tmpdir("fail");
  write(dir / "mig" / "0001_ok.sql", "CREATE TABLE a(x INTEGER);");
  write(dir / "mig" / "0002_bad.sql", "CREATE TABLE b(x INTEGER);\nINSERT INTO nope VALUES(1);");
  DbOptions o;
  o.migrations_dir = dir / "mig";
  auto r = Db::open(dir / "m.db", o);
  ASSERT_FALSE(r);
  EXPECT_NE(r.error().message.find("0002_bad.sql failed"), std::string::npos) << r.error().message;
  fs::remove(dir / "mig" / "0002_bad.sql");
  auto d = open_ok(dir / "m.db", o);
  ASSERT_TRUE(d);
  EXPECT_EQ(d->schema_version(), 1);
  EXPECT_TRUE(d->query("SELECT 1 FROM a")) << "migration 1 was kept";
  EXPECT_FALSE(d->query("SELECT 1 FROM b")) << "table b from the failed migration was rolled back";
}

TEST(storage_db, bad_migration_directories_are_errors) {
  DbOptions o;
  o.migrations_dir = "/definitely/not/here";
  auto r = Db::open(tmpdir("nodir") / "m.db", o);
  ASSERT_FALSE(r);
  EXPECT_EQ(r.error().code, mqde::Errc::kNotFound);
  const auto dir = tmpdir("dupver");
  write(dir / "mig" / "0001_a.sql", "CREATE TABLE a(x);");
  write(dir / "mig" / "0001_b.sql", "CREATE TABLE b(x);");
  o.migrations_dir = dir / "mig";
  EXPECT_FALSE(Db::open(dir / "m.db", o));
}

TEST(storage_db, values_round_trip_and_parameters_are_never_spliced_into_the_sql) {
  auto d = open_ok(tmpdir("vals") / "m.db");
  ASSERT_TRUE(d);
  const std::string nasty = "x'); DROP TABLE jobs; --";
  ASSERT_TRUE(
      d->execute("INSERT INTO jobs(id, name, state, spec_json, gpu, created_at, error) VALUES(?,?,?,?,?,?,?)",
                 {std::string("j"), nasty, std::string("RUNNING"), std::string("{\"k\":\"ü汉字\"}"),
                  int64_t{3}, 2.5, nullptr}));
  auto r =
      d->query("SELECT name, spec_json, gpu, created_at, error FROM jobs WHERE id=?", {std::string("j")});
  ASSERT_TRUE(r);
  ASSERT_EQ(r->size(), 1u);
  EXPECT_EQ(as_text((*r)[0][0]), nasty);
  EXPECT_EQ(as_text((*r)[0][1]), "{\"k\":\"ü汉字\"}");
  EXPECT_EQ(as_int((*r)[0][2]), 3);
  EXPECT_DOUBLE_EQ(as_real((*r)[0][3]), 2.5);
  EXPECT_TRUE(is_null((*r)[0][4]));
  EXPECT_EQ(count(*d, "jobs"), 1) << "the jobs table is still there";
}

TEST(storage_db, a_transaction_commits_or_rolls_back_as_a_whole) {
  auto d = open_ok(tmpdir("tx") / "m.db");
  ASSERT_TRUE(d);
  ASSERT_TRUE(d->transaction([](Db& x) -> mqde::Result<void> {
    add_job(x, "a");
    add_job(x, "b");
    return {};
  }));
  EXPECT_EQ(count(*d, "jobs"), 2);
  auto failed = d->transaction([](Db& x) -> mqde::Result<void> {
    add_job(x, "c");
    return mqde::make_error(mqde::Errc::kInternal, "stop");
  });
  EXPECT_FALSE(failed);
  EXPECT_EQ(count(*d, "jobs"), 2) << "c was rolled back";
  auto thrown = d->transaction([](Db& x) -> mqde::Result<void> {
    add_job(x, "d");
    throw std::runtime_error("oops");
  });
  ASSERT_FALSE(thrown);
  EXPECT_NE(thrown.error().message.find("oops"), std::string::npos);
  EXPECT_EQ(count(*d, "jobs"), 2);
  // nested: the inner one joins; a failure of the inner one aborts the outer one
  auto nested = d->transaction([](Db& x) -> mqde::Result<void> {
    add_job(x, "e");
    return x.transaction([](Db& y) -> mqde::Result<void> {
      add_job(y, "f");
      return mqde::make_error(mqde::Errc::kInternal, "inner");
    });
  });
  EXPECT_FALSE(nested);
  EXPECT_EQ(count(*d, "jobs"), 2);
  EXPECT_TRUE(d->execute("INSERT INTO jobs(id,state,spec_json,created_at) VALUES('g','QUEUED','{}',1)"))
      << "usable after all of that";
}

TEST(storage_db, constraints_are_enforced_events_key_is_job_and_seq_and_deleting_a_job_cascades) {
  auto d = open_ok(tmpdir("fk") / "m.db");
  ASSERT_TRUE(d);
  EXPECT_FALSE(d->execute("INSERT INTO events(job_id,seq,ts,type,payload) VALUES('ghost',1,1,'log','{}')"))
      << "foreign key";
  add_job(*d, "j");
  ASSERT_TRUE(d->execute("INSERT INTO events(job_id,seq,ts,type,payload) VALUES('j',1,1,'log','{}')"));
  EXPECT_FALSE(d->execute("INSERT INTO events(job_id,seq,ts,type,payload) VALUES('j',1,2,'log','{}')"))
      << "same (job, seq) twice";
  ASSERT_TRUE(d->execute("INSERT INTO events(job_id,seq,ts,type,payload) VALUES('j',2,2,'log','{}')"));
  ASSERT_TRUE(d->execute("INSERT INTO stages(job_id,name,seq,status) VALUES('j','quantize',0,'running')"));
  ASSERT_TRUE(d->execute("DELETE FROM jobs WHERE id='j'"));
  EXPECT_EQ(count(*d, "events"), 0);
  EXPECT_EQ(count(*d, "stages"), 0);
}

TEST(storage_db, eight_threads_sharing_one_connection_lose_no_write) {
  auto d = open_ok(tmpdir("share") / "m.db");
  ASSERT_TRUE(d);
  add_job(*d, "j");
  std::atomic<int> failures{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t)
    ts.emplace_back([&, t] {
      for (int i = 0; i < 100; ++i)
        if (!d->execute("INSERT INTO events(job_id,seq,ts,type,payload) VALUES('j',?,1,'log','{}')",
                        {int64_t{t * 1000 + i}}))
          ++failures;
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(count(*d, "events"), 800);
}

TEST(storage_db, eight_connections_writing_the_same_file_never_surface_sqlite_busy) {
  const auto f = tmpdir("multi") / "m.db";
  {
    auto d = open_ok(f);
    add_job(*d, "j");
  }
  std::atomic<int> failures{0};
  std::string first_error;
  std::mutex m;
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; ++t)
    ts.emplace_back([&, t] {
      auto d = Db::open(f);  // its own connection: this is the case that produces SQLITE_BUSY
      if (!d) {
        ++failures;
        return;
      }
      for (int i = 0; i < 150; ++i) {
        auto r = (*d)->transaction([&](Db& x) -> mqde::Result<void> {
          auto a = x.execute("INSERT INTO events(job_id,seq,ts,type,payload) VALUES('j',?,1,'log','{}')",
                             {int64_t{t * 1000 + i}});
          if (!a) return std::unexpected(a.error());
          auto b = x.execute("UPDATE jobs SET gpu = COALESCE(gpu,0) + 1 WHERE id='j'");
          if (!b) return std::unexpected(b.error());
          return {};
        });
        if (!r) {
          ++failures;
          std::lock_guard lk(m);
          if (first_error.empty()) first_error = r.error().message;
        }
      }
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(failures.load(), 0) << first_error;
  auto d = open_ok(f);
  EXPECT_EQ(count(*d, "events"), 1200);
  EXPECT_EQ(as_int((*d->query("SELECT gpu FROM jobs WHERE id='j'"))[0][0]), 1200)
      << "every read-modify-write was applied exactly once";
}

TEST(storage_db, readers_see_committed_data_while_a_writer_holds_a_transaction_open) {
  const auto f = tmpdir("wal") / "m.db";
  auto w = open_ok(f);
  add_job(*w, "j");
  DbOptions ro;
  ro.read_only = true;
  auto r = open_ok(f, ro);
  ASSERT_TRUE(r);
  std::atomic<bool> in_tx{false}, release{false};
  std::thread writer([&] {
    w->transaction([&](Db& x) -> mqde::Result<void> {
      add_job(x, "k");
      in_tx = true;
      while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(2));
      return {};
    });
  });
  while (!in_tx) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  EXPECT_EQ(count(*r, "jobs"), 1) << "WAL: the reader is not blocked and does not see the uncommitted row";
  release = true;
  writer.join();
  EXPECT_EQ(count(*r, "jobs"), 2);
  EXPECT_FALSE(r->execute("DELETE FROM jobs")) << "a read-only connection cannot write";
}

TEST(storage_db, a_read_only_open_of_a_file_without_a_schema_says_so) {
  const auto f = tmpdir("ro_none") / "m.db";
  {
    std::ofstream(f) << "";
  }
  DbOptions ro;
  ro.read_only = true;
  auto r = Db::open(f, ro);
  ASSERT_FALSE(r);
  EXPECT_NE(r.error().message.find("no schema"), std::string::npos) << r.error().message;
}

TEST(storage_db, several_processes_or_threads_opening_a_new_file_at_once_migrate_it_exactly_once) {
  const auto f = tmpdir("race") / "m.db";
  std::atomic<int> failures{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 6; ++t)
    ts.emplace_back([&] {
      if (!Db::open(f)) ++failures;
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(failures.load(), 0);
  auto d = open_ok(f);
  EXPECT_EQ(count(*d, "schema_migrations"), 1);
  EXPECT_EQ(d->schema_version(), 1);
}
