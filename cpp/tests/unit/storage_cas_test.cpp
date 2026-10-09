// T-0105: content-addressed cache: dedupe, pin / gc, atomic writes and recovery after a kill.
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

#include "mqde/common/hash.h"
#include "mqde/storage/cas.h"

namespace fs = std::filesystem;
using namespace mqde::storage;

namespace {

fs::path tmpdir(const char* n) {
  auto d = fs::temp_directory_path() / (std::string("mqde_cas_") + n + "_" + std::to_string(::getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}
void write(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << s;
}
std::string slurp(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::string blob(size_t n, unsigned seed) {
  std::mt19937 g(seed);
  std::string s(n, '\0');
  for (auto& c : s) c = static_cast<char>(g());
  return s;
}

struct Env {
  fs::path root;
  std::unique_ptr<Db> db;
  std::unique_ptr<Cas> cas;
  explicit Env(const char* name) : root(tmpdir(name)) { reopen(); }
  void reopen() {
    cas.reset();
    db.reset();
    auto d = Db::open(root / "m.db");
    EXPECT_TRUE(d) << (d ? "" : d.error().message);
    db = std::move(*d);
    auto c = Cas::open(root / "cache", *db);
    EXPECT_TRUE(c) << (c ? "" : c.error().message);
    cas = std::move(*c);
  }
};

}  // namespace

TEST(storage_cas, a_file_is_stored_under_its_sha256_and_read_back_identical) {
  Env e("file");
  write(e.root / "in.bin", "hello cas");
  auto r = e.cas->put(e.root / "in.bin");
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_EQ(r->hash, mqde::sha256_hex("hello cas"));
  EXPECT_EQ(r->kind, "file");
  EXPECT_EQ(r->bytes, 9u);
  EXPECT_FALSE(r->existed);
  EXPECT_EQ(slurp(r->path), "hello cas");
  EXPECT_EQ(r->path, e.root / "cache" / "objects" / r->hash.substr(0, 2) / r->hash);
  auto g = e.cas->get(r->hash);
  ASSERT_TRUE(g);
  EXPECT_EQ(slurp(g->path), "hello cas");
  EXPECT_TRUE(e.cas->exists(r->hash));
}

TEST(storage_cas, the_same_content_twice_takes_no_extra_space_even_under_another_name) {
  Env e("dedupe");
  const auto big = blob(2 << 20, 1);
  write(e.root / "a.bin", big);
  write(e.root / "elsewhere" / "other_name.dat", big);
  auto a = e.cas->put(e.root / "a.bin");
  auto b = e.cas->put(e.root / "elsewhere" / "other_name.dat");
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->hash, b->hash);
  EXPECT_FALSE(a->existed);
  EXPECT_TRUE(b->existed) << "nothing copied the second time";
  EXPECT_EQ(*e.cas->total_bytes(), big.size());
  int objects = 0;
  for (const auto& x : fs::recursive_directory_iterator(e.root / "cache" / "objects"))
    objects += x.is_regular_file();
  EXPECT_EQ(objects, 1) << "one object on disk";
}

TEST(storage_cas, a_directory_is_addressed_by_content_not_by_name_or_order) {
  Env e("dir");
  write(e.root / "d1" / "x.txt", "one");
  write(e.root / "d1" / "sub" / "y.txt", "two");
  write(e.root / "d2copy" / "sub" / "y.txt", "two");
  write(e.root / "d2copy" / "x.txt", "one");
  auto a = e.cas->put(e.root / "d1");
  auto b = e.cas->put(e.root / "d2copy");
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->kind, "dir");
  EXPECT_EQ(a->hash, b->hash);
  EXPECT_TRUE(b->existed);
  EXPECT_EQ(slurp(a->path / "sub" / "y.txt"), "two");
  EXPECT_EQ(a->bytes, 6u);
  write(e.root / "d2copy" / "x.txt", "ONE");
  auto c = e.cas->put(e.root / "d2copy");
  ASSERT_TRUE(c);
  EXPECT_NE(c->hash, a->hash);
  EXPECT_FALSE(c->existed);
  ASSERT_TRUE(e.cas->verify(a->hash));
  EXPECT_TRUE(*e.cas->verify(a->hash));
}

TEST(storage_cas, bad_requests_are_errors_that_say_what_is_wrong) {
  Env e("bad");
  EXPECT_EQ(e.cas->put("/definitely/not/here").error().code, mqde::Errc::kNotFound);
  EXPECT_EQ(e.cas->get(std::string(64, 'a')).error().code, mqde::Errc::kNotFound);
  EXPECT_EQ(e.cas->get("../../etc/passwd").error().code, mqde::Errc::kInvalidArgument)
      << "an address is 64 hex characters, never a path";
  EXPECT_FALSE(e.cas->exists("nonsense"));
  EXPECT_FALSE(e.cas->pin(std::string(64, 'b')));
  EXPECT_FALSE(e.cas->unpin(std::string(64, 'b')));
}

TEST(storage_cas, pins_are_a_reference_count_that_never_goes_below_zero) {
  Env e("pins");
  write(e.root / "f", "x");
  auto r = e.cas->put(e.root / "f");
  ASSERT_TRUE(r);
  EXPECT_EQ(*e.cas->pins(r->hash), 0);
  ASSERT_TRUE(e.cas->pin(r->hash));
  ASSERT_TRUE(e.cas->pin(r->hash));
  EXPECT_EQ(*e.cas->pins(r->hash), 2);
  ASSERT_TRUE(e.cas->unpin(r->hash));
  EXPECT_EQ(*e.cas->pins(r->hash), 1);
  ASSERT_TRUE(e.cas->unpin(r->hash));
  ASSERT_TRUE(e.cas->unpin(r->hash));
  EXPECT_EQ(*e.cas->pins(r->hash), 0);
  ASSERT_TRUE(e.cas->put(e.root / "f")) << "putting again keeps the count";
}

TEST(storage_cas, gc_removes_least_recently_used_first_and_stops_at_the_limit) {
  Env e("gc_lru");
  std::vector<std::string> h;
  for (int i = 0; i < 4; ++i) {
    write(e.root / ("f" + std::to_string(i)), blob(1000, 10 + i));
    h.push_back(e.cas->put(e.root / ("f" + std::to_string(i)))->hash);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(e.cas->get(h[0])) << "touching f0 makes it the most recently used";
  auto r = e.cas->gc(2000);
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_EQ(r->removed, 2);
  EXPECT_EQ(r->freed_bytes, 2000u);
  EXPECT_EQ(r->remaining_bytes, 2000u);
  EXPECT_EQ(*e.cas->total_bytes(), 2000u);
  EXPECT_TRUE(e.cas->exists(h[0])) << "recently read: kept";
  EXPECT_FALSE(e.cas->exists(h[1])) << "oldest unused: gone";
  EXPECT_FALSE(e.cas->exists(h[2]));
  EXPECT_TRUE(e.cas->exists(h[3]));
  EXPECT_FALSE(fs::exists(e.root / "cache" / "objects" / h[1].substr(0, 2) / h[1]))
      << "and removed from disk, not only from the table";
  auto none = e.cas->gc(1 << 30);
  EXPECT_EQ(none->removed, 0);
}

TEST(storage_cas, a_pinned_object_survives_gc_even_when_the_limit_cannot_be_met) {
  Env e("gc_pin");
  write(e.root / "keep", blob(5000, 1));
  write(e.root / "drop", blob(5000, 2));
  auto keep = e.cas->put(e.root / "keep");
  auto drop = e.cas->put(e.root / "drop");
  ASSERT_TRUE(keep && drop);
  ASSERT_TRUE(e.cas->pin(keep->hash));
  auto r = e.cas->gc(0);
  ASSERT_TRUE(r);
  EXPECT_EQ(r->removed, 1);
  EXPECT_EQ(r->pinned_kept, 1);
  EXPECT_EQ(r->remaining_bytes, 5000u) << "the limit was 0 but a running job still needs that object";
  EXPECT_TRUE(e.cas->exists(keep->hash));
  EXPECT_EQ(slurp(e.cas->get(keep->hash)->path), slurp(e.root / "keep"));
  EXPECT_FALSE(e.cas->exists(drop->hash));
  ASSERT_TRUE(e.cas->unpin(keep->hash));
  EXPECT_EQ(e.cas->gc(0)->removed, 1) << "after the job releases it, it can go";
}

TEST(storage_cas, an_object_put_again_after_gc_is_stored_again) {
  Env e("regc");
  write(e.root / "f", "again");
  const auto h = e.cas->put(e.root / "f")->hash;
  ASSERT_TRUE(e.cas->gc(0));
  EXPECT_FALSE(e.cas->exists(h));
  auto r = e.cas->put(e.root / "f");
  ASSERT_TRUE(r);
  EXPECT_FALSE(r->existed);
  EXPECT_EQ(slurp(r->path), "again");
}

TEST(storage_cas, open_repairs_what_an_interrupted_run_left_behind) {
  Env e("recover");
  write(e.root / "good", "good content");
  write(e.root / "lost", "lost content");
  write(e.root / "orphan", "orphan content");
  const auto good = e.cas->put(e.root / "good")->hash;
  const auto lost = e.cas->put(e.root / "lost")->hash;
  const auto orphan = e.cas->put(e.root / "orphan")->hash;
  const fs::path objects = e.root / "cache" / "objects";
  // 1. a half-written upload in tmp/
  write(e.root / "cache" / "tmp" / "123-partial" / "half.bin", "xx");
  // 2. an object whose metadata row is missing (crash between rename and insert)
  e.db->execute("DELETE FROM cache_refs WHERE hash=?", {orphan});
  // 3. a metadata row whose object is gone
  fs::remove_all(objects / lost.substr(0, 2) / lost);
  // 4. an object whose bytes are not its address (bit rot / manual edit)
  write(objects / "ab" / (std::string(63, '0') + "1"), "claims to be something else");
  // 5. junk in the objects tree
  write(objects / "zz" / "not-a-hash", "junk");
  e.reopen();
  const auto& rep = e.cas->recovery();
  EXPECT_EQ(rep.tmp_removed, 1);
  EXPECT_EQ(rep.objects_adopted, 1);
  EXPECT_EQ(rep.rows_dropped, 1);
  EXPECT_GE(rep.objects_removed, 2);
  EXPECT_TRUE(e.cas->exists(good));
  EXPECT_TRUE(e.cas->exists(orphan)) << "adopted: its content matches its name";
  EXPECT_FALSE(e.cas->exists(lost));
  EXPECT_FALSE(fs::exists(objects / "ab" / (std::string(63, '0') + "1")));
  EXPECT_FALSE(fs::exists(objects / "zz" / "not-a-hash"));
  EXPECT_TRUE(fs::is_empty(e.root / "cache" / "tmp"));
  EXPECT_EQ(*e.cas->total_bytes(), std::string("good content").size() + std::string("orphan content").size());
}

TEST(storage_cas, verify_notices_a_corrupted_object) {
  Env e("verify");
  write(e.root / "f", "pristine");
  auto r = e.cas->put(e.root / "f");
  ASSERT_TRUE(r);
  EXPECT_TRUE(*e.cas->verify(r->hash));
  fs::permissions(r->path, fs::perms::owner_write, fs::perm_options::add);
  write(r->path, "tampered");
  EXPECT_FALSE(*e.cas->verify(r->hash));
}

// A real SIGKILL in the middle of put(): whatever the timing, reopening must leave a consistent cache.
TEST(storage_cas, killing_a_writer_mid_put_never_leaves_a_corrupt_or_half_visible_object) {
  for (int delay_ms : {5, 20, 60, 120, 250}) {
    const auto root = tmpdir(("kill" + std::to_string(delay_ms)).c_str());
    fs::create_directories(root / "src");
    for (int i = 0; i < 40; ++i) write(root / "src" / ("f" + std::to_string(i)), blob(1 << 20, 100 + i));
    const pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
      auto d = Db::open(root / "m.db");
      if (!d) _exit(2);
      auto c = Cas::open(root / "cache", **d);
      if (!c) _exit(3);
      for (;;)
        for (int i = 0; i < 40; ++i) (*c)->put(root / "src" / ("f" + std::to_string(i)));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    ::kill(pid, SIGKILL);
    int status = 0;
    ::waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFSIGNALED(status)) << "the child was killed, not exited (status " << status << ")";

    auto d = Db::open(root / "m.db");
    ASSERT_TRUE(d) << d.error().message;
    auto c = Cas::open(root / "cache", **d);
    ASSERT_TRUE(c) << c.error().message;
    EXPECT_TRUE(fs::is_empty(root / "cache" / "tmp"))
        << "delay " << delay_ms << ": no write left in progress";
    auto rows = (*d)->query("SELECT hash FROM cache_refs");
    ASSERT_TRUE(rows);
    for (const auto& r : *rows) {
      const auto h = as_text(r[0]);
      auto v = (*c)->verify(h);
      ASSERT_TRUE(v) << "delay " << delay_ms << ": a row points at a missing object " << h;
      EXPECT_TRUE(*v) << "delay " << delay_ms << ": object " << h << " does not match its address";
    }
    int on_disk = 0;
    for (const auto& x : fs::recursive_directory_iterator(root / "cache" / "objects"))
      if (x.is_regular_file()) {
        ++on_disk;
        EXPECT_TRUE((*c)->exists(x.path().filename().string()))
            << "an object on disk without a row after recovery";
      }
    EXPECT_EQ(static_cast<size_t>(on_disk), rows->size()) << "delay " << delay_ms;
    // and the cache is fully usable afterwards
    EXPECT_TRUE((*c)->put(root / "src" / "f0"));
  }
}
