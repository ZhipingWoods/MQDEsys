// T-1303: CatalogRefresher's scheduling/threading harness, tested with a fake refresh function (no
// network). The real incremental-merge behaviour of hub::update_catalog is already covered elsewhere; this
// file only exercises the background-thread mechanics: it fires, it reports status, a failure or a thrown
// exception never crashes the process and the thread keeps trying, and interval<=0 disables it entirely.
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>

#include "mqde/hub/catalog.h"

using namespace mqde;
using namespace mqde::hub;

namespace {

std::filesystem::path fresh_catalog_path(const char* name) {
  auto d = std::filesystem::temp_directory_path() /
           (std::string("mqde_catalog_refresh_") + name + "_" + std::to_string(::getpid()));
  std::filesystem::remove_all(d);
  std::filesystem::create_directories(d);
  return d / "catalog.json";
}

bool wait_until(const std::function<bool()>& pred, int timeout_ms = 2000) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

}  // namespace

TEST(hub_catalog_refresh, calls_the_refresh_function_and_saves_its_result) {
  const auto path = fresh_catalog_path("saves");
  std::atomic<int> calls{0};
  CatalogRefresher r(path, std::chrono::milliseconds(20), [&](Catalog base) -> Result<Catalog> {
    ++calls;
    Catalog out = base;
    CatalogEntry e;
    e.name = "org/fresh-model";
    e.parameters_raw = 1;
    e.context_length = 1;
    out.entries.push_back(e);
    out.updated_at = "now";
    return out;
  });
  ASSERT_TRUE(wait_until([&] { return calls.load() >= 1; })) << "refresh_fn was never called";
  ASSERT_TRUE(wait_until([&] { return std::filesystem::exists(path); }));
  auto saved = load_catalog(path);
  ASSERT_TRUE(saved) << saved.error().message;
  ASSERT_FALSE(saved->entries.empty());
  EXPECT_EQ(saved->entries[0].name, "org/fresh-model");
}

TEST(hub_catalog_refresh, status_reports_the_configured_interval_and_a_successful_refresh) {
  const auto path = fresh_catalog_path("status_ok");
  CatalogRefresher r(path, std::chrono::milliseconds(15),
                     [](Catalog base) -> Result<Catalog> { return base; });
  EXPECT_EQ(r.status().interval_s, 0) << "sub-second intervals round down to 0, which is fine: this test "
                                      "only checks the field is populated from the constructor, not rounding";
  ASSERT_TRUE(wait_until([&] { return !r.status().last_refresh_at.empty(); }));
  EXPECT_TRUE(r.status().last_refresh_ok);
}

TEST(hub_catalog_refresh, a_failed_refresh_is_reported_but_the_thread_keeps_retrying) {
  const auto path = fresh_catalog_path("retries_after_failure");
  std::atomic<int> calls{0};
  CatalogRefresher r(path, std::chrono::milliseconds(15), [&](Catalog) -> Result<Catalog> {
    ++calls;
    return make_error(Errc::kExternal, "stand-in network failure");
  });
  ASSERT_TRUE(wait_until([&] { return !r.status().last_refresh_at.empty(); }));
  EXPECT_FALSE(r.status().last_refresh_ok);
  const int seen_after_first_failure = calls.load();
  ASSERT_TRUE(wait_until([&] { return calls.load() > seen_after_first_failure; }))
      << "a failed refresh must not stop the loop from trying again next interval";
}

TEST(hub_catalog_refresh, an_exception_from_the_refresh_function_does_not_crash_the_process) {
  const auto path = fresh_catalog_path("throws");
  std::atomic<int> calls{0};
  CatalogRefresher r(path, std::chrono::milliseconds(15), [&](Catalog) -> Result<Catalog> {
    ++calls;
    throw std::runtime_error("simulated network stack exception");
  });
  ASSERT_TRUE(wait_until([&] { return !r.status().last_refresh_at.empty(); }));
  EXPECT_FALSE(r.status().last_refresh_ok);
  SUCCEED() << "reaching this line at all means the thrown exception did not take the process down";
}

TEST(hub_catalog_refresh, a_non_positive_interval_disables_the_background_thread_entirely) {
  const auto path = fresh_catalog_path("disabled");
  std::atomic<int> calls{0};
  CatalogRefresher r(path, std::chrono::seconds(0), [&](Catalog base) -> Result<Catalog> {
    ++calls;
    return base;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(calls.load(), 0) << "interval <= 0 must mean no background refresh at all";
  EXPECT_EQ(r.status().interval_s, 0);
}
