// T-0103: ThreadPool and CancelToken.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <set>
#include <stdexcept>
#include <thread>

#include "mqde/common/thread_pool.h"

using namespace std::chrono_literals;

TEST(common_thread_pool, runs_every_task_and_returns_values) {
  mqde::ThreadPool pool(4);
  std::vector<std::future<int>> f;
  for (int i = 0; i < 100; ++i) f.push_back(pool.submit([i] { return i * i; }));
  long sum = 0;
  for (auto& x : f) sum += x.get();
  EXPECT_EQ(sum, 328350);
}

TEST(common_thread_pool, uses_more_than_one_thread_when_tasks_overlap) {
  mqde::ThreadPool pool(4);
  std::mutex m;
  std::set<std::thread::id> ids;
  std::atomic<int> running{0}, peak{0};
  for (int i = 0; i < 8; ++i)
    pool.submit([&] {
      const int now = ++running;
      int p = peak.load();
      while (now > p && !peak.compare_exchange_weak(p, now)) {
      }
      {
        std::lock_guard lk(m);
        ids.insert(std::this_thread::get_id());
      }
      std::this_thread::sleep_for(30ms);
      --running;
    });
  pool.wait_idle();
  EXPECT_GE(peak.load(), 2);
  EXPECT_GE(ids.size(), 2u);
  EXPECT_LE(ids.size(), 4u);
}

TEST(common_thread_pool, an_exception_reaches_the_future_and_the_worker_survives) {
  mqde::ThreadPool pool(1);
  auto bad = pool.submit([]() -> int { throw std::runtime_error("boom"); });
  EXPECT_THROW(bad.get(), std::runtime_error);
  EXPECT_EQ(pool.submit([] { return 7; }).get(), 7) << "the single worker is still alive";
}

TEST(common_thread_pool, wait_idle_waits_for_running_and_queued_work) {
  mqde::ThreadPool pool(2);
  std::atomic<int> done{0};
  for (int i = 0; i < 10; ++i)
    pool.submit([&] {
      std::this_thread::sleep_for(10ms);
      ++done;
    });
  pool.wait_idle();
  EXPECT_EQ(done.load(), 10);
}

TEST(common_thread_pool, the_destructor_finishes_queued_tasks) {
  std::atomic<int> done{0};
  {
    mqde::ThreadPool pool(1);
    for (int i = 0; i < 20; ++i)
      pool.submit([&] {
        std::this_thread::sleep_for(2ms);
        ++done;
      });
  }
  EXPECT_EQ(done.load(), 20);
}

TEST(common_thread_pool, zero_threads_means_hardware_concurrency_at_least_one) {
  mqde::ThreadPool pool(0);
  EXPECT_GE(pool.size(), 1u);
}

TEST(common_cancel, a_token_is_shared_between_copies_and_tasks_stop_at_the_next_check) {
  mqde::CancelToken tok;
  auto copy = tok;
  EXPECT_FALSE(copy.cancelled());
  mqde::ThreadPool pool(1);
  std::atomic<int> iterations{0};
  auto f = pool.submit([copy, &iterations] {
    while (!copy.cancelled()) {
      ++iterations;
      std::this_thread::sleep_for(1ms);
    }
    return true;
  });
  std::this_thread::sleep_for(20ms);
  tok.cancel();
  tok.cancel();  // idempotent
  ASSERT_EQ(f.wait_for(2s), std::future_status::ready);
  EXPECT_TRUE(copy.cancelled());
  EXPECT_GT(iterations.load(), 0);
}
