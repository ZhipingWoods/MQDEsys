#include "mqde/common/thread_pool.h"

#include <algorithm>

namespace mqde {

ThreadPool::ThreadPool(size_t threads) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  for (size_t i = 0; i < threads; ++i) workers_.emplace_back([this] { run(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::enqueue(std::function<void()> job) {
  {
    std::lock_guard lk(mu_);
    q_.push(std::move(job));
  }
  cv_.notify_one();
}

void ThreadPool::run() {
  for (;;) {
    std::function<void()> job;
    {
      std::unique_lock lk(mu_);
      cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
      if (q_.empty()) return;  // stop_ is set and nothing is left: queued work is finished first
      job = std::move(q_.front());
      q_.pop();
      ++active_;
    }
    job();  // packaged_task: exceptions go to the future
    {
      std::lock_guard lk(mu_);
      --active_;
      if (q_.empty() && active_ == 0) idle_cv_.notify_all();
    }
  }
}

void ThreadPool::wait_idle() {
  std::unique_lock lk(mu_);
  idle_cv_.wait(lk, [this] { return q_.empty() && active_ == 0; });
}

}  // namespace mqde
