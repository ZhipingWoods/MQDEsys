#pragma once
// Fixed-size thread pool and a cooperative cancellation token.
//
// ThreadPool::submit returns a std::future; an exception thrown by a task is delivered through its future and
// never kills a worker. wait_idle() blocks until every queued and running task has finished. The destructor
// finishes queued tasks, then joins. A CancelToken is shared by the code that requests cancellation and the
// tasks that poll it; cancelling is idempotent.

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace mqde {

class CancelToken {
 public:
  CancelToken() : flag_(std::make_shared<std::atomic<bool>>(false)) {}
  void cancel() { flag_->store(true, std::memory_order_release); }
  bool cancelled() const { return flag_->load(std::memory_order_acquire); }
  // Copies share the flag: hand a copy to each task.
 private:
  std::shared_ptr<std::atomic<bool>> flag_;
};

class ThreadPool {
 public:
  explicit ThreadPool(size_t threads = 0);  // 0 = hardware concurrency (at least 1)
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <class F>
  auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut = task->get_future();
    enqueue([task] { (*task)(); });
    return fut;
  }

  void wait_idle();
  size_t size() const { return workers_.size(); }

 private:
  void enqueue(std::function<void()> job);
  void run();
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> q_;
  std::mutex mu_;
  std::condition_variable cv_, idle_cv_;
  size_t active_ = 0;
  bool stop_ = false;
};

}  // namespace mqde
