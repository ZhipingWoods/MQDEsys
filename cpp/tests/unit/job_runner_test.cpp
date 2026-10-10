// T-0504 (retry/恢复 half): JobRunner retries a failed launch up to max_attempts with backoff, never
// retries a user-cancelled job, and `merged_state` reports a job that is retrying as QUEUED/RUNNING rather
// than the stale FAILED line an earlier attempt left in events.jsonl.
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include "job_runner.h"

using namespace mqde;
using namespace mqde::server;
namespace fs = std::filesystem;

namespace {

// Counts its own invocations (via a file next to job.yaml, so each job id gets its own counter) and fails
// the first MQDE_TEST_FAIL_UNTIL of them, then exits 0. Stands in for the Python worker: JobRunner only
// cares about the exit code.
fs::path write_fake_worker(const fs::path& dir) {
  fs::create_directories(dir);
  const auto p = dir / "fake_worker.sh";
  std::ofstream f(p);
  f << "#!/bin/sh\n"
       "job_yaml=\"$3\"\n"
       "d=$(dirname \"$job_yaml\")\n"
       "cnt=\"$d/.attempt_count\"\n"
       "n=0\n"
       "[ -f \"$cnt\" ] && n=$(cat \"$cnt\")\n"
       "n=$((n+1))\n"
       "echo \"$n\" > \"$cnt\"\n"
       "fail_until=\"${MQDE_TEST_FAIL_UNTIL:-0}\"\n"
       "if [ \"$n\" -le \"$fail_until\" ]; then exit 1; fi\n"
       "exit 0\n";
  f.close();
  ::chmod(p.c_str(), 0755);
  return p;
}

fs::path fresh_dir(const char* name) {
  auto d = fs::temp_directory_path() /
           (std::string("mqde_job_runner_") + name + "_" + std::to_string(::getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

JobProc wait_for_terminal(JobRunner& r, const std::string& id, int timeout_ms = 8000) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    auto p = r.get(id);
    if (p && (p->state == "SUCCEEDED" || p->state == "FAILED" || p->state == "CANCELLED")) return *p;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ADD_FAILURE() << "job " << id << " never reached a terminal state within " << timeout_ms << "ms";
  return *r.get(id);
}

}  // namespace

TEST(job_runner_retry, a_job_that_fails_fewer_than_max_attempts_times_eventually_succeeds) {
  auto base = fresh_dir("retries_then_ok");
  const auto script = write_fake_worker(base / "bin");
  ::setenv("MQDE_TEST_FAIL_UNTIL", "2", 1);  // fails attempts 1 and 2, succeeds on attempt 3
  JobRunner runner(base / "runs", base, script.string(), {}, /*max_attempts=*/3, /*base_backoff_s=*/0.05);
  const auto id = runner.submit(Json{{"deploy", {{"device", "cpu"}}}});
  const auto final_ = wait_for_terminal(runner, id);
  EXPECT_EQ(final_.state, "SUCCEEDED");
  EXPECT_EQ(final_.attempt, 3);
  EXPECT_EQ(final_.exit_code, 0);
}

TEST(job_runner_retry, a_job_that_keeps_failing_stops_after_max_attempts) {
  auto base = fresh_dir("exhausts_retries");
  const auto script = write_fake_worker(base / "bin");
  ::setenv("MQDE_TEST_FAIL_UNTIL", "999", 1);  // always fails
  JobRunner runner(base / "runs", base, script.string(), {}, /*max_attempts=*/3, /*base_backoff_s=*/0.05);
  const auto id = runner.submit(Json{{"deploy", {{"device", "cpu"}}}});
  const auto final_ = wait_for_terminal(runner, id);
  EXPECT_EQ(final_.state, "FAILED");
  EXPECT_EQ(final_.attempt, 3) << "must stop retrying once max_attempts is reached, not loop forever";
  EXPECT_NE(final_.exit_code, 0);
}

TEST(job_runner_retry, a_job_that_succeeds_on_the_first_try_is_not_retried) {
  auto base = fresh_dir("first_try_ok");
  const auto script = write_fake_worker(base / "bin");
  ::setenv("MQDE_TEST_FAIL_UNTIL", "0", 1);
  JobRunner runner(base / "runs", base, script.string(), {}, /*max_attempts=*/3, /*base_backoff_s=*/0.05);
  const auto id = runner.submit(Json{{"deploy", {{"device", "cpu"}}}});
  const auto final_ = wait_for_terminal(runner, id);
  EXPECT_EQ(final_.state, "SUCCEEDED");
  EXPECT_EQ(final_.attempt, 1);
}

TEST(job_runner_retry, cancelling_a_running_job_prevents_any_retry_even_though_it_exits_nonzero) {
  auto base = fresh_dir("cancel_no_retry");
  const auto d = base / "bin";
  fs::create_directories(d);
  const auto script = d / "sleep_forever.sh";
  {
    std::ofstream f(script);
    f << "#!/bin/sh\nsleep 30\n";
  }
  ::chmod(script.c_str(), 0755);
  JobRunner runner(base / "runs", base, script.string(), {}, /*max_attempts=*/3, /*base_backoff_s=*/0.05);
  const auto id = runner.submit(Json{{"deploy", {{"device", "cpu"}}}});
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto p = runner.get(id); p && p->state == "RUNNING") break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ASSERT_TRUE(runner.cancel(id));
  // cancel() sets state=CANCELLED synchronously, before the killed process is even reaped. The real risk
  // is reap() clobbering that once it actually collects the dead pid (and, if cancellation were not
  // checked there, requeuing a retry) -- so keep watching well past when reap() must have run (every
  // ~500ms) and past the 0.05s backoff that a wrongly-triggered retry would use.
  bool left_cancelled = false;
  JobProc last;
  const auto watch_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
  while (std::chrono::steady_clock::now() < watch_until) {
    auto p = runner.get(id);
    ASSERT_TRUE(p);
    last = *p;
    if (p->state != "CANCELLED") {
      left_cancelled = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_FALSE(left_cancelled) << "state changed away from CANCELLED to " << last.state
                               << " once the killed process was reaped -- it must not be retried";
  EXPECT_EQ(last.state, "CANCELLED");
  EXPECT_EQ(last.attempt, 1) << "a cancelled job must not be retried";
}

TEST(job_runner_retry, cancelling_a_queued_retry_before_its_backoff_elapses_prevents_relaunch) {
  auto base = fresh_dir("cancel_while_backing_off");
  const auto script = write_fake_worker(base / "bin");
  ::setenv("MQDE_TEST_FAIL_UNTIL", "999", 1);
  // A long backoff so the job is reliably still waiting (state QUEUED, attempt 2) when we cancel it.
  JobRunner runner(base / "runs", base, script.string(), {}, /*max_attempts=*/5, /*base_backoff_s=*/5.0);
  const auto id = runner.submit(Json{{"deploy", {{"device", "cpu"}}}});
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    auto p = runner.get(id);
    if (p && p->state == "QUEUED" && p->attempt == 2) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  auto before_cancel = runner.get(id);
  ASSERT_TRUE(before_cancel && before_cancel->state == "QUEUED" && before_cancel->attempt == 2)
      << "fixture did not reach a backed-off retry in time";
  ASSERT_TRUE(runner.cancel(id));
  const auto final_ = wait_for_terminal(runner, id, 2000);
  EXPECT_EQ(final_.state, "CANCELLED");
  EXPECT_EQ(final_.attempt, 2) << "cancel must not trigger yet another attempt";
}

// ---- merged_state: the retry-aware merge of events.jsonl with JobRunner's live view ----

TEST(job_runner_merged_state, a_stale_failed_event_from_an_earlier_attempt_does_not_mask_a_queued_retry) {
  JobProc p;
  p.state = "QUEUED";  // attempt 1 failed and was requeued; attempt 2 has not launched yet
  p.exit_code = 1;
  EXPECT_EQ(merged_state("FAILED", p), "QUEUED")
      << "events.jsonl still holds attempt 1's FAILED line; the live process view must win while retrying";
}

TEST(job_runner_merged_state, a_stale_failed_event_does_not_mask_a_running_retry) {
  JobProc p;
  p.state = "RUNNING";  // attempt 2 is now running
  p.exit_code = 1;
  EXPECT_EQ(merged_state("FAILED", p), "RUNNING");
}

TEST(job_runner_merged_state, cancelled_always_wins_even_over_a_succeeded_event) {
  JobProc p;
  p.state = "CANCELLED";
  EXPECT_EQ(merged_state("SUCCEEDED", p), "CANCELLED");
}

TEST(job_runner_merged_state, a_genuinely_terminal_event_is_reported_once_the_process_view_agrees) {
  JobProc p;
  p.state = "FAILED";
  p.exit_code = 1;
  EXPECT_EQ(merged_state("FAILED", p), "FAILED");
}

TEST(job_runner_merged_state, no_process_view_falls_back_to_the_events_file) {
  EXPECT_EQ(merged_state("SUCCEEDED", std::nullopt), "SUCCEEDED");
}

TEST(job_runner_merged_state, a_crashed_process_with_no_terminal_event_falls_back_to_its_exit_code) {
  JobProc p;
  p.state = "FAILED";  // the process exited non-zero before writing its own job_state event
  p.exit_code = 9;
  EXPECT_EQ(merged_state("UNKNOWN", p), "FAILED");
}
