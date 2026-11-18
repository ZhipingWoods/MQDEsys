#pragma once
// Launches Python executor jobs as child processes on a GPU pool and tracks their lifecycle.

#include <sys/types.h>

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "mqde/common/config.h"

namespace mqde::server {

struct JobProc {
  std::string id;
  std::string state;  // QUEUED | RUNNING | SUCCEEDED | FAILED | CANCELLED
  int gpu = -1;       // -1 while queued, and for CPU jobs
  bool cpu = false;   // no GPU slot needed: deploy.device == "cpu" and neither quant_device nor
                      // evaluation.device (below) asks for one either
  std::string module = "mqde.pipeline.vla_job";  // model.type: llm/vlm/world|ewm -> matching python module
  // T-1304/T-1306: top-level quant_device / evaluation.device, read verbatim from the submitted job (same
  // "cpu"|"gpu" symbols the frontend's single device selector already sends as deploy.device) -- unset
  // when the job did not request a split, in which case launch() passes only the one --device flag it
  // always has, exactly as before this existed.
  std::optional<std::string> quant_device, eval_device;
  pid_t pid = -1;
  int exit_code = 0;
  double submitted_at = 0, started_at = 0, finished_at = 0;
  int attempt = 1;         // 1-based: which launch this is
  int max_attempts = 3;    // stop retrying once attempt reaches this
  double retry_after = 0;  // while QUEUED again after a failure: epoch seconds before it may be relaunched
};

// Merges the worker-written `events.jsonl` state (`from_events`) with JobRunner's own live process view
// (`p`). A terminal line in events.jsonl (SUCCEEDED/FAILED) only wins when the process view agrees the job
// has actually finished: once a failed attempt is requeued for retry, `p->state` goes back to QUEUED (or
// RUNNING once the next attempt starts), and that must be reported over a stale terminal line left behind
// by the attempt that already failed. CANCELLED always wins.
std::string merged_state(const std::string& from_events, const std::optional<JobProc>& p);

class JobRunner {
 public:
  // max_attempts: total launch attempts per job before giving up (1 = no retry). base_backoff_s: delay
  // before attempt N+1 is (attempt - 1)-th power-of-two multiples of this, e.g. with the default 2.0s:
  // attempt 2 waits 2s, attempt 3 waits 4s. There is no per-error-code retry classification yet (the Python
  // workers don't emit one): any non-zero exit that wasn't a user cancellation is retried the same way.
  JobRunner(std::filesystem::path runs_dir, std::filesystem::path repo_root, std::string python,
            std::vector<int> gpus, int max_attempts = 3, double base_backoff_s = 2.0);
  ~JobRunner();

  // Writes runs/<id>/job.yaml and enqueues it. Returns the job id.
  std::string submit(const Json& job_yaml_json);
  bool cancel(const std::string& id);
  std::optional<JobProc> get(const std::string& id) const;
  std::vector<JobProc> list() const;
  Json gpu_status() const;

 private:
  void loop();
  void launch(JobProc& j, int gpu);
  void reap();
  double backoff_seconds(int next_attempt) const;

  std::filesystem::path runs_;
  std::filesystem::path repo_;
  std::string python_;
  std::vector<int> gpus_;
  int max_attempts_;
  double base_backoff_s_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::string> queue_;
  std::map<std::string, JobProc> jobs_;
  std::map<int, std::string> busy_;  // gpu -> job id
  std::string cpu_busy_;             // job id holding the CPU slot
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace mqde::server
