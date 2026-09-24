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
  int gpu = -1;
  pid_t pid = -1;
  int exit_code = 0;
  double submitted_at = 0, started_at = 0, finished_at = 0;
};

class JobRunner {
 public:
  JobRunner(std::filesystem::path runs_dir, std::filesystem::path repo_root, std::string python, std::vector<int> gpus);
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

  std::filesystem::path runs_;
  std::filesystem::path repo_;
  std::string python_;
  std::vector<int> gpus_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::string> queue_;
  std::map<std::string, JobProc> jobs_;
  std::map<int, std::string> busy_;  // gpu -> job id
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace mqde::server
