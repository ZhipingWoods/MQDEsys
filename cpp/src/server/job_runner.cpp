#include "job_runner.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>

#include <yaml-cpp/yaml.h>

#include "mqde/common/log.h"

namespace mqde::server {
namespace fs = std::filesystem;
namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string new_job_id() {
  const auto t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
  std::mt19937 rng(std::random_device{}());
  return fmt::format("{}-{:06x}", buf, rng() & 0xffffff);
}

YAML::Node to_yaml(const Json& j) {
  if (j.is_object()) {
    YAML::Node n(YAML::NodeType::Map);
    for (auto it = j.begin(); it != j.end(); ++it) n[it.key()] = to_yaml(*it);
    return n;
  }
  if (j.is_array()) {
    YAML::Node n(YAML::NodeType::Sequence);
    for (const auto& e : j) n.push_back(to_yaml(e));
    return n;
  }
  if (j.is_string()) return YAML::Node(j.get<std::string>());
  if (j.is_boolean()) return YAML::Node(j.get<bool>());
  if (j.is_number_integer()) return YAML::Node(j.get<int64_t>());
  if (j.is_number()) return YAML::Node(j.get<double>());
  return YAML::Node();
}

}  // namespace

std::string merged_state(const std::string& from_events, const std::optional<JobProc>& p) {
  if (!p) return from_events;
  if (p->state == "CANCELLED") return "CANCELLED";
  // Mid-retry (requeued after a failed attempt) or already relaunched: the live process view is ground
  // truth, since events.jsonl may still hold the previous attempt's terminal SUCCEEDED/FAILED line.
  if (p->state == "QUEUED" || p->state == "RUNNING") return p->state;
  const bool terminal = from_events == "SUCCEEDED" || from_events == "FAILED";
  if (terminal) return from_events;
  if (p->state == "SUCCEEDED" || p->state == "FAILED") return p->exit_code == 0 ? "SUCCEEDED" : "FAILED";
  return p->state;
}

JobRunner::JobRunner(fs::path runs_dir, fs::path repo_root, std::string python, std::vector<int> gpus,
                      int max_attempts, double base_backoff_s)
    : runs_(std::move(runs_dir)),
      repo_(std::move(repo_root)),
      python_(std::move(python)),
      gpus_(std::move(gpus)),
      max_attempts_(max_attempts),
      base_backoff_s_(base_backoff_s) {
  fs::create_directories(runs_);
  thread_ = std::thread([this] { loop(); });
}

JobRunner::~JobRunner() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

std::string JobRunner::submit(const Json& job) {
  const std::string id = new_job_id();
  const auto dir = runs_ / id;
  fs::create_directories(dir);
  Json j = job;
  j["output_dir"] = fs::absolute(runs_).string();
  YAML::Emitter out;
  out << to_yaml(j);
  std::ofstream(dir / "job.yaml") << out.c_str() << "\n";
  JobProc proc;
  proc.id = id;
  proc.state = "QUEUED";
  proc.submitted_at = now_s();
  proc.max_attempts = max_attempts_;
  if (const auto& qd = job.value("/quant_device"_json_pointer, Json()); qd.is_string()) proc.quant_device = qd.get<std::string>();
  if (const auto& ed = job.value("/evaluation/device"_json_pointer, Json()); ed.is_string()) proc.eval_device = ed.get<std::string>();
  const bool wants_gpu = (proc.quant_device && *proc.quant_device == "gpu") || (proc.eval_device && *proc.eval_device == "gpu");
  proc.cpu = job.value("/deploy/device"_json_pointer, std::string()) == "cpu" && !wants_gpu;
  if (job.value("/model/type"_json_pointer, std::string()) == "llm") proc.module = "mqde.pipeline.llm_job";
  {
    std::lock_guard lk(mu_);
    jobs_[id] = proc;
    queue_.push_back(id);
  }
  cv_.notify_all();
  MQDE_LOG_INFO("job {} queued", id);
  return id;
}

bool JobRunner::cancel(const std::string& id) {
  std::lock_guard lk(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return false;
  if (it->second.state == "QUEUED") {
    std::erase(queue_, id);
    it->second.state = "CANCELLED";
    it->second.finished_at = now_s();
    return true;
  }
  if (it->second.state == "RUNNING" && it->second.pid > 0) {
    ::kill(-it->second.pid, SIGTERM);  // whole process group
    it->second.state = "CANCELLED";
    return true;
  }
  return false;
}

std::optional<JobProc> JobRunner::get(const std::string& id) const {
  std::lock_guard lk(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return std::nullopt;
  return it->second;
}

std::vector<JobProc> JobRunner::list() const {
  std::lock_guard lk(mu_);
  std::vector<JobProc> out;
  for (const auto& [_, j] : jobs_) out.push_back(j);
  return out;
}

Json JobRunner::gpu_status() const {
  std::lock_guard lk(mu_);
  Json arr = Json::array();
  for (int g : gpus_)
    arr.push_back({{"gpu", g}, {"job", busy_.contains(g) ? Json(busy_.at(g)) : Json(nullptr)}});
  return {
      {"pool", arr}, {"cpu", cpu_busy_.empty() ? Json(nullptr) : Json(cpu_busy_)}, {"queued", queue_.size()}};
}

void JobRunner::launch(JobProc& j, int gpu) {
  const auto dir = runs_ / j.id;
  const std::string job_yaml = (dir / "job.yaml").string();
  const std::string log_path = (dir / "worker.log").string();
  const std::string pythonpath = (repo_ / "python").string();
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::setpgid(0, 0);
    const int fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
      ::dup2(fd, STDOUT_FILENO);
      ::dup2(fd, STDERR_FILENO);
      ::close(fd);
    }
    ::setenv("CUDA_VISIBLE_DEVICES", gpu >= 0 ? std::to_string(gpu).c_str() : "", 1);
    const char* old = std::getenv("PYTHONPATH");
    ::setenv("PYTHONPATH", old && *old ? (pythonpath + ":" + old).c_str() : pythonpath.c_str(), 1);
    ::setenv("MQDE_DATA_DIR", (repo_ / "data").c_str(), 1);
    ::setenv("PYTHONUNBUFFERED", "1", 1);
    // "gpu" (the only other symbol the frontend's device selectors send) resolves the same way --device
    // always has: the one slot this launch was actually given, or "cpu" if none was (which wants_gpu at
    // submit() time should have prevented, but a literal real device string -- a direct, non-server
    // invocation's job.yaml -- is passed through unchanged either way).
    auto resolve = [&](const std::string& sym) { return sym == "gpu" ? std::string(gpu >= 0 ? "cuda:0" : "cpu") : sym; };
    std::vector<std::string> args = {python_, "-m", j.module, job_yaml, "--job-id", j.id, "--device",
                                      gpu >= 0 ? "cuda:0" : "cpu"};
    if (j.quant_device) {
      args.emplace_back("--quant-device");
      args.push_back(resolve(*j.quant_device));
    }
    if (j.eval_device) {
      args.emplace_back("--eval-device");
      args.push_back(resolve(*j.eval_device));
    }
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    ::execvp(python_.c_str(), argv.data());
    std::fprintf(stderr, "exec %s failed: %s\n", python_.c_str(), std::strerror(errno));
    ::_exit(127);
  }
  j.pid = pid;
  j.gpu = gpu;
  j.state = pid > 0 ? "RUNNING" : "FAILED";
  j.started_at = now_s();
  if (pid > 0 && gpu >= 0) busy_[gpu] = j.id;
  if (pid > 0 && gpu < 0) cpu_busy_ = j.id;
  MQDE_LOG_INFO("job {} started on {} (pid {}, {})", j.id, gpu >= 0 ? fmt::format("GPU {}", gpu) : "CPU", pid,
                j.module);
}

double JobRunner::backoff_seconds(int next_attempt) const {
  // next_attempt is the attempt about to run (2, 3, ...): attempt 2 waits 1x base, attempt 3 waits 2x, ...
  return base_backoff_s_ * std::pow(2.0, static_cast<double>(next_attempt - 2));
}

void JobRunner::reap() {
  int status = 0;
  pid_t pid;
  while ((pid = ::waitpid(-1, &status, WNOHANG)) > 0) {
    for (auto& [id, j] : jobs_) {
      if (j.pid != pid) continue;
      j.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
      if (j.gpu >= 0) busy_.erase(j.gpu);
      if (cpu_busy_ == id) cpu_busy_.clear();
      j.pid = -1;
      if (j.state == "CANCELLED") {
        j.finished_at = now_s();
        MQDE_LOG_INFO("job {} finished: CANCELLED (exit {})", id, j.exit_code);
      } else if (j.exit_code == 0) {
        j.state = "SUCCEEDED";
        j.finished_at = now_s();
        MQDE_LOG_INFO("job {} finished: SUCCEEDED (exit 0)", id);
      } else if (j.attempt < j.max_attempts) {
        ++j.attempt;
        j.retry_after = now_s() + backoff_seconds(j.attempt);
        j.state = "QUEUED";
        queue_.push_back(id);
        MQDE_LOG_INFO("job {} failed (exit {}), retrying as attempt {}/{} after {:.1f}s", id, j.exit_code,
                      j.attempt, j.max_attempts, backoff_seconds(j.attempt));
      } else {
        j.state = "FAILED";
        j.finished_at = now_s();
        MQDE_LOG_INFO("job {} finished: FAILED (exit {}, attempt {}/{}, giving up)", id, j.exit_code,
                      j.attempt, j.max_attempts);
      }
    }
  }
}

void JobRunner::loop() {
  std::unique_lock lk(mu_);
  while (!stop_) {
    reap();
    const double now = now_s();
    // FIFO per resource: a CPU job does not wait behind GPU jobs that cannot start yet, and vice versa. A
    // job retrying after a failure is skipped (not launched, not blocking anything behind it) until its
    // backoff elapses.
    for (auto it = queue_.begin(); it != queue_.end();) {
      JobProc& j = jobs_[*it];
      if (j.retry_after > now) {
        ++it;
        continue;
      }
      int slot = -2;
      if (j.cpu) {
        if (cpu_busy_.empty()) slot = -1;
      } else {
        for (int g : gpus_)
          if (!busy_.contains(g)) {
            slot = g;
            break;
          }
      }
      if (slot == -2) {
        ++it;
        continue;
      }
      it = queue_.erase(it);
      launch(j, slot);
    }
    cv_.wait_for(lk, std::chrono::milliseconds(500));
  }
  for (auto& [_, j] : jobs_)
    if (j.state == "RUNNING" && j.pid > 0) ::kill(-j.pid, SIGTERM);
}

}  // namespace mqde::server
