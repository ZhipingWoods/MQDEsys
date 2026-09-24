#include "job_runner.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
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

JobRunner::JobRunner(fs::path runs_dir, fs::path repo_root, std::string python, std::vector<int> gpus)
    : runs_(std::move(runs_dir)), repo_(std::move(repo_root)), python_(std::move(python)), gpus_(std::move(gpus)) {
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
  {
    std::lock_guard lk(mu_);
    jobs_[id] = JobProc{.id = id, .state = "QUEUED", .submitted_at = now_s()};
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
  for (int g : gpus_) arr.push_back({{"gpu", g}, {"job", busy_.contains(g) ? Json(busy_.at(g)) : Json(nullptr)}});
  return {{"pool", arr}, {"queued", queue_.size()}};
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
    ::setenv("CUDA_VISIBLE_DEVICES", std::to_string(gpu).c_str(), 1);
    const char* old = std::getenv("PYTHONPATH");
    ::setenv("PYTHONPATH", old && *old ? (pythonpath + ":" + old).c_str() : pythonpath.c_str(), 1);
    ::setenv("MQDE_DATA_DIR", (repo_ / "data").c_str(), 1);
    ::setenv("PYTHONUNBUFFERED", "1", 1);
    ::execlp(python_.c_str(), python_.c_str(), "-m", "mqde.pipeline.vla_job", job_yaml.c_str(), "--job-id", j.id.c_str(),
             "--device", "cuda:0", static_cast<char*>(nullptr));
    std::fprintf(stderr, "exec %s failed: %s\n", python_.c_str(), std::strerror(errno));
    ::_exit(127);
  }
  j.pid = pid;
  j.gpu = gpu;
  j.state = pid > 0 ? "RUNNING" : "FAILED";
  j.started_at = now_s();
  if (pid > 0) busy_[gpu] = j.id;
  MQDE_LOG_INFO("job {} started on GPU {} (pid {})", j.id, gpu, pid);
}

void JobRunner::reap() {
  int status = 0;
  pid_t pid;
  while ((pid = ::waitpid(-1, &status, WNOHANG)) > 0) {
    for (auto& [id, j] : jobs_) {
      if (j.pid != pid) continue;
      j.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
      if (j.state != "CANCELLED") j.state = j.exit_code == 0 ? "SUCCEEDED" : "FAILED";
      j.finished_at = now_s();
      busy_.erase(j.gpu);
      MQDE_LOG_INFO("job {} finished: {} (exit {})", id, j.state, j.exit_code);
    }
  }
}

void JobRunner::loop() {
  std::unique_lock lk(mu_);
  while (!stop_) {
    reap();
    while (!queue_.empty()) {
      int free_gpu = -1;
      for (int g : gpus_)
        if (!busy_.contains(g)) {
          free_gpu = g;
          break;
        }
      if (free_gpu < 0) break;
      const auto id = queue_.front();
      queue_.pop_front();
      launch(jobs_[id], free_gpu);
    }
    cv_.wait_for(lk, std::chrono::milliseconds(500));
  }
  for (auto& [_, j] : jobs_)
    if (j.state == "RUNNING" && j.pid > 0) ::kill(-j.pid, SIGTERM);
}

}  // namespace mqde::server
