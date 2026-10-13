#include "backend_envs.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <random>

#include "mqde/common/log.h"

namespace mqde::server {
namespace fs = std::filesystem;
namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

bool alnum(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }

std::string new_id() {
  std::mt19937 rng(std::random_device{}());
  return fmt::format("{:08x}{:08x}", rng(), rng());
}

std::vector<char*> argv_of(std::vector<std::string>& v) {
  std::vector<char*> a;
  a.reserve(v.size() + 1);
  for (auto& s : v) a.push_back(s.data());
  a.push_back(nullptr);
  return a;
}

std::string last_line(const std::string& s) {
  std::string t = s;
  while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
  const auto p = t.find_last_of('\n');
  return p == std::string::npos ? t : t.substr(p + 1);
}

// Runs argv, returns its stdout; stderr is read too (for the error message). Kills the whole process group at
// the timeout.
Result<std::string> run_capture(std::vector<std::string> args, double timeout_s) {
  int out[2], err[2];
  if (::pipe(out) != 0 || ::pipe(err) != 0)
    return make_error(Errc::kIo, "pipe failed: {}", std::strerror(errno));
  const pid_t pid = ::fork();
  if (pid < 0) return make_error(Errc::kIo, "fork failed: {}", std::strerror(errno));
  if (pid == 0) {
    ::setpgid(0, 0);
    ::dup2(out[1], STDOUT_FILENO);
    ::dup2(err[1], STDERR_FILENO);
    ::close(out[0]);
    ::close(out[1]);
    ::close(err[0]);
    ::close(err[1]);
    const int dn = ::open("/dev/null", O_RDONLY);
    if (dn >= 0) ::dup2(dn, STDIN_FILENO);
    auto a = argv_of(args);
    ::execvp(a[0], a.data());
    std::fprintf(stderr, "exec %s failed: %s\n", a[0], std::strerror(errno));
    ::_exit(127);
  }
  ::close(out[1]);
  ::close(err[1]);
  std::string so, se;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  pollfd fds[2] = {{out[0], POLLIN, 0}, {err[0], POLLIN, 0}};
  bool timed_out = false;
  int open_fds = 2;
  while (open_fds > 0) {
    const auto left =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now())
            .count();
    if (left <= 0) {
      timed_out = true;
      break;
    }
    if (::poll(fds, 2, static_cast<int>(std::min<long>(left, 1000))) < 0 && errno != EINTR) break;
    for (int i = 0; i < 2; ++i) {
      if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP))) continue;
      char buf[4096];
      const ssize_t n = ::read(fds[i].fd, buf, sizeof buf);
      if (n > 0)
        (i == 0 ? so : se).append(buf, static_cast<size_t>(n));
      else {
        ::close(fds[i].fd);
        fds[i].fd = -1;
        --open_fds;
      }
    }
  }
  if (timed_out) ::kill(-pid, SIGKILL);
  for (auto& f : fds)
    if (f.fd >= 0) ::close(f.fd);
  int st = 0;
  ::waitpid(pid, &st, 0);
  if (timed_out)
    return make_error(Errc::kTimeout, "setup_backend.py did not answer within {:.0f}s", timeout_s);
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
    return make_error(Errc::kExternal, "setup_backend.py failed ({}): {}",
                      WIFEXITED(st) ? WEXITSTATUS(st) : -1,
                      last_line(se).empty() ? last_line(so) : last_line(se));
  return so;
}

}  // namespace

bool valid_backend_id(std::string_view id) { return id == "vllm" || id == "sglang" || id == "llamacpp"; }

bool valid_backend_version(std::string_view v) {
  if (v.empty()) return true;
  if (v.size() > 40 || !alnum(v[0])) return false;
  return std::all_of(v.begin(), v.end(), [](char c) {
    return alnum(c) || c == '.' || c == '+' || c == '!' || c == '_' || c == '-';
  });
}

BackendEnvs::BackendEnvs(std::vector<std::string> command, fs::path log_dir, double status_timeout_s)
    : command_(std::move(command)), log_dir_(std::move(log_dir)), status_timeout_s_(status_timeout_s) {}

BackendEnvs::~BackendEnvs() {
  {
    std::lock_guard lk(mu_);
    if (!running_.empty()) ::kill(-installs_.at(running_).pid, SIGTERM);
  }
  if (waiter_.joinable()) waiter_.join();
}

Result<Json> BackendEnvs::status(bool check) {
  auto args = command_;
  args.insert(args.end(), {"status", "--json"});
  if (check) args.emplace_back("--check");
  MQDE_ASSIGN_OR_RETURN(std::string out, run_capture(std::move(args), status_timeout_s_));
  // the script prints exactly one JSON line; take the last non-empty one in case a tool wrote something
  // before it
  const auto line = last_line(out);
  try {
    return Json::parse(line);
  } catch (const std::exception& e) {
    return make_error(Errc::kParse, "setup_backend.py status did not print JSON: {}", e.what());
  }
}

Result<std::string> BackendEnvs::start_install(const std::string& backend, const std::string& version) {
  if (!valid_backend_id(backend)) return make_error(Errc::kInvalidArgument, "unknown backend '{}'", backend);
  if (!valid_backend_version(version))
    return make_error(Errc::kInvalidArgument, "invalid version '{}'", version);
  std::lock_guard lk(mu_);
  if (!running_.empty())
    return make_error(Errc::kResourceExhausted, "an install is already running ({})",
                      installs_.at(running_).backend);
  std::error_code ec;
  fs::create_directories(log_dir_, ec);
  Install in;
  in.id = new_id();
  in.backend = backend;
  in.version = version;
  in.log = log_dir_ / fmt::format("install-{}.log", in.id);
  auto args = command_;
  args.insert(args.end(), {"install", backend});
  if (!version.empty()) args.insert(args.end(), {"--version", version});

  const pid_t pid = ::fork();
  if (pid < 0) return make_error(Errc::kIo, "fork failed: {}", std::strerror(errno));
  if (pid == 0) {
    ::setpgid(0, 0);
    const int fd = ::open(in.log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
      ::dup2(fd, STDOUT_FILENO);
      ::dup2(fd, STDERR_FILENO);
      ::close(fd);
    }
    const int dn = ::open("/dev/null", O_RDONLY);
    if (dn >= 0) ::dup2(dn, STDIN_FILENO);
    ::setenv("PYTHONUNBUFFERED", "1", 1);
    auto a = argv_of(args);
    ::execvp(a[0], a.data());
    std::fprintf(stderr, "exec %s failed: %s\n", a[0], std::strerror(errno));
    ::_exit(127);
  }
  in.pid = pid;
  in.started_at = now_s();
  const std::string id = in.id;
  installs_[id] = std::move(in);
  running_ = id;
  while (installs_.size() > 20) {  // keep the last 20; finished ones only
    auto it =
        std::find_if(installs_.begin(), installs_.end(), [&](auto& kv) { return kv.first != running_; });
    if (it == installs_.end()) break;
    std::error_code rm;
    fs::remove(it->second.log, rm);
    installs_.erase(it);
  }
  if (waiter_.joinable()) waiter_.join();  // the previous install already finished (running_ was empty)
  waiter_ = std::thread([this, id, pid] { wait_for(id, pid); });
  MQDE_LOG_INFO("backend install {} started: {} {}", id, backend, version);
  return id;
}

void BackendEnvs::wait_for(const std::string& id, pid_t pid) {
  int st = 0;
  while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {
  }
  std::lock_guard lk(mu_);
  auto& in = installs_[id];
  in.exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  in.state = in.exit_code == 0 ? "succeeded" : "failed";
  in.finished_at = now_s();
  if (running_ == id) running_.clear();
}

Json BackendEnvs::to_json(const Install& i) const {
  std::deque<std::string> tail;
  std::ifstream f(i.log);
  for (std::string line; std::getline(f, line);) {
    tail.push_back(line);
    if (tail.size() > 200) tail.pop_front();
  }
  return {{"id", i.id},
          {"backend", i.backend},
          {"version", i.version},
          {"state", i.state},
          {"exit_code", i.exit_code},
          {"started_at", i.started_at},
          {"finished_at", i.finished_at},
          {"log_tail", tail}};
}

std::optional<Json> BackendEnvs::install_status(const std::string& id) const {
  std::lock_guard lk(mu_);
  const auto it = installs_.find(id);
  if (it == installs_.end()) return std::nullopt;
  return to_json(it->second);
}

std::optional<Json> BackendEnvs::current_install() const {
  std::lock_guard lk(mu_);
  if (running_.empty()) return std::nullopt;
  return to_json(installs_.at(running_));
}

}  // namespace mqde::server
