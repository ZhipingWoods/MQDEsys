#pragma once
// The server side of "install / update vLLM, SGLang, llama.cpp": runs scripts/setup_backend.py (stdlib-only Python,
// python/mqde/backend/envs.py) as a child process. The backend id and version are validated here before anything is
// executed and passed as separate argv entries -- there is no shell and no way to name a different package.
// One install runs at a time; its output goes to a log file whose tail is served as progress.

#include <sys/types.h>

#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "mqde/common/config.h"
#include "mqde/common/result.h"

namespace mqde::server {

bool valid_backend_id(std::string_view id);       // exactly vllm | sglang | llamacpp
bool valid_backend_version(std::string_view v);   // empty (= newest) or [0-9A-Za-z][0-9A-Za-z.+!_-]{0,39}

class BackendEnvs {
 public:
  // `command` is the argv prefix that runs the setup script, e.g. {python, "<repo>/scripts/setup_backend.py"}.
  BackendEnvs(std::vector<std::string> command, std::filesystem::path log_dir, double status_timeout_s = 120);
  ~BackendEnvs();  // stops a running install (its whole process group) instead of orphaning it

  // `status --json [--check]`: installed version / location per backend, and with `check` the newest version.
  Result<Json> status(bool check);
  // Starts `install <backend> [--version <v>]`. kInvalidArgument for a bad id/version, kResourceExhausted while
  // another install is running. Returns the install id.
  Result<std::string> start_install(const std::string& backend, const std::string& version);
  std::optional<Json> install_status(const std::string& id) const;  // state, exit_code, last 200 log lines
  std::optional<Json> current_install() const;                       // the running one, if any

 private:
  struct Install {
    std::string id, backend, version, state = "running";
    pid_t pid = -1;
    int exit_code = -1;
    double started_at = 0, finished_at = 0;
    std::filesystem::path log;
  };
  Json to_json(const Install& i) const;
  void wait_for(const std::string& id, pid_t pid);

  std::vector<std::string> command_;
  std::filesystem::path log_dir_;
  double status_timeout_s_;
  mutable std::mutex mu_;
  std::map<std::string, Install> installs_;
  std::string running_;
  std::thread waiter_;
};

}  // namespace mqde::server
