// BackendEnvs: the server side of "update vLLM / SGLang / llama.cpp" -- validated, non-shell invocation of
// scripts/setup_backend.py, one install at a time, progress in a log. Driven by a fake script, no packages
// touched.
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "backend_envs.h"

using namespace mqde;
using namespace mqde::server;
namespace fs = std::filesystem;

namespace {

fs::path fresh_dir(const char* name) {
  auto d = fs::temp_directory_path() /
           (std::string("mqde_backend_envs_") + name + "_" + std::to_string(::getpid()));
  fs::remove_all(d);
  fs::create_directories(d);
  return d;
}

// Records every invocation's arguments (one line each) in $FAKE_LOG, then behaves per the first argument.
fs::path fake_script(const fs::path& dir) {
  const auto p = dir / "setup_backend.sh";
  std::ofstream(p)
      << "#!/bin/sh\n"
         "echo \"$@\" >> \"$FAKE_LOG\"\n"
         "echo $$ > \"$FAKE_LOG.pid\"\n"
         "case \"$1\" in\n"
         "  status) [ -n \"$FAKE_STATUS_FAIL\" ] && { echo 'boom: uv exploded' >&2; exit 1; }\n"
         "          sleep \"${FAKE_STATUS_SLEEP:-0}\"\n"
         "          echo 'warning on stdout first'\n"
         "          echo '[{\"id\":\"vllm\",\"installed\":true,\"version\":\"0.30.0\"}]' ;;\n"
         "  install) echo \"installing $2\"; echo 'to stderr' >&2\n"
         "           i=0; while [ $i -lt \"${FAKE_LINES:-0}\" ]; do echo \"line $i\"; i=$((i+1)); done\n"
         "           sleep \"${FAKE_SLEEP:-0}\"; exit \"${FAKE_EXIT:-0}\" ;;\n"
         "esac\n";
  ::chmod(p.c_str(), 0755);
  return p;
}

struct Env {
  fs::path dir, script, log;
  Env(const char* name) : dir(fresh_dir(name)), script(fake_script(dir)), log(dir / "calls.log") {
    ::setenv("FAKE_LOG", log.c_str(), 1);
    for (const char* k : {"FAKE_STATUS_FAIL", "FAKE_STATUS_SLEEP", "FAKE_LINES", "FAKE_SLEEP", "FAKE_EXIT"})
      ::unsetenv(k);
  }
  std::string calls() const {
    std::ifstream f(log);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
  }
};

Json wait_done(BackendEnvs& b, const std::string& id, int timeout_ms = 8000) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    auto s = b.install_status(id);
    if (s && s->at("state") != "running") return *s;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  ADD_FAILURE() << "install " << id << " did not finish";
  return *b.install_status(id);
}

}  // namespace

TEST(backend_envs, status_returns_the_scripts_json_and_asks_for_the_network_check_only_when_told_to) {
  Env e("status");
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto s = b.status(false);
  ASSERT_TRUE(s) << s.error().message;
  EXPECT_EQ(s->at(0).at("version"), "0.30.0") << "the JSON is found even after a stray line on stdout";
  ASSERT_TRUE(b.status(true));
  EXPECT_EQ(e.calls(), "status --json\nstatus --json --check\n");
}

TEST(backend_envs, a_failing_status_reports_the_scripts_own_error_text) {
  Env e("status_fail");
  ::setenv("FAKE_STATUS_FAIL", "1", 1);
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto s = b.status(false);
  ASSERT_FALSE(s);
  EXPECT_NE(s.error().message.find("boom: uv exploded"), std::string::npos) << s.error().message;
}

TEST(backend_envs, a_status_that_hangs_is_killed_at_the_timeout) {
  Env e("status_timeout");
  ::setenv("FAKE_STATUS_SLEEP", "30", 1);
  BackendEnvs b({e.script.string()}, e.dir / "logs", /*status_timeout_s=*/1.0);
  const auto t0 = std::chrono::steady_clock::now();
  auto s = b.status(false);
  ASSERT_FALSE(s);
  EXPECT_EQ(s.error().code, Errc::kTimeout);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(10));
}

TEST(backend_envs, hostile_backend_ids_and_versions_are_rejected_and_nothing_runs) {
  Env e("hostile");
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  for (const char* backend : {"pip", "vllm; rm -rf /", "../vllm", "", "VLLM"}) {
    auto r = b.start_install(backend, "");
    ASSERT_FALSE(r) << backend;
    EXPECT_EQ(r.error().code, Errc::kInvalidArgument) << backend;
  }
  for (const char* version : {"1.0; rm -rf /", "$(id)", "1 2", "../x", "-1", "==1.0", "1.0\n",
                              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}) {
    auto r = b.start_install("vllm", version);
    ASSERT_FALSE(r) << version;
    EXPECT_EQ(r.error().code, Errc::kInvalidArgument) << version;
  }
  EXPECT_EQ(e.calls(), "") << "validation happens before anything is executed";
}

TEST(backend_envs, valid_ids_and_versions) {
  for (const char* v : {"0.31.0", "b11565", "0.5.22.dev586+g86624f14e", "1!2.0"})
    EXPECT_TRUE(valid_backend_version(v)) << v;
  EXPECT_TRUE(valid_backend_version("")) << "no version = the newest";
  for (const char* id : {"vllm", "sglang", "llamacpp"}) EXPECT_TRUE(valid_backend_id(id)) << id;
}

TEST(backend_envs, an_install_runs_the_script_with_exactly_these_arguments_and_succeeds) {
  Env e("install_ok");
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto id = b.start_install("vllm", "0.31.0");
  ASSERT_TRUE(id) << id.error().message;
  const auto s = wait_done(b, *id);
  EXPECT_EQ(s.at("state"), "succeeded");
  EXPECT_EQ(s.at("exit_code"), 0);
  EXPECT_EQ(s.at("backend"), "vllm");
  EXPECT_EQ(s.at("version"), "0.31.0");
  EXPECT_EQ(e.calls(), "install vllm --version 0.31.0\n");
  const auto tail = s.at("log_tail").dump();
  EXPECT_NE(tail.find("installing vllm"), std::string::npos) << tail;
  EXPECT_NE(tail.find("to stderr"), std::string::npos) << "stderr is part of the log, like uv's own messages";
}

TEST(backend_envs, no_version_means_no_version_argument) {
  Env e("no_version");
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto id = b.start_install("llamacpp", "");
  ASSERT_TRUE(id);
  wait_done(b, *id);
  EXPECT_EQ(e.calls(), "install llamacpp\n");
}

TEST(backend_envs, a_failed_install_is_reported_with_its_exit_code_and_log) {
  Env e("install_fail");
  ::setenv("FAKE_EXIT", "3", 1);
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto id = b.start_install("sglang", "0.5.21");
  ASSERT_TRUE(id);
  const auto s = wait_done(b, *id);
  EXPECT_EQ(s.at("state"), "failed");
  EXPECT_EQ(s.at("exit_code"), 3);
  EXPECT_NE(s.at("log_tail").dump().find("installing sglang"), std::string::npos);
}

TEST(backend_envs, only_one_install_runs_at_a_time) {
  Env e("one_at_a_time");
  ::setenv("FAKE_SLEEP", "30", 1);
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto first = b.start_install("vllm", "0.31.0");
  ASSERT_TRUE(first);
  auto second = b.start_install("sglang", "0.5.21");
  ASSERT_FALSE(second);
  EXPECT_EQ(second.error().code, Errc::kResourceExhausted);
  EXPECT_NE(second.error().message.find("vllm"), std::string::npos) << "says which install is in the way";
  ASSERT_TRUE(b.current_install());
  EXPECT_EQ(b.current_install()->at("id"), *first);
  for (int i = 0; i < 150 && e.calls().empty(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let the first script start
  EXPECT_EQ(e.calls(), "install vllm --version 0.31.0\n") << "the refused request ran nothing";
}

TEST(backend_envs, after_one_finishes_the_next_is_accepted) {
  Env e("sequential");
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto a = b.start_install("vllm", "0.31.0");
  ASSERT_TRUE(a);
  wait_done(b, *a);
  EXPECT_FALSE(b.current_install());
  auto c = b.start_install("sglang", "0.5.21");
  ASSERT_TRUE(c) << c.error().message;
  EXPECT_NE(*a, *c);
  wait_done(b, *c);
  EXPECT_EQ(b.install_status(*a)->at("state"), "succeeded") << "finished installs stay inspectable";
}

TEST(backend_envs, an_unknown_install_id_is_not_found) {
  Env e("unknown_id");
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  EXPECT_FALSE(b.install_status("nope"));
  EXPECT_FALSE(b.current_install());
}

TEST(backend_envs, the_log_tail_is_capped_to_the_last_200_lines) {
  Env e("tail");
  ::setenv("FAKE_LINES", "1000", 1);
  BackendEnvs b({e.script.string()}, e.dir / "logs");
  auto id = b.start_install("vllm", "");
  ASSERT_TRUE(id);
  const auto s = wait_done(b, *id);
  ASSERT_TRUE(s.at("log_tail").is_array());
  EXPECT_EQ(s.at("log_tail").size(), 200u);
  EXPECT_EQ(s.at("log_tail").back(), "line 999");
}

TEST(backend_envs, destroying_the_manager_stops_a_running_install) {
  Env e("destroy");
  ::setenv("FAKE_SLEEP", "60", 1);
  pid_t pid = 0;
  {
    BackendEnvs b({e.script.string()}, e.dir / "logs");
    ASSERT_TRUE(b.start_install("vllm", ""));
    for (int i = 0; i < 100 && !fs::exists(e.log.string() + ".pid"); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::ifstream(e.log.string() + ".pid") >> pid;
    ASSERT_GT(pid, 0);
    ASSERT_EQ(::kill(pid, 0), 0) << "the install is running";
  }
  for (int i = 0; i < 100 && ::kill(pid, 0) == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_NE(::kill(pid, 0), 0) << "no orphaned installer left behind when the server stops";
}
