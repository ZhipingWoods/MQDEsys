#pragma once
// mqde-server: REST API + Server-Sent Events over cpp-httplib (the architecture specification §6.14).
// The filesystem (runs/<job_id>/{job.yaml, events.jsonl, report.json}) is the source of truth for
// jobs, so the server is restartable and the CLI and server see the same jobs.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "mqde/common/result.h"

namespace mqde::server {

struct ServerOptions {
  std::string host = "127.0.0.1";  // loopback by default; reach it via `ssh -L`
  int port = 8080;
  std::filesystem::path runs_dir = "runs";
  std::filesystem::path web_dir;    // built frontend (web/dist); empty = API only
  std::filesystem::path repo_root;  // for PYTHONPATH=<repo>/python
  std::string python = "python";
  std::vector<int> gpus;                   // GPU pool handed to jobs via CUDA_VISIBLE_DEVICES
  std::vector<std::string> allowed_roots;  // model / dataset paths must live under one of these
  // T-1303: seconds between automatic HF catalog refreshes while this process runs. 0 (default) disables
  // it -- existing deployments and every test that starts a server keep today's behavior (catalog only
  // changes via `mqde catalog update` or a client-triggered /download) unless this is explicitly set.
  int catalog_refresh_interval_s = 0;
};

// Blocks until the server stops.
Result<void> serve(const ServerOptions& opts);

}  // namespace mqde::server
