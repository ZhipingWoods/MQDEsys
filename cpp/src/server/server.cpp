#include "mqde/server/server.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <thread>

#include <fmt/ranges.h>
#include <httplib.h>

#include "advice.h"
#include "gguf.h"
#include "backend_envs.h"
#include "job_runner.h"
#include "mqde/common/log.h"
#include "mqde/estimator/estimator.h"
#include "mqde/fit/claim.h"
#include "mqde/fit/concurrency.h"
#include "mqde/fit/fit.h"
#include "mqde/fit/installed.h"
#include "mqde/hal/mqde_hal.h"
#include "mqde/hub/catalog.h"
#include "mqde/hub/hf.h"
#include "mqde/planner/validate.h"
#include "mqde/quant/metrics.h"
#include "mqde/quant/ops.h"
#include "mqde_probe.h"

namespace mqde::server {
namespace fs = std::filesystem;
namespace {

void send_json(httplib::Response& res, const Json& j, int status = 200) {
  res.status = status;
  res.set_content(j.dump(), "application/json");
}

void send_error(httplib::Response& res, int status, std::string_view code, const std::string& msg) {
  send_json(res, {{"code", code}, {"message", msg}}, status);
}

void send_error(httplib::Response& res, const Error& e) {
  const int status = e.code == Errc::kNotFound                                    ? 404
                     : e.code == Errc::kInvalidArgument || e.code == Errc::kParse ? 400
                                                                                  : 500;
  send_error(res, status, errc_name(e.code), e.message);
}

std::optional<Json> parse_body(const httplib::Request& req, httplib::Response& res) {
  try {
    return Json::parse(req.body.empty() ? "{}" : req.body);
  } catch (const std::exception& e) {
    send_error(res, 400, "parse", std::string("body is not valid JSON: ") + e.what());
    return std::nullopt;
  }
}

bool path_allowed(const std::vector<std::string>& roots, const std::string& p) {
  // Built-in datasets resolved by the executor from the local HF cache (not filesystem paths).
  if (p == "synthetic" || p == "wikitext-2-raw-v1:train" || p == "wikitext-2-raw-v1:test") return true;
  std::error_code ec;
  const auto canon = fs::weakly_canonical(p, ec).string();
  if (ec) return false;
  for (const auto& r : roots) {
    const auto rc = fs::weakly_canonical(r, ec).string();
    if (!ec && (canon == rc || canon.starts_with(rc + "/"))) return true;
  }
  return false;
}

bool valid_job_id(const std::string& id) {
  if (id.empty() || id.size() > 64) return false;
  for (char c : id)
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) return false;
  return true;
}

Json read_json_file(const fs::path& p) {
  std::ifstream in(p);
  if (!in) return nullptr;
  try {
    return Json::parse(in);
  } catch (...) {
    return nullptr;
  }
}

// Last `job_state` and per-stage status from events.jsonl (cheap enough for listing).
Json summarize_events(const fs::path& p) {
  std::ifstream in(p);
  Json s = {{"state", "UNKNOWN"}, {"stage", nullptr}, {"n_events", 0}};
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    ++n;
    try {
      auto ev = Json::parse(line);
      const auto type = ev.value("type", "");
      if (type == "job_state") s["state"] = ev.value("state", "UNKNOWN");
      if (type == "stage_started") s["stage"] = ev.value("stage", "");
      if (n == 1) s["started_ts"] = ev.value("ts", 0.0);
      s["last_ts"] = ev.value("ts", 0.0);
    } catch (...) {
    }
  }
  s["n_events"] = n;
  return s;
}

estimator::RuntimeParams runtime_from(const Json& j) {
  estimator::RuntimeParams rt;
  rt.concurrency = j.value("concurrency", rt.concurrency);
  rt.images_per_step = j.value("images_per_step", rt.images_per_step);
  rt.text_tokens = j.value("text_tokens", rt.text_tokens);
  rt.context_tokens = j.value("context_tokens", rt.context_tokens);
  rt.image_resolution = j.value("image_resolution", rt.image_resolution);
  rt.gpu_memory_utilization = j.value("gpu_memory_utilization", rt.gpu_memory_utilization);
  rt.extra_modalities = j.value("extra_modalities", rt.extra_modalities);
  return rt;
}

}  // namespace

Result<void> serve(const ServerOptions& o) {
  httplib::Server svr;
  JobRunner runner(o.runs_dir, o.repo_root, o.python, o.gpus);

  // Install / update of the vLLM, SGLang and llama.cpp environments (scripts/setup_backend.py). The script is stdlib-only
  // Python run by the executor's interpreter; MQDE_SETUP_BACKEND_BIN replaces the whole command (tests use a fake).
  // Logs live outside runs_dir: every directory there is listed as a job.
  std::vector<std::string> setup_cmd;
  if (const char* b = std::getenv("MQDE_SETUP_BACKEND_BIN"); b && *b) setup_cmd = {b};
  else setup_cmd = {o.python, (o.repo_root / "scripts" / "setup_backend.py").string()};
  BackendEnvs backend_envs(setup_cmd, o.repo_root / "build" / "backend-installs");

  // T-1303: background HF catalog refresh. refresh_fn wraps the same hub::update_catalog the CLI's
  // `mqde catalog update` uses, so a running server stays current without a human running that command.
  // catalog_refresh_interval_s == 0 means CatalogRefresher starts no thread at all (see its constructor).
  hub::CatalogRefresher catalog_refresher(
      hub::default_catalog_path(), std::chrono::seconds(o.catalog_refresh_interval_s),
      [](hub::Catalog base) -> Result<hub::Catalog> {
        return hub::update_catalog(hub::HfOptions{}, hub::UpdateOptions{}, std::move(base));
      });
  std::mutex spec_mu;
  std::map<std::string, modelspec::ModelSpec> spec_cache;  // model path -> spec

  // HF catalog (T-R25), reloaded when data/models/catalog.json changes.
  std::mutex catalog_mu;
  hub::Catalog catalog;
  std::filesystem::file_time_type catalog_mtime{};
  auto catalog_snapshot = [&]() -> hub::Catalog {
    std::lock_guard lk(catalog_mu);
    const auto p = hub::default_catalog_path();
    std::error_code ec;
    const auto mt = std::filesystem::last_write_time(p, ec);
    if (!ec && mt != catalog_mtime) {
      if (auto c = hub::load_catalog(p))
        catalog = std::move(*c), catalog_mtime = mt;
      else
        MQDE_LOG_WARN("catalog: {}", c.error().to_string());
    }
    return catalog;
  };
  // "hf:<repo>" names a catalog entry (no local file access); anything else must be under an allowed root.
  auto model_allowed = [&](const std::string& m) {
    return m.starts_with("hf:") || path_allowed(o.allowed_roots, m);
  };

  // need_tensors: the catalog only has config-level metadata; structure views need the real tensor list.
  auto get_spec = [&](const std::string& path, bool need_tensors = false) -> Result<modelspec::ModelSpec> {
    if (path.starts_with("hf:")) {
      const auto c = catalog_snapshot();
      if (const auto* e = hub::find(c, path); e && !need_tensors) return hub::to_spec(*e);
      // Not in the catalog: read config.json + safetensors headers from the Hub (no weights), cached.
      {
        std::lock_guard lk(spec_mu);
        if (auto it = spec_cache.find(path); it != spec_cache.end()) return it->second;
      }
      auto s = hub::inspect_remote(hub::HfOptions{}, path.substr(3));
      if (!s)
        return std::unexpected(
            s.error().with_context("model " + path + " is not in the catalog; Hub lookup failed"));
      std::lock_guard lk(spec_mu);
      spec_cache[path] = *s;
      return s;
    }
    std::lock_guard lk(spec_mu);
    if (auto it = spec_cache.find(path); it != spec_cache.end()) return it->second;
    auto s = modelspec::inspect_local(path);
    if (s) spec_cache[path] = *s;
    return s;
  };

  svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
    if (!req.path.starts_with("/api/v1/jobs/") || !req.path.ends_with("/stream"))
      MQDE_LOG_DEBUG("{} {} -> {}", req.method, req.path, res.status);
  });
  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string msg = "unknown error";
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      msg = e.what();
    } catch (...) {
    }
    send_error(res, 500, "internal", msg);
  });

  // ---------------- system ----------------
  svr.Get("/api/v1/health", [](const httplib::Request&, httplib::Response& res) {
    send_json(res, {{"status", "ok"}, {"version", MQDE_VERSION}});
  });
  // Prometheus text exposition format (T-1205). Counters/gauges come from the in-process JobRunner, which is
  // the live view of jobs this process launched; a job submitted to a different server process (or before a
  // restart) is not reflected here (the filesystem under runs_dir is the durable source of truth, this is a
  // cheap live view).
  const double start_time =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  svr.Get("/metrics", [&, start_time](const httplib::Request&, httplib::Response& res) {
    const auto jobs = runner.list();
    std::map<std::string, int> by_state;
    for (const auto& j : jobs) ++by_state[j.state];
    const auto gpus = runner.gpu_status();
    const int gpu_total = static_cast<int>(o.gpus.size());
    int gpu_busy = 0;
    for (const auto& p : gpus.value("pool", Json::array()))
      if (!p.value("job", Json(nullptr)).is_null()) ++gpu_busy;
    const double now =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string out;
    auto line = [&](std::string_view name, std::string_view help, std::string_view type) {
      out += fmt::format("# HELP {} {}\n# TYPE {} {}\n", name, help, name, type);
    };
    line("mqde_uptime_seconds", "seconds since this server process started", "gauge");
    out += fmt::format("mqde_uptime_seconds {:.3f}\n", now - start_time);
    line("mqde_jobs_total", "jobs known to this process, by state", "gauge");
    for (const char* s : {"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"})
      out += fmt::format("mqde_jobs_total{{state=\"{}\"}} {}\n", s, by_state.count(s) ? by_state.at(s) : 0);
    line("mqde_gpu_pool_size", "GPUs configured with --gpus", "gauge");
    out += fmt::format("mqde_gpu_pool_size {}\n", gpu_total);
    line("mqde_gpu_busy", "GPUs currently running a job", "gauge");
    out += fmt::format("mqde_gpu_busy {}\n", gpu_busy);
    line("mqde_queue_length", "jobs waiting for a free GPU or CPU slot", "gauge");
    out += fmt::format("mqde_queue_length {}\n", gpus.value("queued", 0));
    res.set_content(out, "text/plain; version=0.0.4; charset=utf-8");
  });
  svr.Get("/api/v1/hardware", [&](const httplib::Request&, httplib::Response& res) {
    mqde_device_info devs[32];
    int n = 0;
    Json arr = Json::array();
    if (mqde_probe_all(devs, 32, &n) == 0)
      for (int i = 0; i < n; ++i) {
        Json d = {{"kind", mqde_dev_kind_name(devs[i].kind)},
                  {"index", devs[i].index},
                  {"vendor", devs[i].vendor},
                  {"name", devs[i].name},
                  {"arch", devs[i].arch},
                  {"mem_total_bytes", devs[i].mem_total_bytes},
                  {"mem_free_bytes", devs[i].mem_free_bytes},
                  {"compute_units", devs[i].compute_units},
                  {"driver_version", devs[i].driver_version}};
        if (devs[i].kind == MQDE_DEV_GPU)
          if (auto hw = estimator::match_hardware(devs[i].name)) d["profile"] = hw->id;
        arr.push_back(d);
      }
    send_json(res, {{"devices", arr}, {"job_gpus", runner.gpu_status()}});
  });
  // HAL plugins: what was found, in which state, and why a plugin was rejected or disabled.
  svr.Get("/api/v1/hardware/plugins", [](const httplib::Request&, httplib::Response& res) {
    mqde_hal_ensure_loaded();
    static const char* const kStatus[] = {"loaded", "inactive", "rejected", "disabled"};
    Json arr = Json::array();
    for (int i = 0; i < mqde_hal_plugin_count(); ++i) {
      mqde_hal_plugin_info p;
      if (mqde_hal_plugin_get(i, &p) != 0) continue;
      const bool known = p.status >= 0 && p.status < 4;
      arr.push_back({{"name", p.name},
                     {"vendor", p.vendor},
                     {"version", p.version},
                     {"status", known ? kStatus[p.status] : "unknown"},
                     {"devices", p.devices},
                     {"reason", p.reason},
                     {"has_profile", mqde_hal_plugin_by_name(p.name) != nullptr &&
                                         mqde_hal_plugin_by_name(p.name)->profile != nullptr}});
    }
    send_json(res, arr);
  });
  // Micro-benchmark on one device of one plugin: {"plugin","device","kind":"bandwidth|gemm","dtype"?}.
  // Serialised, because two benchmarks at once would measure each other.
  svr.Post("/api/v1/hardware/profile", [](const httplib::Request& req, httplib::Response& res) {
    static std::mutex bench_mu;
    auto body = parse_body(req, res);
    if (!body) return;
    const std::string name = body->value("plugin", "");
    const std::string kind = body->value("kind", "");
    const std::string dtype = body->value("dtype", "fp32");
    if (kind != "bandwidth" && kind != "gemm")
      return send_error(res, 400, "bad_request", "kind must be 'bandwidth' or 'gemm'");
    if (dtype.size() >= 16) return send_error(res, 400, "bad_request", "dtype is too long");
    mqde_hal_ensure_loaded();
    const auto* api = mqde_hal_plugin_by_name(name.c_str());
    if (!api) return send_error(res, 404, "not_found", "no loaded plugin named '" + name + "'");
    if (!api->profile)
      return send_error(res, 400, "unsupported", "plugin '" + name + "' has no micro-benchmarks");
    mqde_hal_profile_request r{};
    r.device = body->value("device", 0);
    std::snprintf(r.kind, sizeof r.kind, "%s", kind.c_str());
    std::snprintf(r.dtype, sizeof r.dtype, "%s", dtype.c_str());
    mqde_hal_profile_result out{};
    int rc;
    {
      std::lock_guard lk(bench_mu);
      rc = api->profile(r.device, &r, &out);
    }
    if (rc != 0)
      return send_error(res, 422, "profile_failed", api->last_error ? api->last_error() : "benchmark failed");
    send_json(res, {{"plugin", name},
                    {"device", r.device},
                    {"kind", kind},
                    {"dtype", dtype},
                    {"value", out.value},
                    {"unit", out.unit},
                    {"seconds", out.seconds}});
  });
  // Serving-runtime profiles for the estimator (hf_transformers, vllm, ...), with the fields that change the
  // estimate.
  svr.Get("/api/v1/backends", [](const httplib::Request&, httplib::Response& res) {
    auto ids = estimator::list_data_entries("backends");
    if (!ids) return send_error(res, ids.error());
    Json arr = Json::array();
    for (const auto& id : *ids)
      if (auto b = estimator::load_backend(id))
        arr.push_back({{"id", id},
                       {"activation_model", b->activation_model},
                       {"max_batched_tokens", b->max_batched_tokens},
                       {"runtime_overhead_bytes", b->runtime_overhead_bytes},
                       {"allocator_factor", b->allocator_factor}});
    send_json(res, arr);
  });
  svr.Get("/api/v1/hardware/profiles", [](const httplib::Request&, httplib::Response& res) {
    auto ids = estimator::list_data_entries("hardware_db");
    if (!ids) return send_error(res, ids.error());
    Json arr = Json::array();
    for (const auto& id : *ids)
      if (auto h = estimator::load_hardware(id)) arr.push_back(estimator::to_json(*h));
    send_json(res, arr);
  });
  // Domain registry (T-1301): the top-level problem domain (llm/vla/vlm/world/ewm) a JobSpec targets.
  svr.Get("/api/v1/domains", [](const httplib::Request&, httplib::Response& res) {
    auto ds = estimator::list_domains();  // in wizard order
    if (!ds) return send_error(res, ds.error());
    Json arr = Json::array();
    for (const auto& d : *ds) arr.push_back(estimator::to_json(d));
    send_json(res, arr);
  });
  svr.Get("/api/v1/recipes", [](const httplib::Request&, httplib::Response& res) {
    auto ids = estimator::list_data_entries("recipes");
    if (!ids) return send_error(res, ids.error());
    Json arr = Json::array();
    for (const auto& id : *ids)
      if (auto r = estimator::load_recipe(id)) {
        Json j = estimator::to_json(*r);
        j["id"] = id;
        arr.push_back(j);
      }
    send_json(res, arr);
  });
  // Quantization algorithms (data/algorithms.yaml). The job endpoint validates quant.algorithm against it.
  auto load_algorithms = []() -> Result<Json> {
    MQDE_ASSIGN_OR_RETURN(Json j, load_structured_file(data_dir() / "algorithms.yaml"));
    return j.value("algorithms", Json::array());
  };
  svr.Get("/api/v1/algorithms", [&](const httplib::Request&, httplib::Response& res) {
    auto a = load_algorithms();
    if (!a) return send_error(res, a.error());
    send_json(res, *a);
  });
  svr.Get("/api/v1/config", [&](const httplib::Request&, httplib::Response& res) {
    send_json(res, {{"allowed_roots", o.allowed_roots},
                    {"gpus", o.gpus},
                    {"runs_dir", fs::absolute(o.runs_dir).string()}});
  });

  // ---------------- models & estimation ----------------
  svr.Post("/api/v1/models:inspect", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("path", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "path is outside allowed roots");
    auto s = get_spec(path, true);
    if (!s) return send_error(res, s.error());
    send_json(res, modelspec::to_json(*s, body->value("tensors", false)));
  });

  svr.Post("/api/v1/estimate", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto spec = get_spec(path);
    if (!spec) return send_error(res, spec.error());
    auto hw = estimator::load_hardware(body->value("hardware", "nvidia/rtx4090"));
    if (!hw) return send_error(res, hw.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    const auto rt = runtime_from(body->value("runtime", Json::object()));
    std::vector<std::string> recipes = body->value("recipes", std::vector<std::string>{"bf16", "w4a16"});
    const auto conc_xs = body->value("concurrency_curve", std::vector<int64_t>{1, 2, 4, 8, 16, 32, 64, 128});
    const auto img_xs = body->value("images_curve", std::vector<int64_t>{1, 2, 4, 8, 16, 30, 60});
    Json out = Json::array();
    for (const auto& name : recipes) {
      auto r = estimator::load_recipe(name);
      if (!r) return send_error(res, r.error());
      auto e = estimator::estimate(*spec, *r, *hw, *be, rt);
      Json j = estimator::to_json(e);
      j["recipe_id"] = name;
      j["recipe_detail"] = estimator::to_json(*r);
      j["curves"] = {
          {"concurrency",
           estimator::to_json(estimator::memory_vs_concurrency(*spec, *r, *hw, *be, rt, conc_xs))},
          {"images", estimator::to_json(estimator::memory_vs_images(*spec, *r, *hw, *be, rt, img_xs))}};
      out.push_back(j);
    }
    send_json(res, {{"hardware", estimator::to_json(*hw)}, {"estimates", out}});
  });

  // Deployability matrix: recipes x hardware profiles.
  svr.Post("/api/v1/estimate:matrix", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto spec = get_spec(path);
    if (!spec) return send_error(res, spec.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    const auto rt = runtime_from(body->value("runtime", Json::object()));
    auto hw_ids = estimator::list_data_entries("hardware_db");
    auto recipe_ids = estimator::list_data_entries("recipes");
    if (!hw_ids || !recipe_ids) return send_error(res, 500, "internal", "data directory unreadable");
    Json cells = Json::array();
    for (const auto& h : *hw_ids) {
      auto hw = estimator::load_hardware(h);
      if (!hw) continue;
      for (const auto& rid : *recipe_ids) {
        auto r = estimator::load_recipe(rid);
        if (!r) continue;
        auto e = estimator::estimate(*spec, *r, *hw, *be, rt);
        cells.push_back({{"hardware", hw->id},
                         {"hardware_name", hw->name},
                         {"recipe", r->name},
                         {"feasible", e.feasible},
                         {"peak_bytes", e.peak_bytes},
                         {"capacity_bytes", e.capacity_bytes},
                         {"max_concurrency", e.max_concurrency},
                         {"latency_floor_ms", e.latency_floor_ms},
                         {"n_notes", e.reasons.size()}});
      }
    }
    send_json(res, {{"cells", cells}});
  });

  // Top-K (recipe, algorithm, backend, hardware) recommendations under memory + deployability constraints
  // (T-0211). Body: {model, runtime, hardware?: [ids] (default: every hardware_db entry), backend?: [ids]
  // (default: vllm), top_k?: default 5}.
  svr.Post("/api/v1/recommend", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto spec = get_spec(path);
    if (!spec) return send_error(res, spec.error());
    const auto rt = runtime_from(body->value("runtime", Json::object()));
    const int top_k = body->value("top_k", 5);
    // "throughput" is planner::recommend's own order (most sequences first); "balanced" / "accuracy" also
    // weigh the accuracy risk, measured on this model by finished jobs where there are any (see advice.h).
    const auto objective = body->value("objective", std::string("throughput"));
    if (!valid_objective(objective))
      return send_error(res, 400, "invalid_argument", "objective must be accuracy, balanced or throughput");
    std::vector<estimator::HardwareProfile> hws;
    if (auto want = body->value("hardware", Json::array()); !want.empty()) {
      for (const auto& h : want)
        if (auto hw = estimator::load_hardware(h.get<std::string>())) hws.push_back(*hw);
    } else {
      auto ids = estimator::list_data_entries("hardware_db");
      if (!ids) return send_error(res, ids.error());
      for (const auto& h : *ids)
        if (auto hw = estimator::load_hardware(h)) hws.push_back(*hw);
    }
    std::vector<estimator::BackendProfile> bes;
    auto want_be = body->value("backend", Json::array({"vllm"}));
    for (const auto& b : want_be)
      if (auto be = estimator::load_backend(b.get<std::string>())) bes.push_back(*be);
    if (hws.empty() || bes.empty())
      return send_error(res, 400, "invalid_argument", "no usable hardware / backend profile");
    const auto r = planner::recommend(*spec, bes, hws, rt, 1 << 20);  // rank everything here, truncate below
    const auto measured = measured_for(o.runs_dir, path);
    std::map<std::string, estimator::Recipe> recipes;
    if (auto ids = estimator::list_data_entries("recipes"))
      for (const auto& id : *ids)
        if (auto rc = estimator::load_recipe(id)) recipes[rc->name] = *rc;
    std::vector<Json> all;
    for (const auto& rec : r.top) {
      Json j = {{"recipe", rec.recipe},
                {"algorithm", rec.algorithm},
                {"backend", rec.backend},
                {"hardware", rec.hardware},
                {"maturity", rec.maturity},
                {"feasible", rec.feasible},
                {"deploy_verdict", std::string(planner::verdict_name(rec.deploy_verdict))},
                {"peak_bytes", rec.peak_bytes},
                {"capacity_bytes", rec.capacity_bytes},
                {"max_concurrency", rec.max_concurrency},
                {"export_format", rec.export_format},
                {"reasons", rec.reasons}};
      if (auto m = measured.find(rec.recipe); m != measured.end()) {
        j["risk"] = risk_name(m->second.risk);
        j["risk_source"] = "measured";
        j["measured"] = {{"job_id", m->second.job_id}, {"kind", m->second.kind}, {"algorithm", m->second.algorithm},
                         {"metric", m->second.metric}, {"value", m->second.value}};
      } else {
        auto rc = recipes.find(rec.recipe);
        j["risk"] = risk_name(rc != recipes.end() ? prior_risk(rc->second) : Risk::kMedium);
        j["risk_source"] = "prior";
      }
      all.push_back(std::move(j));
    }
    rank(all, objective);
    if (static_cast<int>(all.size()) > top_k) all.resize(static_cast<size_t>(top_k));
    Json top = Json::array();
    for (auto& j : all) top.push_back(std::move(j));
    send_json(res, {{"top", top}, {"excluded", r.excluded}, {"considered", static_cast<int>(r.top.size()) + r.excluded},
                    {"objective", objective}, {"measured_jobs", static_cast<int>(measured.size())}});
  });

  // Which export format a chosen (recipe, backend) pair should default to (T-1304): query params
  // recipe, backend, hardware (all required). Separate from /recommend, which searches every
  // combination; this is the one-shot lookup the new-job wizard needs once the user has already picked a
  // backend, so it can pre-select a format instead of leaving two independent chip lists to line up by hand.
  svr.Get("/api/v1/export-suggestion", [&](const httplib::Request& req, httplib::Response& res) {
    const auto recipe_id = req.get_param_value("recipe");
    const auto backend_id = req.get_param_value("backend");
    const auto hw_id = req.get_param_value("hardware");
    if (recipe_id.empty() || backend_id.empty() || hw_id.empty())
      return send_error(res, 400, "invalid_argument", "recipe, backend and hardware are all required");
    auto recipe = estimator::load_recipe(recipe_id);
    if (!recipe) return send_error(res, recipe.error());
    auto hw = estimator::load_hardware(hw_id);
    if (!hw) return send_error(res, hw.error());
    const auto format = planner::preferred_export_format(recipe->weight, backend_id);
    if (format.empty())
      return send_json(res, {{"format", ""}, {"verdict", "native"}, {"reason", "bf16: no export needed"}});
    const auto c = planner::check_deploy(format, *recipe, backend_id, *hw);
    send_json(res, {{"format", format},
                    {"verdict", std::string(planner::verdict_name(c.verdict))},
                    {"reason", c.reason}});
  });

  // ---------------- model library + graph (netron-style) ----------------
  auto scan_library = [&]() {
    std::vector<fs::path> out;
    for (const auto& r : o.allowed_roots) {
      std::error_code ec;
      for (const auto& e : fs::directory_iterator(r, ec))
        if (e.is_directory() && fs::is_regular_file(e.path() / "config.json")) out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  svr.Get("/api/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    Json arr = Json::array();
    for (const auto& p : scan_library()) {
      auto s = get_spec(p.string());
      if (!s) {
        arr.push_back({{"path", p.string()}, {"name", p.filename().string()}, {"error", s.error().message}});
        continue;
      }
      Json j = modelspec::to_json(*s, false);
      j["name"] = p.filename().string();
      arr.push_back(j);
    }
    send_json(res, arr);
  });
  // GGUF files (T-R05): listed from the allowed roots (two levels deep: a root, or a model directory inside it), and
  // read header-only. The tensor payload is never touched.
  auto is_gguf = [](const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext == ".gguf";
  };
  svr.Get("/api/v1/gguf", [&](const httplib::Request&, httplib::Response& res) {
    std::vector<fs::path> found;
    for (const auto& r : o.allowed_roots) {
      std::error_code ec;
      for (const auto& e : fs::directory_iterator(r, ec)) {
        if (e.is_regular_file(ec) && is_gguf(e.path())) found.push_back(e.path());
        else if (e.is_directory(ec))
          for (const auto& f : fs::directory_iterator(e.path(), ec))
            if (f.is_regular_file(ec) && is_gguf(f.path())) found.push_back(f.path());
      }
    }
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end()), found.end());
    Json arr = Json::array();
    for (const auto& p : found) {
      if (arr.size() >= 500) break;
      std::error_code ec;
      arr.push_back({{"path", p.string()}, {"name", p.filename().string()}, {"dir", p.parent_path().filename().string()},
                     {"bytes", fs::file_size(p, ec)}});
    }
    send_json(res, arr);
  });
  svr.Post("/api/v1/models:gguf", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("path", "");
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "path is outside allowed roots");
    if (!is_gguf(path) || !fs::is_regular_file(path)) return send_error(res, 400, "invalid_argument", "path must be an existing .gguf file");
    auto g = modelspec::read_gguf(path);
    if (!g) return send_error(res, g.error());
    Json out = modelspec::gguf_summary(*g, body->value("tensors", true));
    out["path"] = path;
    send_json(res, out);
  });
  svr.Post("/api/v1/models:graph", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("path", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "path is outside allowed roots");
    auto s = get_spec(path, true);
    if (!s) return send_error(res, s.error());
    send_json(res, modelspec::to_graph_json(*s, body->value("focus", ""), body->value("depth", 2)));
  });

  // netron-style tensor inspector: stats + histogram (+ what a recipe would do to it).
  svr.Post("/api/v1/models:tensor", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("path", "");
    if (!path_allowed(o.allowed_roots, path))
      return send_error(res, 403, "forbidden", "path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    const auto name = body->value("tensor", "");
    auto it =
        std::find_if(s->tensors.begin(), s->tensors.end(), [&](const auto& t) { return t.name == name; });
    if (it == s->tensors.end()) return send_error(res, 404, "not_found", "no such tensor");
    auto vals = modelspec::read_tensor_f32(path, *it);
    Json out = {{"name", it->name}, {"dtype", it->dtype}, {"shape", it->shape}, {"numel", it->numel}};
    if (!vals) {
      out["unsupported"] = vals.error().message;
      return send_json(res, out);
    }
    out["sampled"] = vals->size();
    out["stats"] = quant::to_json(quant::tensor_stats(vals->data(), static_cast<int64_t>(vals->size())));
    // 64-bin histogram over [min, max] (log-count friendly for heavy tails).
    const int bins = 64;
    float mn = *std::min_element(vals->begin(), vals->end()),
          mx = *std::max_element(vals->begin(), vals->end());
    if (!(mx > mn)) mx = mn + 1e-6f;
    std::vector<int64_t> h(bins, 0);
    for (float v : *vals) h[std::clamp(static_cast<int>((v - mn) / (mx - mn) * bins), 0, bins - 1)]++;
    out["hist"] = {{"min", mn}, {"max", mx}, {"counts", h}};
    // Optional: preview the quantization error of a recipe on this tensor (2-D only).
    const auto recipe = body->value("recipe", "");
    if (!recipe.empty() && it->shape.size() == 2 && static_cast<int64_t>(vals->size()) == it->numel) {
      auto r = estimator::load_recipe(recipe);
      if (r && r->weight.enabled()) {
        std::vector<float> q(vals->size());
        if (quant::fake_quant(vals->data(), it->shape[0], it->shape[1], r->weight, q.data()))
          out["quant_preview"] = {{"recipe", r->name},
                                  {"metrics", quant::to_json(quant::measure(
                                                  vals->data(), q.data(), static_cast<int64_t>(q.size())))}};
      }
    }
    send_json(res, out);
  });

  // ---------------- deployment pre-evaluation (llmfit-style) ----------------
  auto fit_cfg = fit::load_fit_config();
  if (!fit_cfg) return std::unexpected(fit_cfg.error().with_context("loading data/fit/fit.yaml"));
  auto host_ram_gb = [] {
    mqde_device_info devs[32];
    int n = 0;
    if (mqde_probe_all(devs, 32, &n) == 0 && n > 0)
      return static_cast<double>(devs[0].mem_total_bytes) / (1024.0 * 1024.0 * 1024.0);
    return 0.0;
  };
  auto host_cores = [] {
    mqde_device_info devs[32];
    int n = 0;
    if (mqde_probe_all(devs, 32, &n) == 0 && n > 0) return static_cast<int>(devs[0].compute_units);
    return 0;
  };
  // System from a hardware profile id, overridable field by field (llmfit "plan" what-if).
  auto system_from = [&](const Json& b) -> Result<fit::System> {
    MQDE_ASSIGN_OR_RETURN(auto hw, estimator::load_hardware(b.value("hardware", "nvidia/rtx4090")));
    auto sys = fit::system_from_profile(hw, b.value("ram_gb", host_ram_gb()));
    sys.gpu_count = b.value("gpu_count", 1);
    sys.cpu_cores = b.value("cpu_cores", host_cores());
    if (b.contains("vram_gb")) sys.vram_gb = b["vram_gb"].get<double>();
    if (b.contains("mem_bw_gbps")) sys.mem_bw_gbps = b["mem_bw_gbps"].get<double>();
    return sys;
  };
  // This machine's identity, and its baseline index (reloaded per request so a fresh `mqde bench`
  // shows up without restarting the server).
  auto host_id = [] {
    fit::HardwareId id;
    mqde_device_info devs[32];
    int n = 0;
    if (mqde_probe_all(devs, 32, &n) == 0)
      for (int i = 0; i < n; ++i) {
        if (devs[i].kind == MQDE_DEV_CPU)
          id.cpu_name = devs[i].name;
        else if (devs[i].kind == MQDE_DEV_GPU && id.gpu_name.empty())
          id.gpu_name = devs[i].name;
      }
    return id;
  }();
  std::string host_profile;
  if (!host_id.gpu_name.empty())
    if (auto hp = estimator::match_hardware(host_id.gpu_name)) host_profile = hp->id;
  // Measurements apply only when the evaluated profile is this machine and the run is single-GPU.
  auto attach_bench = [&](fit::Request& r, const std::string& profile, const fit::System& sys,
                          std::optional<fit::BenchmarkIndex>& holder) {
    if (host_profile.empty() || profile != host_profile || sys.gpu_count != 1) return;
    holder = fit::BenchmarkIndex::load(host_id);
    r.bench = &*holder;
  };
  auto fit_request = [](const Json& b) {
    fit::Request r;
    r.use_case = b.value("use_case", "auto");
    r.context = b.value("context", int64_t{0});
    r.concurrency = b.value("concurrency", int64_t{1});
    r.force_quant = b.value("quant", "");
    return r;
  };
  // Runtime probing costs up to 2 s, so cache it for 30 s across /fit calls.
  std::mutex installed_mu;
  std::optional<fit::InstalledIndex> installed_cache;
  auto installed_at = std::chrono::steady_clock::time_point{};
  auto installed_index = [&]() -> fit::InstalledIndex {
    std::lock_guard lk(installed_mu);
    const auto now = std::chrono::steady_clock::now();
    if (!installed_cache || now - installed_at > std::chrono::seconds(30)) {
      installed_cache = fit::probe_installed();
      installed_at = now;
    }
    return *installed_cache;
  };

  svr.Post("/api/v1/fit", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    auto sys = system_from(*body);
    if (!sys) return send_error(res, sys.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    auto r = fit_request(*body);
    std::optional<fit::BenchmarkIndex> bench;
    attach_bench(r, body->value("hardware", "nvidia/rtx4090"), *sys, bench);
    std::vector<std::string> models = body->value("models", std::vector<std::string>{});
    // source: "local" (default: scan the model library) | "catalog" (HF catalog, filtered by search).
    const std::string source = body->value("source", "local");
    if (models.empty() && source == "catalog") {
      const auto c = catalog_snapshot();
      for (const auto* e : hub::search(c, body->value("search", ""), body->value("provider", ""),
                                       body->value("capability", ""), body->value("limit", size_t{50})))
        models.push_back("hf:" + e->name);
    } else if (models.empty())
      for (const auto& p : scan_library()) models.push_back(p.string());
    std::vector<fit::ModelFit> fits;
    Json errors = Json::array();
    for (const auto& m : models) {
      if (!model_allowed(m)) {
        errors.push_back({{"model", m}, {"error", "outside allowed roots"}});
        continue;
      }
      auto s = get_spec(m);
      if (!s) {
        errors.push_back({{"model", m}, {"error", s.error().message}});
        continue;
      }
      auto f = fit::analyze(*s, *sys, *fit_cfg, *be, r);
      if (!f) {
        errors.push_back({{"model", m}, {"error", f.error().message}});
        continue;
      }
      // T-R24: a model is installed when a local runtime/cache holds it. A model directory on this
      // machine also counts ("local_dir"): the HF-transformers runtime can load it as-is.
      f->installed_providers = installed_index().installed_providers(m.starts_with("hf:") ? m.substr(3) : m);
      if (std::filesystem::exists(std::filesystem::path(m) / "config.json"))
        f->installed_providers.insert(f->installed_providers.begin(), "local_dir");
      f->installed = !f->installed_providers.empty();
      fits.push_back(*f);
    }
    fit::rank(fits, body->value("sort", "score"));
    Json arr = Json::array();
    for (const auto& f : fits) arr.push_back(fit::to_json(f));
    send_json(
        res,
        {{"system", fit::to_json(*sys)}, {"models", arr}, {"errors", errors}, {"use_cases", [&] {
                                                                                 Json u = Json::array();
                                                                                 for (const auto& [k, _] :
                                                                                      fit_cfg->use_cases)
                                                                                   u.push_back(k);
                                                                                 return u;
                                                                               }()}});
  });
  // One model across every hardware profile (llmfit "plan": which device should I buy / deploy to?).
  svr.Post("/api/v1/fit:plan", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    auto ids = estimator::list_data_entries("hardware_db");
    if (!ids) return send_error(res, ids.error());
    const auto r = fit_request(*body);

    // Collect all per-hardware rows.
    Json rows = Json::array();
    std::optional<Json> min_hw, rec_hw;
    double best_score = -1;
    for (const auto& id : *ids) {
      Json b = *body;
      b["hardware"] = id;
      auto sys = system_from(b);
      if (!sys) continue;
      auto rr = r;
      std::optional<fit::BenchmarkIndex> bench;
      attach_bench(rr, id, *sys, bench);
      auto f = fit::analyze(*s, *sys, *fit_cfg, *be, rr);
      if (!f) continue;
      Json j = fit::to_json(*f);
      j.erase("detail");
      j["hardware"] = id;
      j["system"] = fit::to_json(*sys);
      rows.push_back(j);

      // Minimum hardware: first row that isn't TooTight.
      if (!min_hw && f->fit_level != fit::FitLevel::kTooTight) min_hw = j;
      // Recommended hardware: highest score.
      if (f->score > best_score) {
        best_score = f->score;
        rec_hw = j;
      }
    }

    // Run paths for the requested hardware: try all run modes for the chosen quant.
    Json run_paths = Json::array();
    {
      Json b = *body;
      auto sys = system_from(b);
      if (sys) {
        const std::vector<std::string> modes = {"gpu", "tensor_parallel", "moe_offload", "cpu_offload",
                                                "cpu_only"};
        for (const auto& mode : modes) {
          fit::Request rr = r;
          rr.force_run_mode = mode;
          auto f = fit::analyze(*s, *sys, *fit_cfg, *be, rr);
          if (!f || f->forced_mode_rejected) continue;
          Json j;
          j["run_mode"] = mode;
          j["utilization_pct"] = f->utilization_pct;
          j["best_quant"] = f->best_quant;
          j["memory_required_gb"] = f->memory_required_gb;
          j["estimated_tps"] = f->estimated_tps;
          j["speed_unit"] = f->speed_unit;
          j["fit_level"] = fit::to_string(f->fit_level);
          j["runtime"] = fit::to_string(f->runtime);
          run_paths.push_back(j);
        }
      }
    }

    // Upgrades: for each hardware row, compare to the current hardware row.
    Json upgrades = Json::array();
    {
      Json b = *body;
      auto cur_sys = system_from(b);
      if (cur_sys) {
        auto cur_f = fit::analyze(*s, *cur_sys, *fit_cfg, *be, r);
        if (cur_f) {
          for (const auto& row : rows) {
            const std::string hw_id = row["hardware"];
            if (hw_id == body->value("hardware", "nvidia/rtx4090")) continue;
            const double vram = row["system"].value("vram_gb", 0.0);
            if (vram <= cur_sys->vram_gb * cur_sys->gpu_count) continue;  // not bigger
            Json upg;
            upg["hardware"] = hw_id;
            upg["vram_delta_gb"] = vram - cur_sys->vram_gb * cur_sys->gpu_count;
            upg["tps_delta"] = static_cast<double>(row["estimated_tps"]) - cur_f->estimated_tps;
            upg["best_quant"] = row["best_quant"];
            upgrades.push_back(upg);
          }
        }
      }
    }

    // KV alternatives: vary kv quant (fp8 / q8) to show context/memory tradeoff.
    Json kv_alts = Json::array();
    {
      Json b = *body;
      auto sys = system_from(b);
      if (sys) {
        auto base_f = fit::analyze(*s, *sys, *fit_cfg, *be, r);
        for (const auto& kv_fmt : std::vector<std::string>{"fp8", "q8_0"}) {
          auto kv_recipe = estimator::load_recipe(kv_fmt);
          if (!kv_recipe) continue;
          estimator::RuntimeParams rt1;
          rt1.concurrency = r.concurrency;
          rt1.context_tokens = r.context > 0 ? r.context : 4096;
          auto e = estimator::estimate(*s, *kv_recipe,
                                       *(estimator::load_hardware(body->value("hardware", "nvidia/rtx4090"))),
                                       *be, rt1);
          Json alt;
          alt["kv_quant"] = kv_fmt;
          alt["kv_bytes_per_token"] = static_cast<int64_t>(e.kv_bytes_per_token);
          alt["memory_delta_gb"] = base_f ? static_cast<double>(e.kv_bytes) / (1ull << 30) -
                                                static_cast<double>(base_f->detail.kv_bytes) / (1ull << 30)
                                          : 0.0;
          kv_alts.push_back(alt);
        }
      }
    }

    send_json(res, {{"model", path},
                    {"rows", rows},
                    {"run_paths", run_paths},
                    {"minimum_hardware", min_hw ? *min_hw : Json(nullptr)},
                    {"recommended_hardware", rec_hw ? *rec_hw : Json(nullptr)},
                    {"upgrades", upgrades},
                    {"kv_alternatives", kv_alts}});
  });

  // ---------------- installed runtimes (T-R24) ----------------
  svr.Get("/api/v1/installed", [&](const httplib::Request&, httplib::Response& res) {
    auto idx = fit::probe_installed();
    {
      std::lock_guard lk(installed_mu);
      installed_cache = idx;
      installed_at = std::chrono::steady_clock::now();
    }
    send_json(res, fit::to_json(idx));
  });

  // ---------------- fit:claim ----------------
  svr.Post("/api/v1/fit:claim", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    auto hw = estimator::load_hardware(body->value("hardware", "nvidia/rtx4090"));
    if (!hw) return send_error(res, hw.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    auto sys = fit::system_from_profile(*hw);
    fit::Request cr;
    std::optional<fit::BenchmarkIndex> cbench;
    attach_bench(cr, body->value("hardware", "nvidia/rtx4090"), sys, cbench);
    auto b = fit::fit_bounds(*s, sys, *fit_cfg, *be, body->value("min_tps", 1.0), cr);
    send_json(res, fit::to_json(b));
  });

  // ---------------- fit:concurrency ----------------
  svr.Post("/api/v1/fit:concurrency", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!model_allowed(path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    auto hw = estimator::load_hardware(body->value("hardware", "nvidia/rtx4090"));
    if (!hw) return send_error(res, hw.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    const std::string recipe = body->value("recipe", "bf16");
    const int64_t users = body->value("users", 1);
    const int64_t context = body->value("context", 4096);
    auto c = fit::estimate_concurrency(*s, recipe, *hw, *be, users, context);
    Json out = fit::to_json(c);
    // Curves for the frontend: memory vs users at this context, and max context vs users.
    if (body->value("curve", true)) {
      const int64_t top = std::clamp<int64_t>(c.max_users > 0 ? c.max_users * 5 / 4 : 8, 4, 512);
      std::vector<int64_t> xs;
      for (int64_t u = 1; u <= top; u = u < 8 ? u + 1 : u * 3 / 2) xs.push_back(u);
      if (xs.back() != top) xs.push_back(top);
      Json by_users = Json::array();
      for (int64_t u : xs) {
        auto e = fit::estimate_concurrency(*s, recipe, *hw, *be, u, context);
        by_users.push_back({{"users", u},
                            {"total_gb", e.total_gb},
                            {"weights_gb", e.weights_gb},
                            {"kv_gb", e.kv_per_user_gb * static_cast<double>(u)},
                            {"overhead_gb", e.overhead_gb},
                            {"utilization_pct", e.utilization_pct},
                            {"fits", e.fits},
                            {"max_context", e.max_context_for_users}});
      }
      out["curve"] = by_users;
    }
    send_json(res, out);
  });

  // ---------------- catalog / HF search / download (T-R25) ----------------
  svr.Get("/api/v1/catalog", [&](const httplib::Request& req, httplib::Response& res) {
    const auto c = catalog_snapshot();
    const size_t limit = req.has_param("limit") ? std::stoul(req.get_param_value("limit")) : 100;
    const auto hits =
        hub::search(c, req.get_param_value("search"), req.get_param_value("provider"),
                    req.get_param_value("capability"), limit, req.get_param_value("include_demoted") == "1");
    Json arr = Json::array();
    for (const auto* e : hits) arr.push_back(hub::to_json(*e));
    size_t demoted = 0;
    for (const auto& e : c.entries) demoted += e.demoted;
    const auto rs = catalog_refresher.status();
    send_json(res, {{"updated_at", c.updated_at},
                   {"total", c.entries.size()},
                   {"demoted", demoted},
                   {"models", arr},
                   {"refresh_interval_s", rs.interval_s},
                   {"last_refresh_ok", rs.last_refresh_ok},
                   {"last_refresh_at", rs.last_refresh_at}});
  });

  // Live Hub search (llmfit hf-search). Needs network; the token (if any) comes from HF_TOKEN.
  svr.Get("/api/v1/hf/search", [&](const httplib::Request& req, httplib::Response& res) {
    const std::string q = req.get_param_value("q");
    if (q.empty()) return send_error(res, 400, "invalid_argument", "q is required");
    const std::string limit = req.has_param("limit") ? req.get_param_value("limit") : "20";
    auto l =
        hub::list_models({}, "search=" + hub::url_encode(q) + "&limit=" + std::to_string(std::stoi(limit)) +
                                 "&sort=downloads&direction=-1&expand[]=safetensors&expand[]=downloads"
                                 "&expand[]=likes&expand[]=pipeline_tag&expand[]=author");
    if (!l) return send_error(res, l.error());
    Json arr = Json::array();
    for (const auto& m : *l)
      arr.push_back(
          Json{{"name", m.value("id", "")},
               {"provider", m.value("author", "")},
               {"pipeline_tag", m.value("pipeline_tag", "")},
               {"hf_downloads", m.value("downloads", 0)},
               {"hf_likes", m.value("likes", 0)},
               {"parameters_raw",
                m.contains("safetensors") ? m["safetensors"].value("total", int64_t{0}) : int64_t{0}}});
    send_json(res, {{"models", arr}});
  });

  // Written by the download thread, read by status handlers: counters are atomic, state/error under `m`.
  struct Download {
    std::string id, repo, group, dir;  // immutable after creation
    uint64_t total = 0;
    size_t files_total = 0;
    std::atomic<uint64_t> done{0};
    std::atomic<size_t> files_done{0};
    std::atomic<bool> cancel{false};
    mutable std::mutex m;
    std::string state = "queued", error;
    void set(std::string st, std::string err = {}) {
      std::lock_guard lk(m);
      state = std::move(st);
      error = std::move(err);
    }
  };
  std::mutex dl_mu;
  std::map<std::string, std::shared_ptr<Download>> downloads;
  auto dl_json = [](const Download& d) {
    std::lock_guard lk(d.m);
    return Json{{"id", d.id},
                {"repo", d.repo},
                {"group", d.group},
                {"dir", d.dir},
                {"state", d.state},
                {"error", d.error},
                {"bytes_done", d.done.load()},
                {"bytes_total", d.total},
                {"files_done", d.files_done.load()},
                {"files_total", d.files_total}};
  };

  svr.Post("/api/v1/download", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    std::string repo = body->value("model", "");
    if (repo.starts_with("hf:")) repo = repo.substr(3);
    static const std::regex repo_re(R"(^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$)");
    if (!std::regex_match(repo, repo_re))
      return send_error(res, 400, "invalid_argument", "model must be <org>/<name>");
    if (o.allowed_roots.empty())
      return send_error(res, 403, "forbidden", "server has no --allow root to download into");
    const std::string out_root = body->value("output_dir", o.allowed_roots.front());
    if (!path_allowed(o.allowed_roots, out_root))
      return send_error(res, 403, "forbidden", "output_dir is outside allowed roots");
    auto files = hub::list_files({}, repo);
    if (!files) return send_error(res, files.error());
    const auto groups = hub::group_files(*files);
    uint64_t budget = 0;
    if (const auto b = body->value("budget", ""); !b.empty()) {
      auto p = hub::parse_size(b);
      if (!p) return send_error(res, 400, "invalid_argument", "bad budget '" + b + "'");
      budget = *p;
    }
    const auto g = hub::choose_group(groups, budget, body->value("quant", ""));
    if (!g)
      return send_error(res, 404, "not_found", "no downloadable weights (or requested quant) in " + repo);
    if (body->value("list", false)) {
      Json arr = Json::array();
      for (const auto& x : groups)
        arr.push_back(
            {{"group", x.key}, {"bytes", x.bytes}, {"files", x.files.size()}, {"quality", x.quality}});
      return send_json(res, {{"groups", arr}, {"chosen", g->key}});
    }
    auto d = std::make_shared<Download>();
    d->id = fmt::format("dl-{}", std::chrono::steady_clock::now().time_since_epoch().count());
    d->repo = repo;
    d->group = g->key;
    d->dir = (std::filesystem::path(out_root) / repo.substr(repo.find('/') + 1)).string();
    d->total = g->bytes;
    d->files_total = g->files.size();
    {
      std::lock_guard lk(dl_mu);
      downloads[d->id] = d;
    }
    std::thread([d, files = g->files] {
      d->set("running");
      constexpr int kDownloadChunks = 4;  // download_file_chunked falls back to a single stream on its own
                                          // for small files or a server that does not support Range
      for (const auto& f : files) {
        const uint64_t base = d->done.load();
        auto r = hub::download_file_chunked({}, d->repo, f.path, std::filesystem::path(d->dir) / f.path,
                                            kDownloadChunks, [&](uint64_t cur, uint64_t) {
                                              d->done = base + cur;
                                              return !d->cancel.load();
                                            });
        if (!r) return d->set(d->cancel ? "cancelled" : "failed", r.error().message);
        d->done = base + *r;
        ++d->files_done;
      }
      d->set("succeeded");
    }).detach();
    send_json(res, dl_json(*d), 202);
  });

  svr.Get(R"(/api/v1/download/([A-Za-z0-9-]+)/status)",
          [&](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard lk(dl_mu);
            const auto it = downloads.find(req.matches[1]);
            if (it == downloads.end()) return send_error(res, 404, "not_found", "no such download");
            send_json(res, dl_json(*it->second));
          });
  svr.Post(R"(/api/v1/download/([A-Za-z0-9-]+):cancel)",
           [&](const httplib::Request& req, httplib::Response& res) {
             std::lock_guard lk(dl_mu);
             const auto it = downloads.find(req.matches[1]);
             if (it == downloads.end()) return send_error(res, 404, "not_found", "no such download");
             it->second->cancel = true;
             send_json(res, dl_json(*it->second));
           });

  // ---------------- inference backend environments ----------------
  // GET  /api/v1/backend-envs[?check=1]   installed version / location per backend; check=1 also asks for the newest
  // POST /api/v1/backend-envs/<id>/install  {"version": "x"} (optional) -> {"install_id"}; one install at a time
  // GET  /api/v1/backend-envs/install/<install_id>  state, exit code and the last 200 log lines
  auto envs_error = [](httplib::Response& res, const Error& e) {
    const int status = e.code == Errc::kInvalidArgument ? 400 : e.code == Errc::kResourceExhausted ? 409
                       : e.code == Errc::kNotFound ? 404 : 502;
    send_error(res, status, errc_name(e.code), e.message);
  };
  svr.Get("/api/v1/backend-envs", [&](const httplib::Request& req, httplib::Response& res) {
    const bool check = req.get_param_value("check") == "1";
    auto st = backend_envs.status(check);
    if (!st) return envs_error(res, st.error());
    const auto cur = backend_envs.current_install();
    send_json(res, {{"backends", *st}, {"checked", check}, {"installing", cur ? *cur : Json(nullptr)}});
  });
  svr.Post(R"(/api/v1/backend-envs/([a-z]+)/install)", [&](const httplib::Request& req, httplib::Response& res) {
    std::string version;
    if (!req.body.empty()) {
      auto body = parse_body(req, res);
      if (!body) return;
      if (body->contains("version")) {
        if (!(*body)["version"].is_string()) return send_error(res, 400, "invalid_argument", "version must be a string");
        version = (*body)["version"].get<std::string>();
      }
    }
    auto id = backend_envs.start_install(req.matches[1], version);
    if (!id) return envs_error(res, id.error());
    send_json(res, {{"install_id", *id}}, 202);
  });
  svr.Get(R"(/api/v1/backend-envs/install/([0-9a-f]+))", [&](const httplib::Request& req, httplib::Response& res) {
    auto st = backend_envs.install_status(req.matches[1]);
    if (!st) return send_error(res, 404, "not_found", "no such install");
    send_json(res, *st);
  });

  // ---------------- jobs ----------------
  svr.Post("/api/v1/jobs", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    Json job = *body;
    const auto model = job.value("/model/path"_json_pointer, "");
    if (!path_allowed(o.allowed_roots, model))
      return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    for (const char* ptr : {"/calibration/dataset", "/evaluation/dataset"}) {
      const auto ds = job.value(Json::json_pointer(ptr), std::string("synthetic"));
      if (!path_allowed(o.allowed_roots, ds))
        return send_error(res, 403, "forbidden", std::string(ptr) + " is outside allowed roots");
    }
    // Every static JobSpec rule this module knows (recipe/algorithm existence, calibration, export
    // compatibility, model.type / device) is reported at once, not just the first one
    // (planner::validate_jobspec, T-0210).
    if (auto problems = planner::validate_jobspec(job); !problems.empty()) {
      std::vector<std::string> lines;
      for (const auto& p : problems) lines.push_back(fmt::format("{}: {}", p.field, p.message));
      Json detail = Json::array();
      for (const auto& p : problems)
        detail.push_back({{"field", p.field}, {"message", p.message}, {"suggestion", p.suggestion}});
      res.status = 400;
      res.set_content(Json{{"code", "invalid_argument"},
                           {"message", fmt::format("{}", fmt::join(lines, "; "))},
                           {"problems", detail}}
                          .dump(),
                      "application/json");
      return;
    }
    // validate_jobspec checked the artifact's id is a plain job id; only the server knows whether that job exists
    if (job.contains("artifact") && job["artifact"].is_object()) {
      const auto pj = job["artifact"].value("job", std::string());
      if (!fs::is_regular_file(o.runs_dir / pj / "report.json"))
        return send_error(res, 400, "invalid_argument", fmt::format("artifact.job: job {} not found", pj));
    }
    // ?dry_run=1: everything above (schema, planner, the artifact's job) without queueing anything
    if (req.has_param("dry_run")) return send_json(res, {{"valid", true}});
    const auto mtype = job.value("/model/type"_json_pointer, std::string("vla"));
    const bool cpu = job.value("/deploy/device"_json_pointer, std::string()) == "cpu";
    if (!cpu && o.gpus.empty())
      return send_error(res, 503, "unavailable", "server started without a GPU pool (--gpus)");
    job["apiVersion"] = "mqde/v1";
    job["kind"] = "Job";
    const auto id = runner.submit(job);
    send_json(res, {{"job_id", id}}, 201);
  });

  svr.Get("/api/v1/jobs", [&](const httplib::Request&, httplib::Response& res) {
    Json arr = Json::array();
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(o.runs_dir, ec)) {
      if (!e.is_directory()) continue;
      const auto id = e.path().filename().string();
      Json j = {{"job_id", id}};
      j.update(summarize_events(e.path() / "events.jsonl"));
      if (auto p = runner.get(id)) {
        j["gpu"] = p->gpu;
        j["attempt"] = p->attempt;
        j["max_attempts"] = p->max_attempts;
      }
      j["state"] = merged_state(j["state"].get<std::string>(), runner.get(id));
      const auto rep = read_json_file(e.path() / "report.json");
      auto spec_yaml = load_structured_file(e.path() / "job.yaml");
      const Json jy = spec_yaml ? *spec_yaml : Json::object();
      // report.kind is written by llm_job; job.yaml carries model.type for server-submitted jobs; else VLA
      const auto rep_kind = rep.is_object() ? rep.value("kind", "") : std::string();
      j["kind"] = rep_kind == "llm" || rep_kind == "vlm" || rep_kind == "llm_deploy" ? rep_kind : jy.value("/model/type"_json_pointer, std::string("vla"));
      j["device"] =
          jy.value("/deploy/device"_json_pointer,
                   rep.is_object() ? rep.value("/spec/device"_json_pointer, std::string()) : std::string());
      if (rep.is_object()) {
        // the measured device (report env), so the list can say "RTX 4090" after a restart lost the pool slot
        for (const char* k : {"/env/device_name", "/env/eval_device_name", "/env/quant_device_name"})
          if (const auto v = rep.value(Json::json_pointer(k), std::string()); !v.empty() && v != "cpu") {
            j["device_name"] = v;
            break;
          }
        j["name"] = rep.value("/spec/name"_json_pointer, id);
        j["recipes"] = rep.value("/spec/recipes"_json_pointer, Json::array());
        j["model"] = rep.value("/spec/model_path"_json_pointer, "");
      } else if (spec_yaml) {
        j["name"] = jy.value("/metadata/name"_json_pointer, id);
        j["recipes"] = jy.value("/quant/recipe"_json_pointer, Json::array());
        j["model"] = jy.value("/model/path"_json_pointer, "");
      }
      if (const auto pj = jy.value("/metadata/parent_job"_json_pointer, std::string()); !pj.empty()) j["parent_job"] = pj;
      arr.push_back(j);
    }
    std::sort(arr.begin(), arr.end(), [](const Json& a, const Json& b) { return a["job_id"] > b["job_id"]; });
    send_json(res, arr);
  });

  svr.Get(R"(/api/v1/jobs/([A-Za-z0-9_\-]+))", [&](const httplib::Request& req, httplib::Response& res) {
    const auto id = req.matches[1].str();
    const auto dir = o.runs_dir / id;
    if (!fs::is_directory(dir)) return send_error(res, 404, "not_found", "no such job");
    Json j = {{"job_id", id}, {"report", read_json_file(dir / "report.json")}};
    j.update(summarize_events(dir / "events.jsonl"));
    if (auto p = runner.get(id)) {
      j["gpu"] = p->gpu;
      j["exit_code"] = p->exit_code;
      j["attempt"] = p->attempt;
      j["max_attempts"] = p->max_attempts;
    }
    j["state"] = merged_state(j["state"].get<std::string>(), runner.get(id));
    if (auto spec = load_structured_file(dir / "job.yaml")) j["job"] = *spec;
    send_json(res, j);
  });

  svr.Post(R"(/api/v1/jobs/([A-Za-z0-9_\-]+):cancel)",
           [&](const httplib::Request& req, httplib::Response& res) {
             const auto id = req.matches[1].str();
             if (!runner.cancel(id)) return send_error(res, 409, "conflict", "job is not queued or running");
             send_json(res, {{"job_id", id}, {"state", "CANCELLED"}});
           });

  svr.Get(R"(/api/v1/jobs/([A-Za-z0-9_\-]+)/log)", [&](const httplib::Request& req, httplib::Response& res) {
    const auto id = req.matches[1].str();
    std::ifstream in(o.runs_dir / id / "worker.log");
    if (!in) return send_error(res, 404, "not_found", "no worker log");
    std::stringstream ss;
    ss << in.rdbuf();
    auto s = ss.str();
    if (s.size() > 256 * 1024) s = s.substr(s.size() - 256 * 1024);
    res.set_content(s, "text/plain");
  });

  // Self-contained HTML report written by the job (report.en.html / report.zh.html). Only these two fixed
  // file names are served, so no part of the request becomes a path.
  svr.Get(R"(/api/v1/jobs/([A-Za-z0-9_\-]+)/report\.(en|zh)\.html)",
          [&](const httplib::Request& req, httplib::Response& res) {
            const auto id = req.matches[1].str();
            const auto lang = req.matches[2].str();
            std::ifstream in(o.runs_dir / id / ("report." + lang + ".html"), std::ios::binary);
            if (!in) return send_error(res, 404, "not_found", "this job has no HTML report");
            std::stringstream ss;
            ss << in.rdbuf();
            res.set_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'");
            res.set_header("X-Content-Type-Options", "nosniff");
            res.set_content(ss.str(), "text/html; charset=utf-8");
          });

  // Events: replay from `after_seq`, then (stream) follow the file until the job ends.
  auto events_handler = [&](const httplib::Request& req, httplib::Response& res, bool stream) {
    const auto id = req.matches[1].str();
    if (!valid_job_id(id) || !fs::is_directory(o.runs_dir / id))
      return send_error(res, 404, "not_found", "no such job");
    const int64_t after = req.has_param("after_seq") ? std::stoll(req.get_param_value("after_seq")) : 0;
    const auto path = o.runs_dir / id / "events.jsonl";
    if (!stream) {
      std::ifstream in(path);
      Json arr = Json::array();
      std::string line;
      while (std::getline(in, line)) {
        try {
          auto ev = Json::parse(line);
          if (ev.value("seq", 0) > after) arr.push_back(ev);
        } catch (...) {
        }
      }
      return send_json(res, arr);
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    auto offset = std::make_shared<std::streamoff>(0);
    auto last_seq = std::make_shared<int64_t>(after);
    auto idle = std::make_shared<int>(0);
    res.set_chunked_content_provider(
        "text/event-stream", [&, path, id, offset, last_seq, idle](size_t, httplib::DataSink& sink) {
          std::ifstream in(path);
          bool terminal = false;
          if (in) {
            in.seekg(*offset);
            std::string line;
            while (std::getline(in, line)) {
              if (in.eof()) break;  // partial line; re-read next round
              *offset = in.tellg();
              try {
                auto ev = Json::parse(line);
                const int64_t seq = ev.value("seq", 0);
                if (seq <= *last_seq) continue;
                *last_seq = seq;
                const auto msg = fmt::format("id: {}\ndata: {}\n\n", seq, line);
                if (!sink.write(msg.data(), msg.size())) return false;
                if (ev.value("type", "") == "job_state" && ev.value("state", "") != "RUNNING")
                  terminal = true;
              } catch (...) {
              }
            }
          }
          auto p = runner.get(id);
          const bool running = p && (p->state == "QUEUED" || p->state == "RUNNING");
          if (terminal || (!running && ++*idle > 4)) {
            sink.done();
            return true;
          }
          static const std::string ping = ": ping\n\n";
          if (!sink.write(ping.data(), ping.size())) return false;
          std::this_thread::sleep_for(std::chrono::milliseconds(400));
          return true;
        });
  };
  svr.Get(R"(/api/v1/jobs/([A-Za-z0-9_\-]+)/events)",
          [&](const httplib::Request& req, httplib::Response& res) { events_handler(req, res, false); });
  svr.Get(R"(/api/v1/jobs/([A-Za-z0-9_\-]+)/stream)",
          [&](const httplib::Request& req, httplib::Response& res) { events_handler(req, res, true); });

  // ---------------- static frontend ----------------
  if (!o.web_dir.empty() && fs::is_directory(o.web_dir)) {
    svr.set_mount_point("/", o.web_dir.string());
    svr.set_error_handler([&](const httplib::Request& req, httplib::Response& res) {
      // SPA fallback: unknown non-API GETs serve index.html
      if (req.method == "GET" && !req.path.starts_with("/api/") && res.status == 404) {
        std::ifstream in(o.web_dir / "index.html");
        std::stringstream ss;
        ss << in.rdbuf();
        res.status = 200;
        res.set_content(ss.str(), "text/html");
      }
    });
  }

  svr.new_task_queue = [] { return new httplib::ThreadPool(16); };
  MQDE_LOG_WARN("mqde-server listening on http://{}:{} (runs: {}, gpus: [{}], web: {})", o.host, o.port,
                fs::absolute(o.runs_dir).string(), fmt::join(o.gpus, ","),
                o.web_dir.empty() ? "-" : o.web_dir.string());
  if (!svr.listen(o.host, o.port)) return make_error(Errc::kIo, "cannot listen on {}:{}", o.host, o.port);
  return {};
}

}  // namespace mqde::server
