#include "mqde/server/server.h"

#include <chrono>
#include <fstream>
#include <thread>

#include <fmt/ranges.h>
#include <httplib.h>

#include "job_runner.h"
#include "mqde/common/log.h"
#include "mqde/estimator/estimator.h"
#include "mqde/fit/concurrency.h"
#include "mqde/fit/fit.h"
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
  const int status = e.code == Errc::kNotFound ? 404 : e.code == Errc::kInvalidArgument || e.code == Errc::kParse ? 400 : 500;
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
  if (p == "synthetic") return true;
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

// Terminal states written by the worker win; otherwise the process view (queued / running /
// crashed before writing a terminal event / cancelled) is authoritative.
std::string merged_state(const std::string& from_events, const std::optional<JobProc>& p) {
  const bool terminal = from_events == "SUCCEEDED" || from_events == "FAILED";
  if (!p) return from_events;
  if (p->state == "CANCELLED") return "CANCELLED";
  if (terminal) return from_events;
  if (p->state == "SUCCEEDED" || p->state == "FAILED") return p->exit_code == 0 ? "SUCCEEDED" : "FAILED";
  return p->state;
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
  std::mutex spec_mu;
  std::map<std::string, modelspec::ModelSpec> spec_cache;  // model path -> spec

  auto get_spec = [&](const std::string& path) -> Result<modelspec::ModelSpec> {
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
  svr.Get("/api/v1/hardware", [&](const httplib::Request&, httplib::Response& res) {
    mqde_device_info devs[32];
    int n = 0;
    Json arr = Json::array();
    if (mqde_probe_all(devs, 32, &n) == 0)
      for (int i = 0; i < n; ++i) {
        Json d = {{"kind", devs[i].kind == MQDE_DEV_CPU ? "cpu" : "gpu"}, {"index", devs[i].index},
                  {"name", devs[i].name},  {"arch", devs[i].arch},
                  {"mem_total_bytes", devs[i].mem_total_bytes}, {"mem_free_bytes", devs[i].mem_free_bytes},
                  {"compute_units", devs[i].compute_units}, {"driver_version", devs[i].driver_version}};
        if (devs[i].kind == MQDE_DEV_GPU)
          if (auto hw = estimator::match_hardware(devs[i].name)) d["profile"] = hw->id;
        arr.push_back(d);
      }
    send_json(res, {{"devices", arr}, {"job_gpus", runner.gpu_status()}});
  });
  svr.Get("/api/v1/hardware/profiles", [](const httplib::Request&, httplib::Response& res) {
    auto ids = estimator::list_data_entries("hardware_db");
    if (!ids) return send_error(res, ids.error());
    Json arr = Json::array();
    for (const auto& id : *ids)
      if (auto h = estimator::load_hardware(id)) arr.push_back(estimator::to_json(*h));
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
  svr.Get("/api/v1/config", [&](const httplib::Request&, httplib::Response& res) {
    send_json(res, {{"allowed_roots", o.allowed_roots}, {"gpus", o.gpus}, {"runs_dir", fs::absolute(o.runs_dir).string()}});
  });

  // ---------------- models & estimation ----------------
  svr.Post("/api/v1/models:inspect", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("path", "");
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    send_json(res, modelspec::to_json(*s, body->value("tensors", false)));
  });

  svr.Post("/api/v1/estimate", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
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
      j["curves"] = {{"concurrency", estimator::to_json(estimator::memory_vs_concurrency(*spec, *r, *hw, *be, rt, conc_xs))},
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
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
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
        cells.push_back({{"hardware", hw->id}, {"hardware_name", hw->name}, {"recipe", r->name}, {"feasible", e.feasible},
                         {"peak_bytes", e.peak_bytes}, {"capacity_bytes", e.capacity_bytes},
                         {"max_concurrency", e.max_concurrency}, {"latency_floor_ms", e.latency_floor_ms},
                         {"n_notes", e.reasons.size()}});
      }
    }
    send_json(res, {{"cells", cells}});
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
  svr.Post("/api/v1/models:graph", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("path", "");
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    send_json(res, modelspec::to_graph_json(*s, body->value("focus", ""), body->value("depth", 2)));
  });

  // netron-style tensor inspector: stats + histogram (+ what a recipe would do to it).
  svr.Post("/api/v1/models:tensor", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("path", "");
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    const auto name = body->value("tensor", "");
    auto it = std::find_if(s->tensors.begin(), s->tensors.end(), [&](const auto& t) { return t.name == name; });
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
    float mn = *std::min_element(vals->begin(), vals->end()), mx = *std::max_element(vals->begin(), vals->end());
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
                                  {"metrics", quant::to_json(quant::measure(vals->data(), q.data(), static_cast<int64_t>(q.size())))}};
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
    if (mqde_probe_all(devs, 32, &n) == 0 && n > 0) return static_cast<double>(devs[0].mem_total_bytes) / (1024.0 * 1024.0 * 1024.0);
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
        if (devs[i].kind == MQDE_DEV_CPU) id.cpu_name = devs[i].name;
        else if (devs[i].kind == MQDE_DEV_GPU && id.gpu_name.empty()) id.gpu_name = devs[i].name;
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
    if (models.empty())
      for (const auto& p : scan_library()) models.push_back(p.string());
    std::vector<fit::ModelFit> fits;
    Json errors = Json::array();
    for (const auto& m : models) {
      if (!path_allowed(o.allowed_roots, m)) {
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
      fits.push_back(*f);
    }
    fit::rank(fits, body->value("sort", "score"));
    Json arr = Json::array();
    for (const auto& f : fits) arr.push_back(fit::to_json(f));
    send_json(res, {{"system", fit::to_json(*sys)}, {"models", arr}, {"errors", errors},
                    {"use_cases", [&] { Json u = Json::array(); for (const auto& [k, _] : fit_cfg->use_cases) u.push_back(k); return u; }()}});
  });
  // One model across every hardware profile (llmfit "plan": which device should I buy / deploy to?).
  svr.Post("/api/v1/fit:plan", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    auto ids = estimator::list_data_entries("hardware_db");
    if (!ids) return send_error(res, ids.error());
    const auto r = fit_request(*body);
    Json rows = Json::array();
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
    }
    send_json(res, {{"model", path}, {"rows", rows}});
  });

  // ---------------- fit:concurrency ----------------
  svr.Post("/api/v1/fit:concurrency", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    const auto path = body->value("model", "");
    if (!path_allowed(o.allowed_roots, path)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    auto s = get_spec(path);
    if (!s) return send_error(res, s.error());
    auto hw = estimator::load_hardware(body->value("hardware", "nvidia/rtx4090"));
    if (!hw) return send_error(res, hw.error());
    auto be = estimator::load_backend(body->value("backend", "hf_transformers"));
    if (!be) return send_error(res, be.error());
    const std::string recipe = body->value("recipe", "bf16");
    const int64_t users   = body->value("users", 1);
    const int64_t context = body->value("context", 4096);
    auto c = fit::estimate_concurrency(*s, recipe, *hw, *be, users, context);
    send_json(res, fit::to_json(c));
  });

  // ---------------- jobs ----------------
  svr.Post("/api/v1/jobs", [&](const httplib::Request& req, httplib::Response& res) {
    auto body = parse_body(req, res);
    if (!body) return;
    Json job = *body;
    const auto model = job.value("/model/path"_json_pointer, "");
    if (!path_allowed(o.allowed_roots, model)) return send_error(res, 403, "forbidden", "model path is outside allowed roots");
    for (const char* ptr : {"/calibration/dataset", "/evaluation/dataset"}) {
      const auto ds = job.value(Json::json_pointer(ptr), std::string("synthetic"));
      if (!path_allowed(o.allowed_roots, ds)) return send_error(res, 403, "forbidden", std::string(ptr) + " is outside allowed roots");
    }
    const auto recipes = job.value("/quant/recipe"_json_pointer, Json::array());
    if (!recipes.is_array() || recipes.empty()) return send_error(res, 400, "invalid_argument", "quant.recipe must be a non-empty list");
    for (const auto& r : recipes)
      if (auto rr = estimator::load_recipe(r.get<std::string>()); !rr) return send_error(res, rr.error());
    if (o.gpus.empty()) return send_error(res, 503, "unavailable", "server started without a GPU pool (--gpus)");
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
      if (auto p = runner.get(id)) j["gpu"] = p->gpu;
      j["state"] = merged_state(j["state"].get<std::string>(), runner.get(id));
      const auto rep = read_json_file(e.path() / "report.json");
      if (rep.is_object()) {
        j["name"] = rep.value("/spec/name"_json_pointer, id);
        j["recipes"] = rep.value("/spec/recipes"_json_pointer, Json::array());
        j["model"] = rep.value("/spec/model_path"_json_pointer, "");
      } else if (auto spec = load_structured_file(e.path() / "job.yaml")) {
        j["name"] = spec->value("/metadata/name"_json_pointer, id);
        j["recipes"] = spec->value("/quant/recipe"_json_pointer, Json::array());
        j["model"] = spec->value("/model/path"_json_pointer, "");
      }
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
    }
    j["state"] = merged_state(j["state"].get<std::string>(), runner.get(id));
    if (auto spec = load_structured_file(dir / "job.yaml")) j["job"] = *spec;
    send_json(res, j);
  });

  svr.Post(R"(/api/v1/jobs/([A-Za-z0-9_\-]+):cancel)", [&](const httplib::Request& req, httplib::Response& res) {
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

  // Events: replay from `after_seq`, then (stream) follow the file until the job ends.
  auto events_handler = [&](const httplib::Request& req, httplib::Response& res, bool stream) {
    const auto id = req.matches[1].str();
    if (!valid_job_id(id) || !fs::is_directory(o.runs_dir / id)) return send_error(res, 404, "not_found", "no such job");
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
    res.set_chunked_content_provider("text/event-stream", [&, path, id, offset, last_seq, idle](size_t, httplib::DataSink& sink) {
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
            if (ev.value("type", "") == "job_state" && ev.value("state", "") != "RUNNING") terminal = true;
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
                fs::absolute(o.runs_dir).string(), fmt::join(o.gpus, ","), o.web_dir.empty() ? "-" : o.web_dir.string());
  if (!svr.listen(o.host, o.port)) return make_error(Errc::kIo, "cannot listen on {}:{}", o.host, o.port);
  return {};
}

}  // namespace mqde::server
