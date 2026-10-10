// mqde - command line entry point (docs/ARCHITECTURE.md §6.16).
// stdout carries command output (tables or --json); diagnostics go to stderr.

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>

#include <CLI/CLI.hpp>

#include "mqde/common/log.h"
#include "mqde/estimator/estimator.h"
#include "mqde/fit/benchmarks.h"
#include "mqde/fit/claim.h"
#include "mqde/fit/fit.h"
#include "mqde/hub/catalog.h"
#include "mqde/hub/hf.h"
#include "mqde/server/server.h"
#include "mqde_probe.h"

namespace {

using mqde::format_bytes;
using mqde::Json;

std::vector<int64_t> parse_list(const std::string& s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(std::stoll(item));
  return out;
}

int fail(const mqde::Error& e) {
  std::cerr << "error: " << e.to_string() << "\n";
  return 2;
}

Json probe_json() {
  mqde_device_info devs[32];
  int n = 0;
  Json arr = Json::array();
  if (mqde_probe_all(devs, 32, &n) != 0) return arr;
  for (int i = 0; i < n; ++i) {
    const auto& d = devs[i];
    Json j = {{"kind", mqde_dev_kind_name(d.kind)},
              {"index", d.index},
              {"vendor", d.vendor},
              {"name", d.name},
              {"arch", d.arch},
              {"mem_total_bytes", d.mem_total_bytes},
              {"mem_free_bytes", d.mem_free_bytes},
              {"compute_units", d.compute_units},
              {"driver_version", d.driver_version}};
    if (d.kind == MQDE_DEV_GPU)
      if (auto hw = mqde::estimator::match_hardware(d.name)) j["profile"] = hw->id;
    arr.push_back(j);
  }
  return arr;
}

// Identity of this machine for benchmark matching: the given GPU (or the first one) + the CPU.
mqde::fit::HardwareId host_identity(int gpu_index, double* vram_gb = nullptr, double* ram_gb = nullptr,
                                    int* cores = nullptr) {
  mqde::fit::HardwareId id;
  for (const auto& d : probe_json()) {
    if (d["kind"] == "cpu") {
      id.cpu_name = d["name"].get<std::string>();
      if (ram_gb) *ram_gb = d["mem_total_bytes"].get<double>() / (1024.0 * 1024.0 * 1024.0);
      if (cores) *cores = d["compute_units"].get<int>();
    } else if (d["kind"] == "gpu" && id.gpu_name.empty() &&
               (gpu_index < 0 || d["index"].get<int>() == gpu_index)) {
      id.gpu_name = d["name"].get<std::string>();
      if (vram_gb) *vram_gb = d["mem_total_bytes"].get<double>() / (1024.0 * 1024.0 * 1024.0);
    }
  }
  return id;
}

int cmd_probe(bool json) {
  const Json devs = probe_json();
  if (json) {
    std::cout << devs.dump(2) << "\n";
    return 0;
  }
  fmt::print("{:<4} {:<5} {:<40} {:<8} {:>12} {:>12}  {}\n", "idx", "kind", "name", "arch", "total", "free",
             "profile");
  for (const auto& d : devs)
    fmt::print("{:<4} {:<5} {:<40} {:<8} {:>12} {:>12}  {}\n", d["index"].get<int>(),
               d["kind"].get<std::string>(), d["name"].get<std::string>(), d["arch"].get<std::string>(),
               format_bytes(d["mem_total_bytes"].get<double>()),
               format_bytes(d["mem_free_bytes"].get<double>()), d.value("profile", "-"));
  return 0;
}

// "hf:org/name[@rev]" (or hf://...) reads config.json + safetensors headers from the Hub over HTTP Range;
// anything else is a local directory.
uint64_t g_remote_bytes = 0;
mqde::Result<mqde::modelspec::ModelSpec> inspect_any(const std::string& path) {
  for (const std::string p : {"hf://", "hf:"})
    if (path.starts_with(p))
      return mqde::hub::inspect_remote(mqde::hub::HfOptions{}, path.substr(p.size()), &g_remote_bytes);
  return mqde::modelspec::inspect_local(path);
}

int cmd_inspect(const std::string& path, bool json, bool tensors) {
  auto spec = inspect_any(path);
  if (!spec) return fail(spec.error());
  if (json) {
    std::cout << mqde::modelspec::to_json(*spec, tensors).dump(2) << "\n";
    return 0;
  }
  const auto& t = spec->text;
  fmt::print("model      {}\nfamily     {} ({})\ntype       {}\nparams     {:.3f} B   ({} on disk)\n",
             spec->path, spec->family, spec->modality, spec->model_type, spec->total_params / 1e9,
             format_bytes(double(spec->total_bytes)));
  fmt::print(
      "language   {} layers ({} full-attn, {} linear-attn), hidden {}, heads {}/{} kv, head_dim {}, vocab "
      "{}\n",
      t.n_layers, t.n_full_attention, t.n_linear_attention, t.hidden, t.n_heads, t.n_kv_heads, t.head_dim,
      t.vocab);
  if (spec->vision)
    fmt::print("vision     {} layers, hidden {}, patch {}, {} tokens/image\n", spec->vision->n_layers,
               spec->vision->hidden, spec->vision->patch_size, spec->vision->tokens_per_image);
  if (spec->action)
    fmt::print("action     horizon {} x dim {}, {} denoising steps, {} query tokens\n", spec->action->horizon,
               spec->action->action_dim, spec->action->inference_steps, spec->action->query_tokens);
  fmt::print("\n{:<12} {:<9} {:>8} {:>10} {:>12} {:>7} {:>9} {:>12}\n", "component", "modality", "tensors",
             "params(M)", "bytes", "blocks", "q-units", "q-params(M)");
  for (const auto& c : spec->components)
    fmt::print("{:<12} {:<9} {:>8} {:>10.2f} {:>12} {:>7} {:>9} {:>12.2f}\n", c.name, c.modality, c.n_tensors,
               c.params / 1e6, format_bytes(double(c.bytes)), c.n_blocks, c.n_quant_units,
               c.quant_params / 1e6);
  for (const auto& w : spec->warnings) fmt::print("warning: {}\n", w);
  if (g_remote_bytes)
    fmt::print("remote     {} of config/headers fetched (no weights)\n",
               format_bytes(double(g_remote_bytes)));
  return 0;
}

struct EstimateArgs {
  std::string model, recipe = "w4a16", hw = "nvidia/rtx4090", backend = "hf_transformers";
  std::string conc = "1", images = "1", extra;
  int64_t text_tokens = 32, context = 0, resolution = 448;
  double util = 0.9;
  bool json = false;
};

int cmd_estimate(const EstimateArgs& a) {
  auto spec = inspect_any(a.model);
  if (!spec) return fail(spec.error());
  Json recipes = Json::array();
  std::vector<std::string> recipe_names;
  {
    std::stringstream ss(a.recipe);
    std::string item;
    while (std::getline(ss, item, ',')) recipe_names.push_back(item);
  }
  auto hw = mqde::estimator::load_hardware(a.hw);
  if (!hw) return fail(hw.error());
  auto be = mqde::estimator::load_backend(a.backend);
  if (!be) return fail(be.error());

  mqde::estimator::RuntimeParams rt;
  rt.text_tokens = a.text_tokens;
  rt.context_tokens = a.context;
  rt.image_resolution = a.resolution;
  rt.gpu_memory_utilization = a.util;
  if (!a.extra.empty()) {
    std::stringstream ss(a.extra);
    std::string m;
    while (std::getline(ss, m, ',')) rt.extra_modalities.push_back(m);
  }
  const auto concs = parse_list(a.conc);
  const auto imgs = parse_list(a.images);
  rt.concurrency = concs.empty() ? 1 : concs.front();
  rt.images_per_step = imgs.empty() ? 1 : imgs.front();

  Json out = Json::array();
  for (const auto& name : recipe_names) {
    auto r = mqde::estimator::load_recipe(name);
    if (!r) return fail(r.error());
    auto e = mqde::estimator::estimate(*spec, *r, *hw, *be, rt);
    Json j = mqde::estimator::to_json(e);
    j["recipe_detail"] = mqde::estimator::to_json(*r);
    j["curves"] = {
        {"concurrency", mqde::estimator::to_json(mqde::estimator::memory_vs_concurrency(
                            *spec, *r, *hw, *be, rt,
                            concs.size() > 1 ? concs : std::vector<int64_t>{1, 2, 4, 8, 16, 32, 64}))},
        {"images", mqde::estimator::to_json(mqde::estimator::memory_vs_images(
                       *spec, *r, *hw, *be, rt,
                       imgs.size() > 1 ? imgs : std::vector<int64_t>{1, 2, 4, 8, 16, 30, 60}))}};
    out.push_back(j);
    if (a.json) continue;

    fmt::print("== {} on {} ({})  [{}]\n", r->name, hw->name, hw->arch,
               mqde::estimator::to_json(*r)["summary"].get<std::string>());
    fmt::print("   context {} tokens, concurrency {}, images/step {}\n", e.context_tokens, rt.concurrency,
               rt.images_per_step);
    fmt::print("   {:<12} {:>12} {:>12} {:>10}\n", "component", "original", "estimated", "quantized");
    for (const auto& c : e.components)
      fmt::print("   {:<12} {:>12} {:>12} {:>10}\n", c.name, format_bytes(double(c.original_bytes)),
                 format_bytes(double(c.weight_bytes)), c.quantized ? "yes" : "-");
    fmt::print(
        "   weights {} (qmeta {}) | KV+state {} ({} /token, {} state/seq) | activation {} | runtime {}\n",
        format_bytes(double(e.weights_bytes)), format_bytes(double(e.qmeta_bytes)),
        format_bytes(double(e.kv_bytes)), format_bytes(double(e.kv_bytes_per_token)),
        format_bytes(double(e.state_bytes_per_seq)), format_bytes(double(e.activation_bytes)),
        format_bytes(double(e.runtime_bytes)));
    fmt::print("   PEAK {} / capacity {}  ->  {}   max concurrency {}\n", format_bytes(double(e.peak_bytes)),
               format_bytes(double(e.capacity_bytes)), e.feasible ? "DEPLOYABLE" : "NOT DEPLOYABLE",
               e.max_concurrency);
    fmt::print("   step: {:.2f} GFLOP, {} read, latency floor {:.2f} ms ({})\n", e.flops_per_step / 1e9,
               format_bytes(e.bytes_per_step), e.latency_floor_ms, e.compute_format);
    for (const auto& reason : e.reasons) fmt::print("   note: {}\n", reason);
    fmt::print("\n");
  }
  if (a.json) std::cout << (out.size() == 1 ? out[0] : out).dump(2) << "\n";
  return 0;
}

// Local model library: every sub-directory of `root` (or `root` itself) with a config.json.
std::vector<std::filesystem::path> scan_models(const std::vector<std::string>& roots) {
  std::vector<std::filesystem::path> out;
  for (const auto& r : roots) {
    std::error_code ec;
    if (std::filesystem::is_regular_file(std::filesystem::path(r) / "config.json")) {
      out.emplace_back(r);
      continue;
    }
    for (const auto& e : std::filesystem::directory_iterator(r, ec))
      if (e.is_directory() && std::filesystem::is_regular_file(e.path() / "config.json"))
        out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

struct FitArgs {
  std::vector<std::string> models;
  std::string hw = "nvidia/rtx4090", use_case = "auto", quant, sort = "score";
  int64_t context = 0, concurrency = 1;
  int gpus = 1;
  double ram_gb = 0;
  bool json = false;
};

int cmd_fit(const FitArgs& a) {
  auto cfg = mqde::fit::load_fit_config();
  if (!cfg) return fail(cfg.error());
  auto hw = mqde::estimator::load_hardware(a.hw);
  if (!hw) return fail(hw.error());
  auto be = mqde::estimator::load_backend("hf_transformers");
  if (!be) return fail(be.error());
  auto sys = mqde::fit::system_from_profile(*hw, a.ram_gb);
  sys.gpu_count = a.gpus;
  if (sys.ram_gb == 0) {
    for (const auto& d : probe_json())
      if (d["kind"] == "cpu") sys.ram_gb = d["mem_total_bytes"].get<double>() / (1024.0 * 1024.0 * 1024.0);
  }
  for (const auto& d : probe_json())
    if (d["kind"] == "cpu") sys.cpu_cores = d["compute_units"].get<int>();
  mqde::fit::Request req;
  req.use_case = a.use_case;
  req.context = a.context;
  req.concurrency = a.concurrency;
  req.force_quant = a.quant;
  // Measured baselines only describe THIS machine: attach them when --hw is the local GPU's profile
  // and the evaluation is single-GPU (bench runs on one GPU).
  const auto host = host_identity(-1);
  std::optional<mqde::fit::BenchmarkIndex> bench;
  if (!host.gpu_name.empty() && sys.gpu_count == 1)
    if (auto local = mqde::estimator::match_hardware(host.gpu_name); local && local->id == hw->id) {
      bench = mqde::fit::BenchmarkIndex::load(host);
      req.bench = &*bench;
    }
  std::vector<mqde::fit::ModelFit> fits;
  for (const auto& m : scan_models(a.models)) {
    auto spec = inspect_any(m);
    if (!spec) {
      std::cerr << "skip " << m << ": " << spec.error().to_string() << "\n";
      continue;
    }
    auto f = mqde::fit::analyze(*spec, sys, *cfg, *be, req);
    if (!f) return fail(f.error());
    fits.push_back(*f);
  }
  mqde::fit::rank(fits, a.sort);
  if (a.json) {
    Json arr = Json::array();
    for (const auto& f : fits) arr.push_back(mqde::fit::to_json(f));
    std::cout << Json{{"system", mqde::fit::to_json(sys)}, {"models", arr}}.dump(2) << "\n";
    return 0;
  }
  fmt::print("{} x{} | {:.1f} GiB VRAM | {:.0f} GB/s | RAM {:.0f} GiB\n\n", sys.name, sys.gpu_count,
             sys.vram_gb, sys.mem_bw_gbps, sys.ram_gb);
  fmt::print("{:<40} {:>8} {:>8} {:>9} {:>16} {:>10} {:>18} {:>7}\n", "model", "params", "quant", "fit",
             "mode", "mem GiB", "speed", "score");
  for (const auto& f : fits)
    fmt::print("{:<40} {:>7.2f}B {:>8} {:>9} {:>16} {:>10.2f} {:>11.1f} {:<6} {:>7.1f}\n",
               std::filesystem::path(f.model).filename().string().substr(0, 40), f.params_b, f.best_quant,
               mqde::fit::to_string(f.fit_level), mqde::fit::to_string(f.run_mode), f.memory_required_gb,
               f.estimated_tps, f.speed_unit, f.score);
  return 0;
}

int cmd_list(const std::string& sub, bool json) {
  auto names = mqde::estimator::list_data_entries(sub);
  if (!names) return fail(names.error());
  Json arr = Json::array();
  for (const auto& n : *names) {
    if (sub == "recipes") {
      if (auto r = mqde::estimator::load_recipe(n)) arr.push_back(mqde::estimator::to_json(*r));
    } else if (sub == "hardware_db") {
      if (auto h = mqde::estimator::load_hardware(n)) arr.push_back(mqde::estimator::to_json(*h));
    }
  }
  if (json) {
    std::cout << arr.dump(2) << "\n";
    return 0;
  }
  for (const auto& j : arr) {
    if (sub == "recipes")
      fmt::print("{:<8} {:<13} {}\n", j["name"].get<std::string>(), j["maturity"].get<std::string>(),
                 j["summary"].get<std::string>());
    else
      fmt::print("{:<28} {:<26} {:<7} {:>10} {:>8.0f} GB/s\n", j["id"].get<std::string>(),
                 j["name"].get<std::string>(), j["arch"].get<std::string>(),
                 format_bytes(j["mem_bytes"].get<double>()), j["mem_bw_gbps"].get<double>());
  }
  return 0;
}

}  // namespace

struct BenchArgs {
  std::string model, dataset = "synthetic", python = "python", repo = ".";
  int gpu = 3, runs = 20, warmup = 3;
  bool dry_run = false;
};

// Runs `python -m mqde.eval.bench` on one GPU and records the result under data/benchmarks/local/.
int cmd_bench(const BenchArgs& a, bool json) {
  namespace fs = std::filesystem;
  if (!fs::is_regular_file(fs::path(a.model) / "config.json"))
    return fail(mqde::Error{mqde::Errc::kNotFound, a.model + " has no config.json"});
  const auto repo = fs::absolute(a.repo);
  if (!fs::is_directory(repo / "python" / "mqde"))
    return fail(mqde::Error{mqde::Errc::kNotFound,
                            "--repo must point at the repository root (python/mqde not found)"});
  double vram = 0, ram = 0;
  int cores = 0;
  const auto hw = host_identity(a.gpu, &vram, &ram, &cores);
  if (hw.gpu_name.empty())
    return fail(
        mqde::Error{mqde::Errc::kNotFound, fmt::format("no GPU with index {} on this machine", a.gpu)});

  int out_pipe[2];
  if (::pipe(out_pipe) != 0) return fail(mqde::Error{mqde::Errc::kIo, std::strerror(errno)});
  const std::string pythonpath = (repo / "python").string();
  const std::string runs = std::to_string(a.runs), warmup = std::to_string(a.warmup);
  const pid_t pid = ::fork();
  if (pid < 0) return fail(mqde::Error{mqde::Errc::kIo, std::strerror(errno)});
  if (pid == 0) {
    ::dup2(out_pipe[1], STDOUT_FILENO);  // stderr stays on the terminal (progress + errors)
    ::close(out_pipe[0]);
    ::close(out_pipe[1]);
    ::setenv("CUDA_VISIBLE_DEVICES", std::to_string(a.gpu).c_str(), 1);
    const char* old = std::getenv("PYTHONPATH");
    ::setenv("PYTHONPATH", old && *old ? (pythonpath + ":" + old).c_str() : pythonpath.c_str(), 1);
    ::setenv("MQDE_DATA_DIR", mqde::data_dir().c_str(), 1);
    ::setenv("PYTHONUNBUFFERED", "1", 1);
    ::execlp(a.python.c_str(), a.python.c_str(), "-m", "mqde.eval.bench", a.model.c_str(), "--runs",
             runs.c_str(), "--warmup", warmup.c_str(), "--dataset", a.dataset.c_str(),
             static_cast<char*>(nullptr));
    std::fprintf(stderr, "exec %s failed: %s\n", a.python.c_str(), std::strerror(errno));
    ::_exit(127);
  }
  ::close(out_pipe[1]);
  std::string out;
  char buf[4096];
  for (ssize_t n; (n = ::read(out_pipe[0], buf, sizeof buf)) > 0;) out.append(buf, static_cast<size_t>(n));
  ::close(out_pipe[0]);
  int status = 0;
  ::waitpid(pid, &status, 0);

  // The contract is "one JSON object on the last stdout line".
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  const auto nl = out.rfind('\n');
  Json r;
  try {
    r = Json::parse(nl == std::string::npos ? out : out.substr(nl + 1));
  } catch (const std::exception&) {
    return fail(mqde::Error{
        mqde::Errc::kExternal,
        fmt::format("benchmark produced no result (exit {})", WIFEXITED(status) ? WEXITSTATUS(status) : -1)});
  }
  if (r.contains("error")) return fail(mqde::Error{mqde::Errc::kExternal, r["error"].get<std::string>()});
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return fail(mqde::Error{mqde::Errc::kExternal, "benchmark exited abnormally"});

  mqde::fit::Submission sub;
  sub.hw = hw;
  sub.vram_gb = vram;
  sub.ram_gb = ram;
  sub.cpu_cores = cores;
  sub.tool_version = MQDE_VERSION;
  mqde::fit::Submission::Entry e;
  e.model = a.model;
  e.quant = r.value("quant", "bf16");
  e.runtime = r.value("runtime", "hf_transformers");
  e.unit = r.value("unit", "tok/s");
  e.avg_tps = r.value("avg_tps", 0.0);
  e.min_tps = r.value("min_tps", 0.0);
  e.max_tps = r.value("max_tps", 0.0);
  e.num_runs = r.value("num_runs", 0);
  e.peak_mem_bytes = r.value("peak_mem_bytes", int64_t{0});
  e.context_tokens = r.value("context_tokens", int64_t{0});
  // For step-based models the recorded "TTFT" is the mean decision-step latency (schema.json).
  if (e.unit == "steps/s" && e.avg_tps > 0) e.avg_ttft_ms = 1e3 / e.avg_tps;
  sub.results.push_back(e);

  const Json doc = mqde::fit::to_json(sub);  // exactly what write_local_submission stores
  if (a.dry_run) {
    std::cout << doc.dump(2) << "\n";
    return 0;
  }
  auto path = mqde::fit::write_local_submission(sub);
  if (!path) return fail(path.error());
  if (json) {
    std::cout << Json{{"path", path->string()}, {"submission", doc}}.dump(2) << "\n";
    return 0;
  }
  fmt::print("{}  {}  {:.2f} {} (min {:.2f}, max {:.2f}, {} runs) on {}\nwrote {}\n",
             fs::path(a.model).filename().string(), e.quant, e.avg_tps, e.unit, e.min_tps, e.max_tps,
             e.num_runs, hw.gpu_name, path->string());
  return 0;
}

// ---- catalog / hf-search / download (T-R25) ----

int cmd_catalog_update(const mqde::hub::HfOptions& ho, const mqde::hub::UpdateOptions& uo, bool json) {
  auto base = mqde::hub::load_catalog();
  if (!base) return fail(base.error());
  auto c = mqde::hub::update_catalog(ho, uo, *base, [](size_t n, size_t total, const std::string& id) {
    fmt::print(stderr, "\r[{}/{}] {:<60}", n, total, id.substr(0, 60));
  });
  fmt::print(stderr, "\n");
  if (!c) return fail(c.error());
  if (auto r = mqde::hub::save_catalog(*c); !r) return fail(r.error());
  size_t demoted = 0;
  for (const auto& e : c->entries) demoted += e.demoted;
  if (json) {
    std::cout << Json{{"path", mqde::hub::default_catalog_path().string()},
                      {"total", c->entries.size()},
                      {"demoted", demoted},
                      {"updated_at", c->updated_at}}
                     .dump(2)
              << "\n";
  } else {
    fmt::print("wrote {} ({} entries, {} demoted)\n", mqde::hub::default_catalog_path().string(),
               c->entries.size(), demoted);
  }
  return 0;
}

void print_catalog_rows(const std::vector<Json>& rows) {
  fmt::print("{:<52} {:>8} {:>6} {:>8} {:>12}  {}\n", "model", "params", "quant", "ctx", "downloads", "note");
  for (const auto& r : rows)
    fmt::print("{:<52} {:>8} {:>6} {:>8} {:>12}  {}\n", r.value("name", "").substr(0, 52),
               r.value("parameter_count", ""), r.value("quantization", ""), r.value("context_length", 0),
               r.value("hf_downloads", 0), r.value("demote_reason", ""));
}

int cmd_catalog_search(const std::string& q, const std::string& provider, size_t n, bool demoted, bool json) {
  auto c = mqde::hub::load_catalog();
  if (!c) return fail(c.error());
  std::vector<Json> rows;
  for (const auto* e : mqde::hub::search(*c, q, provider, "", n, demoted))
    rows.push_back(mqde::hub::to_json(*e));
  if (json)
    std::cout << Json(rows).dump(2) << "\n";
  else
    print_catalog_rows(rows);
  return 0;
}

int cmd_hf_search(const mqde::hub::HfOptions& ho, const std::string& q, int n, bool json) {
  auto l =
      mqde::hub::list_models(ho, "search=" + mqde::hub::url_encode(q) + "&limit=" + std::to_string(n) +
                                     "&sort=downloads&direction=-1&expand[]=safetensors&expand[]=downloads"
                                     "&expand[]=pipeline_tag&expand[]=author");
  if (!l) return fail(l.error());
  if (json) {
    std::cout << l->dump(2) << "\n";
    return 0;
  }
  fmt::print("{:<56} {:>10} {:>12}  {}\n", "model", "params", "downloads", "pipeline");
  for (const auto& m : *l) {
    const int64_t p = m.contains("safetensors") ? m["safetensors"].value("total", int64_t{0}) : 0;
    fmt::print("{:<56} {:>9.2f}B {:>12} {}\n", m.value("id", "").substr(0, 56), static_cast<double>(p) / 1e9,
               m.value("downloads", 0), m.value("pipeline_tag", ""));
  }
  return 0;
}

struct DownloadArgs {
  std::string model, quant, budget, output_dir = ".";
  bool list = false;
};

int cmd_download(const mqde::hub::HfOptions& ho, const DownloadArgs& a, bool json) {
  std::string repo = a.model.starts_with("hf:") ? a.model.substr(3) : a.model;
  auto files = mqde::hub::list_files(ho, repo);
  if (!files) return fail(files.error());
  const auto groups = mqde::hub::group_files(*files);
  if (groups.empty())
    return fail(mqde::Error{mqde::Errc::kNotFound, "no safetensors / gguf weights in " + repo});
  uint64_t budget = 0;
  if (!a.budget.empty()) {
    auto b = mqde::hub::parse_size(a.budget);
    if (!b) return fail(mqde::Error{mqde::Errc::kInvalidArgument, "bad --budget '" + a.budget + "'"});
    budget = *b;
  }
  const auto g = mqde::hub::choose_group(groups, budget, a.quant);
  if (a.list || !g) {
    if (json) {
      Json arr = Json::array();
      for (const auto& x : groups)
        arr.push_back({{"group", x.key}, {"bytes", x.bytes}, {"files", x.files.size()}});
      std::cout << Json{{"groups", arr}, {"chosen", g ? g->key : ""}}.dump(2) << "\n";
    } else {
      for (const auto& x : groups)
        fmt::print("{} {:<14} {:>10}  {} file(s)\n", g && x.key == g->key ? "*" : " ", x.key,
                   format_bytes(x.bytes), x.files.size());
    }
    return g ? 0 : fail(mqde::Error{mqde::Errc::kNotFound, "quant '" + a.quant + "' not found in " + repo});
  }
  if (budget > 0 && g->bytes > budget)
    fmt::print(stderr, "note: nothing fits {} — downloading the smallest group ({})\n", a.budget, g->key);
  const auto dir = std::filesystem::path(a.output_dir) / repo.substr(repo.find('/') + 1);
  uint64_t done = 0;
  constexpr int kDownloadChunks = 4;  // download_file_chunked falls back to a single stream on its own for
                                      // small files or a server that does not support Range
  for (const auto& f : g->files) {
    auto r = mqde::hub::download_file_chunked(
        ho, repo, f.path, dir / f.path, kDownloadChunks, [&](uint64_t cur, uint64_t) {
          fmt::print(stderr, "\r{} / {}  {:<40}", format_bytes(done + cur), format_bytes(g->bytes),
                     f.path.substr(0, 40));
          return true;
        });
    if (!r) {
      fmt::print(stderr, "\n");
      return fail(r.error());
    }
    done += *r;
  }
  fmt::print(stderr, "\n");
  if (json)
    std::cout << Json{{"dir", dir.string()}, {"group", g->key}, {"bytes", done}}.dump(2) << "\n";
  else
    fmt::print("downloaded {} ({}, {}) to {}\n", repo, g->key, format_bytes(done), dir.string());
  return 0;
}

int main(int argc, char** argv) {
  CLI::App app{"MQDEsys - model quantization & deployment evaluation"};
  app.require_subcommand(1);
  std::string log_level = "warn";
  bool json = false;
  app.add_option("--log-level", log_level, "trace|debug|info|warn|error");
  app.add_flag("--json", json, "machine-readable JSON output");
  app.set_version_flag("--version", std::string("mqde ") + MQDE_VERSION);

  auto* probe = app.add_subcommand("probe", "detect CPUs / GPUs / NPUs on this machine");

  auto* model = app.add_subcommand("model", "model utilities");
  model->require_subcommand(1);
  auto* inspect =
      model->add_subcommand("inspect", "parse config.json + safetensors headers (no weights loaded)");
  std::string model_path;
  bool with_tensors = false;
  inspect->add_option("path", model_path, "local model directory, or hf:org/name[@revision]")->required();
  inspect->add_flag("--tensors", with_tensors, "include per-tensor listing (JSON only)");

  auto* est = app.add_subcommand("estimate", "estimate memory / KV / concurrency / latency floor");
  EstimateArgs ea;
  est->add_option("-m,--model", ea.model, "local model directory, or hf:org/name[@revision]")->required();
  est->add_option("-r,--recipe", ea.recipe,
                  "recipe name(s), comma separated (w4a16,w4a8,w2a16,w2a8,fp8,...)");
  est->add_option("--hw", ea.hw, "hardware profile id (see `mqde hardware`)");
  est->add_option("--backend", ea.backend, "backend profile id");
  est->add_option("--conc", ea.conc, "concurrency; a list (1,8,32) also produces a curve");
  est->add_option("--images", ea.images, "images (views x frames) per step; a list produces a curve");
  est->add_option("--text-tokens", ea.text_tokens, "instruction tokens");
  est->add_option("--ctx", ea.context, "override language context tokens");
  est->add_option("--resolution", ea.resolution, "input image resolution");
  est->add_option("--util", ea.util, "usable fraction of device memory");
  est->add_option("--also-quantize", ea.extra, "extra modalities to quantize, e.g. action");

  auto* fitc = app.add_subcommand("fit", "llmfit-style deployment pre-evaluation over a local model library");
  FitArgs fa;
  fitc->add_option("models", fa.models, "model directories or library roots")->required();
  fitc->add_option("--hw", fa.hw, "hardware profile id");
  fitc->add_option("--gpus", fa.gpus, "number of identical GPUs");
  fitc->add_option("--ram", fa.ram_gb, "system RAM in GiB (default: probed)");
  fitc->add_option("--use-case", fa.use_case, "auto|general|chat|coding|reasoning|multimodal|vla");
  fitc->add_option("--ctx", fa.context, "context tokens");
  fitc->add_option("--conc", fa.concurrency, "concurrent sequences");
  fitc->add_option("--quant", fa.quant, "force a recipe instead of choosing the best that fits");
  fitc->add_option("--sort", fa.sort, "score|tps|params|mem");

  auto* bench = app.add_subcommand(
      "bench", "measure real throughput on one GPU and record it in data/benchmarks/local/");
  BenchArgs ba;
  bench->add_option("model", ba.model, "local model directory")->required();
  bench->add_option("--gpu", ba.gpu, "physical GPU index (CUDA_VISIBLE_DEVICES)");
  bench->add_option("--runs", ba.runs, "measured steps");
  bench->add_option("--warmup", ba.warmup, "unmeasured warm-up steps");
  bench->add_option("--dataset", ba.dataset, "'synthetic' or a LeRobot v3 dataset directory");
  bench->add_option("--python", ba.python, "python executable of the mqde environment");
  bench->add_option("--repo", ba.repo, "repository root (for python/)");
  bench->add_flag("--dry-run", ba.dry_run, "print the submission instead of writing it");

  auto* claim = app.add_subcommand("claim", "compute conservative/optimistic throughput bounds for a model");
  std::string claim_model, claim_hw, claim_quant;
  double claim_min_tps = 1.0;
  claim->add_option("model", claim_model, "local model directory")->required();
  claim->add_option("--hardware,-H", claim_hw, "hardware profile (e.g. nvidia/rtx4090)")
      ->default_val("nvidia/rtx4090");
  claim->add_option("--min-tps", claim_min_tps, "minimum acceptable throughput")->default_val(1.0);

  auto* serve = app.add_subcommand("serve", "start the REST/SSE API server (+ web UI)");
  mqde::server::ServerOptions so;
  std::string gpus;
  std::vector<std::string> allow;
  serve->add_option("--host", so.host, "bind address (default loopback; use ssh -L for remote access)");
  serve->add_option("--port", so.port);
  serve->add_option("--runs", so.runs_dir, "job directory");
  serve->add_option("--web", so.web_dir, "built frontend directory (web/dist)");
  serve->add_option("--repo", so.repo_root, "repository root (python/ and data/)")->required();
  serve->add_option("--python", so.python, "python executable for the executor");
  serve->add_option("--gpus", gpus, "GPU pool for jobs, e.g. 2,3");
  serve->add_option("--allow", allow, "allowed root for model/dataset paths (repeatable)");

  // HF token: --token or HF_TOKEN. Never printed.
  mqde::hub::HfOptions ho;
  auto* catalog = app.add_subcommand("catalog", "HF model catalog (data/models/catalog.json)");
  catalog->require_subcommand(1);
  auto* cat_update =
      catalog->add_subcommand("update", "fetch trending + most-downloaded models and their configs");
  mqde::hub::UpdateOptions uo;
  bool only_trending = false, only_downloads = false;
  cat_update->add_option("--limit", uo.limit, "number of entries to fetch")->default_val(500);
  cat_update->add_flag("--trending", only_trending, "only the trending list");
  cat_update->add_flag("--downloads", only_downloads, "only the most-downloaded list");
  cat_update->add_option("--token", ho.token, "HF token (default: $HF_TOKEN)");
  auto* cat_search = catalog->add_subcommand("search", "search the local catalog");
  std::string cq, cprov;
  size_t cn = 20;
  bool cdemoted = false;
  cat_search->add_option("query", cq, "substring of the repo name");
  cat_search->add_option("--provider", cprov, "exact provider / org");
  cat_search->add_option("-n", cn, "max rows");
  cat_search->add_flag("--include-demoted", cdemoted, "also show draft / incomplete entries");

  auto* hfs = app.add_subcommand("hf-search", "search huggingface.co live (needs network)");
  std::string hq;
  int hn = 20;
  hfs->add_option("query", hq)->required();
  hfs->add_option("-n", hn, "max rows");
  hfs->add_option("--token", ho.token, "HF token (default: $HF_TOKEN)");

  auto* dl = app.add_subcommand("download", "download a model's weights (best quant within a budget)");
  DownloadArgs da;
  dl->add_option("model", da.model, "<org>/<name> or hf:<org>/<name>")->required();
  dl->add_option("--quant", da.quant, "GGUF quant tag (Q4_K_M ...) or 'safetensors'");
  dl->add_option("--budget", da.budget, "max download size, e.g. 12G");
  dl->add_option("--output-dir", da.output_dir, "parent directory");
  dl->add_flag("--list", da.list, "list file groups and exit");
  dl->add_option("--token", ho.token, "HF token (default: $HF_TOKEN)");

  auto* recipes = app.add_subcommand("recipes", "list quantization recipes");
  auto* hardware = app.add_subcommand("hardware", "list hardware profiles");

  CLI11_PARSE(app, argc, argv);
  mqde::log::init(log_level);
  ea.json = json;

  if (*probe) return cmd_probe(json);
  if (*serve) {
    mqde::log::init(log_level == "warn" ? "info" : log_level);
    for (auto g : parse_list(gpus)) so.gpus.push_back(static_cast<int>(g));
    so.allowed_roots = allow;
    auto r = mqde::server::serve(so);
    return r ? 0 : fail(r.error());
  }
  if (*inspect) return cmd_inspect(model_path, json, with_tensors);
  if (*est) return cmd_estimate(ea);
  if (*bench) return cmd_bench(ba, json);
  if (*fitc) {
    fa.json = json;
    return cmd_fit(fa);
  }
  if (*claim) {
    auto spec = inspect_any(claim_model);
    if (!spec) {
      fmt::print(stderr, "error: {}\n", spec.error().to_string());
      return 1;
    }
    auto hw = mqde::estimator::load_hardware(claim_hw);
    if (!hw) {
      fmt::print(stderr, "error: {}\n", hw.error().to_string());
      return 1;
    }
    auto be = mqde::estimator::BackendProfile{};
    auto cfg = mqde::fit::load_fit_config();
    if (!cfg) {
      fmt::print(stderr, "error: {}\n", cfg.error().to_string());
      return 1;
    }
    auto sys = mqde::fit::system_from_profile(*hw);
    mqde::fit::Request cr;
    std::optional<mqde::fit::BenchmarkIndex> cbench;
    const auto host = host_identity(-1);
    if (!host.gpu_name.empty())
      if (auto local = mqde::estimator::match_hardware(host.gpu_name); local && local->id == hw->id) {
        cbench = mqde::fit::BenchmarkIndex::load(host);
        cr.bench = &*cbench;
      }
    auto b = mqde::fit::fit_bounds(*spec, sys, *cfg, be, claim_min_tps, cr);
    if (json) {
      fmt::print("{}\n", mqde::fit::to_json(b).dump(2));
    } else {
      fmt::print("model: {}\n", spec->path);
      fmt::print("hardware: {}\n", claim_hw);
      fmt::print("quant: {}\n", b.best_quant);
      fmt::print("conservative: {:.2f} {}  optimistic: {:.2f} {}  (basis: {})\n", b.conservative_tps,
                 b.speed_unit, b.optimistic_tps, b.speed_unit, b.basis);
      fmt::print("holds (>= {:.1f} {}): {}\n", claim_min_tps, b.speed_unit, b.holds ? "yes" : "no");
      if (b.holds) fmt::print("max context at threshold: {} tokens\n", b.max_context_at_min_tps);
      if (b.basis == "formula")
        fmt::print("memory: {:.2f} GiB  efficiency assumed: {:.0f}%\n", b.memory_gb, b.efficiency_used * 100);
      else
        fmt::print("memory: {:.2f} GiB  bounds: measured ±10%\n", b.memory_gb);
    }
    return 0;
  }
  if (*cat_update) {
    if (only_trending != only_downloads) uo.trending = only_trending, uo.downloads = only_downloads;
    return cmd_catalog_update(ho, uo, json);
  }
  if (*cat_search) return cmd_catalog_search(cq, cprov, cn, cdemoted, json);
  if (*hfs) return cmd_hf_search(ho, hq, hn, json);
  if (*dl) return cmd_download(ho, da, json);
  if (*recipes) return cmd_list("recipes", json);
  if (*hardware) return cmd_list("hardware_db", json);
  return 0;
}
