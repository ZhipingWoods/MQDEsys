// mqde - command line entry point (docs/ARCHITECTURE.md §6.16).
// stdout carries command output (tables or --json); diagnostics go to stderr.

#include <cstdio>
#include <iostream>
#include <sstream>

#include <CLI/CLI.hpp>

#include "mqde/common/log.h"
#include "mqde/estimator/estimator.h"
#include "mqde_probe.h"

namespace {

using mqde::Json;
using mqde::format_bytes;

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
    Json j = {{"kind", d.kind == MQDE_DEV_CPU ? "cpu" : d.kind == MQDE_DEV_GPU ? "gpu" : "other"},
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

int cmd_probe(bool json) {
  const Json devs = probe_json();
  if (json) {
    std::cout << devs.dump(2) << "\n";
    return 0;
  }
  fmt::print("{:<4} {:<5} {:<40} {:<8} {:>12} {:>12}  {}\n", "idx", "kind", "name", "arch", "total", "free", "profile");
  for (const auto& d : devs)
    fmt::print("{:<4} {:<5} {:<40} {:<8} {:>12} {:>12}  {}\n", d["index"].get<int>(), d["kind"].get<std::string>(),
               d["name"].get<std::string>(), d["arch"].get<std::string>(),
               format_bytes(d["mem_total_bytes"].get<double>()), format_bytes(d["mem_free_bytes"].get<double>()),
               d.value("profile", "-"));
  return 0;
}

int cmd_inspect(const std::string& path, bool json, bool tensors) {
  auto spec = mqde::modelspec::inspect_local(path);
  if (!spec) return fail(spec.error());
  if (json) {
    std::cout << mqde::modelspec::to_json(*spec, tensors).dump(2) << "\n";
    return 0;
  }
  const auto& t = spec->text;
  fmt::print("model      {}\nfamily     {} ({})\ntype       {}\nparams     {:.3f} B   ({} on disk)\n", spec->path,
             spec->family, spec->modality, spec->model_type, spec->total_params / 1e9, format_bytes(double(spec->total_bytes)));
  fmt::print("language   {} layers ({} full-attn, {} linear-attn), hidden {}, heads {}/{} kv, head_dim {}, vocab {}\n",
             t.n_layers, t.n_full_attention, t.n_linear_attention, t.hidden, t.n_heads, t.n_kv_heads, t.head_dim, t.vocab);
  if (spec->vision)
    fmt::print("vision     {} layers, hidden {}, patch {}, {} tokens/image\n", spec->vision->n_layers, spec->vision->hidden,
               spec->vision->patch_size, spec->vision->tokens_per_image);
  if (spec->action)
    fmt::print("action     horizon {} x dim {}, {} denoising steps, {} query tokens\n", spec->action->horizon,
               spec->action->action_dim, spec->action->inference_steps, spec->action->query_tokens);
  fmt::print("\n{:<12} {:<9} {:>8} {:>10} {:>12} {:>7} {:>9} {:>12}\n", "component", "modality", "tensors", "params(M)",
             "bytes", "blocks", "q-units", "q-params(M)");
  for (const auto& c : spec->components)
    fmt::print("{:<12} {:<9} {:>8} {:>10.2f} {:>12} {:>7} {:>9} {:>12.2f}\n", c.name, c.modality, c.n_tensors,
               c.params / 1e6, format_bytes(double(c.bytes)), c.n_blocks, c.n_quant_units, c.quant_params / 1e6);
  for (const auto& w : spec->warnings) fmt::print("warning: {}\n", w);
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
  auto spec = mqde::modelspec::inspect_local(a.model);
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
        {"concurrency", mqde::estimator::to_json(mqde::estimator::memory_vs_concurrency(*spec, *r, *hw, *be, rt,
                                                                                        concs.size() > 1 ? concs : std::vector<int64_t>{1, 2, 4, 8, 16, 32, 64}))},
        {"images", mqde::estimator::to_json(mqde::estimator::memory_vs_images(*spec, *r, *hw, *be, rt,
                                                                              imgs.size() > 1 ? imgs : std::vector<int64_t>{1, 2, 4, 8, 16, 30, 60}))}};
    out.push_back(j);
    if (a.json) continue;

    fmt::print("== {} on {} ({})  [{}]\n", r->name, hw->name, hw->arch, mqde::estimator::to_json(*r)["summary"].get<std::string>());
    fmt::print("   context {} tokens, concurrency {}, images/step {}\n", e.context_tokens, rt.concurrency, rt.images_per_step);
    fmt::print("   {:<12} {:>12} {:>12} {:>10}\n", "component", "original", "estimated", "quantized");
    for (const auto& c : e.components)
      fmt::print("   {:<12} {:>12} {:>12} {:>10}\n", c.name, format_bytes(double(c.original_bytes)),
                 format_bytes(double(c.weight_bytes)), c.quantized ? "yes" : "-");
    fmt::print("   weights {} (qmeta {}) | KV+state {} ({} /token, {} state/seq) | activation {} | runtime {}\n",
               format_bytes(double(e.weights_bytes)), format_bytes(double(e.qmeta_bytes)), format_bytes(double(e.kv_bytes)),
               format_bytes(double(e.kv_bytes_per_token)), format_bytes(double(e.state_bytes_per_seq)),
               format_bytes(double(e.activation_bytes)), format_bytes(double(e.runtime_bytes)));
    fmt::print("   PEAK {} / capacity {}  ->  {}   max concurrency {}\n", format_bytes(double(e.peak_bytes)),
               format_bytes(double(e.capacity_bytes)), e.feasible ? "DEPLOYABLE" : "NOT DEPLOYABLE", e.max_concurrency);
    fmt::print("   step: {:.2f} GFLOP, {} read, latency floor {:.2f} ms ({})\n", e.flops_per_step / 1e9,
               format_bytes(e.bytes_per_step), e.latency_floor_ms, e.compute_format);
    for (const auto& reason : e.reasons) fmt::print("   note: {}\n", reason);
    fmt::print("\n");
  }
  if (a.json) std::cout << (out.size() == 1 ? out[0] : out).dump(2) << "\n";
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
      fmt::print("{:<28} {:<26} {:<7} {:>10} {:>8.0f} GB/s\n", j["id"].get<std::string>(), j["name"].get<std::string>(),
                 j["arch"].get<std::string>(), format_bytes(j["mem_bytes"].get<double>()), j["mem_bw_gbps"].get<double>());
  }
  return 0;
}

}  // namespace

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
  auto* inspect = model->add_subcommand("inspect", "parse config.json + safetensors headers (no weights loaded)");
  std::string model_path;
  bool with_tensors = false;
  inspect->add_option("path", model_path, "local model directory")->required();
  inspect->add_flag("--tensors", with_tensors, "include per-tensor listing (JSON only)");

  auto* est = app.add_subcommand("estimate", "estimate memory / KV / concurrency / latency floor");
  EstimateArgs ea;
  est->add_option("-m,--model", ea.model, "local model directory")->required();
  est->add_option("-r,--recipe", ea.recipe, "recipe name(s), comma separated (w4a16,w4a8,w2a16,w2a8,fp8,...)");
  est->add_option("--hw", ea.hw, "hardware profile id (see `mqde hardware`)");
  est->add_option("--backend", ea.backend, "backend profile id");
  est->add_option("--conc", ea.conc, "concurrency; a list (1,8,32) also produces a curve");
  est->add_option("--images", ea.images, "images (views x frames) per step; a list produces a curve");
  est->add_option("--text-tokens", ea.text_tokens, "instruction tokens");
  est->add_option("--ctx", ea.context, "override language context tokens");
  est->add_option("--resolution", ea.resolution, "input image resolution");
  est->add_option("--util", ea.util, "usable fraction of device memory");
  est->add_option("--also-quantize", ea.extra, "extra modalities to quantize, e.g. action");

  auto* recipes = app.add_subcommand("recipes", "list quantization recipes");
  auto* hardware = app.add_subcommand("hardware", "list hardware profiles");

  CLI11_PARSE(app, argc, argv);
  mqde::log::init(log_level);
  ea.json = json;

  if (*probe) return cmd_probe(json);
  if (*inspect) return cmd_inspect(model_path, json, with_tensors);
  if (*est) return cmd_estimate(ea);
  if (*recipes) return cmd_list("recipes", json);
  if (*hardware) return cmd_list("hardware_db", json);
  return 0;
}
