#include "mqde/hub/catalog.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <regex>
#include <set>
#include <thread>

namespace mqde::hub {

namespace fs = std::filesystem;

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

// config.json keys the estimator reads (model_spec.cpp). Everything else is dropped to keep the file small.
const std::vector<std::string>& kept_keys() {
  static const std::vector<std::string> k = {
      "model_type", "architectures", "num_hidden_layers", "hidden_size", "intermediate_size",
      "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
      "max_sequence_length", "seq_length", "n_positions", "tie_word_embeddings", "torch_dtype",
      "num_experts", "n_routed_experts", "num_local_experts", "num_experts_per_tok", "moe_topk",
      "moe_intermediate_size", "shared_expert_intermediate_size", "n_shared_experts", "layer_types",
      "sliding_window", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
      "linear_value_head_dim", "linear_conv_kernel_dim", "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim",
      "qk_nope_head_dim", "v_head_dim", "n_layer", "n_embd", "n_head", "n_ctx", "n_inner", "d_model",
      // vision
      "depth", "embed_dim", "patch_size", "image_size", "num_layers"};
  return k;
}

Json subset(const Json& c) {
  Json out = Json::object();
  if (!c.is_object()) return out;
  for (const auto& k : kept_keys())
    if (c.contains(k)) out[k] = c[k];
  // quantization_config: only the scheme, not per-module ignore lists (those were ~70 % of the file).
  if (c.contains("quantization_config") && c["quantization_config"].is_object()) {
    Json q = Json::object();
    for (const char* k : {"quant_method", "bits", "group_size", "fmt", "weight_block_size", "quant_algo"})
      if (c["quantization_config"].contains(k)) q[k] = c["quantization_config"][k];
    out["quantization_config"] = q;
  }
  for (const char* nested : {"text_config", "vision_config", "llm_config"})
    if (c.contains(nested) && c[nested].is_object()) out[nested] = subset(c[nested]);
  return out;
}

// Reads a key from the top level or text_config / llm_config.
int64_t cfg_int(const Json& c, std::initializer_list<const char*> keys) {
  for (const Json* j : {&c, c.contains("text_config") ? &c["text_config"] : nullptr,
                        c.contains("llm_config") ? &c["llm_config"] : nullptr}) {
    if (!j || !j->is_object()) continue;
    for (const char* k : keys)
      if (j->contains(k) && (*j)[k].is_number()) return (*j)[k].get<int64_t>();
  }
  return 0;
}

std::string human_params(int64_t n) {
  if (n <= 0) return "";
  const double b = static_cast<double>(n) / 1e9;
  if (b < 1.0) return fmt::format("{:.0f}M", static_cast<double>(n) / 1e6);
  std::string s = fmt::format("{:.1f}", b);
  if (s.ends_with(".0")) s.resize(s.size() - 2);
  return s + "B";
}

// "Qwen3-30B-A3B" -> 30e9 (first size token in the repo name); nullopt when the name has none.
std::optional<double> name_params(const std::string& name) {
  static const std::regex re(R"((?:^|[-_/.])(\d+(?:\.\d+)?)([BbMm])(?:[-_.]|$))");
  std::smatch m;
  const std::string base = name.substr(name.rfind('/') == std::string::npos ? 0 : name.rfind('/') + 1);
  if (!std::regex_search(base, m, re)) return std::nullopt;
  const double v = std::stod(m[1].str());
  return std::tolower(m[2].str()[0]) == 'b' ? v * 1e9 : v * 1e6;
}

std::string now_iso() {
  const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

}  // namespace

fs::path default_catalog_path() { return data_dir() / "models" / "catalog.json"; }

void apply_hygiene(CatalogEntry& e) {
  e.demoted = false;
  e.demote_reason.clear();
  const std::string n = lower(e.name);
  auto demote = [&](std::string why) {
    e.demoted = true;
    e.demote_reason = std::move(why);
  };
  if (n.find("-draft") != std::string::npos || n.find("eagle") != std::string::npos ||
      n.find("speculator") != std::string::npos)
    return demote("draft / speculative-decoding model");
  if (e.parameters_raw <= 0) return demote("missing parameter count");
  if (e.context_length <= 0) return demote("missing context length");
  if (cfg_int(e.config, {"num_hidden_layers", "n_layer", "num_layers"}) <= 0 ||
      cfg_int(e.config, {"hidden_size", "n_embd", "d_model"}) <= 0)
    return demote("missing model dimensions");
  if (auto np = name_params(e.name)) {
    const double r = static_cast<double>(e.parameters_raw) / *np;
    if (r > 2.0 || r < 0.5) return demote(fmt::format("name says {} but weights hold {}", human_params(static_cast<int64_t>(*np)),
                                                     human_params(e.parameters_raw)));
  }
}

CatalogEntry entry_from_hub(const Json& info, const Json& config) {
  CatalogEntry e;
  e.name = info.value("id", info.value("modelId", ""));
  e.provider = info.value("author", e.name.substr(0, e.name.find('/')));
  e.pipeline_tag = info.value("pipeline_tag", "");
  e.hf_downloads = info.value("downloads", int64_t{0});
  e.hf_likes = info.value("likes", int64_t{0});
  e.release_date = info.value("createdAt", "").substr(0, 10);
  if (info.contains("cardData") && info["cardData"].is_object()) {
    const auto& lic = info["cardData"].value("license", Json());
    e.license = lic.is_string() ? lic.get<std::string>() : lic.is_array() && !lic.empty() && lic[0].is_string() ? lic[0].get<std::string>() : "";
  }
  e.config = subset(config);
  if (config.contains("architectures") && config["architectures"].is_array() && !config["architectures"].empty())
    e.architecture = config["architectures"][0].get<std::string>();
  // Parameters and storage dtype from the safetensors summary.
  std::string dominant;
  if (info.contains("safetensors") && info["safetensors"].is_object()) {
    const auto& st = info["safetensors"];
    e.parameters_raw = st.value("total", int64_t{0});
    int64_t best = -1;
    if (st.contains("parameters") && st["parameters"].is_object())
      for (const auto& [k, v] : st["parameters"].items())
        if (v.is_number() && v.get<int64_t>() > best) best = v.get<int64_t>(), dominant = k;
  }
  e.format = "safetensors";
  const std::string qm = config.contains("quantization_config") && config["quantization_config"].is_object()
                             ? lower(config["quantization_config"].value("quant_method", ""))
                             : "";
  if (qm == "fp8" || dominant.starts_with("F8")) e.quantization = "FP8";
  else if (!qm.empty() || dominant == "I32" || dominant == "U8" || dominant == "I8") {
    const int bits = config["quantization_config"].is_object() ? config["quantization_config"].value("bits", 4) : 4;
    e.quantization = fmt::format("INT{}", bits);
  } else if (dominant == "U32") e.quantization = "INT4";  // MLX packed
  else if (dominant == "F32") e.quantization = "F32";
  else if (dominant == "F16") e.quantization = "F16";
  else e.quantization = "BF16";
  // Packed INT / MLX checkpoints (I32/U32/U8 containers) report packed element counts, not logical
  // parameters. When the repo name states a size and the raw count is well below it, trust the name.
  if (e.quantization.starts_with("INT") || dominant == "U32" || dominant == "I32" || dominant == "U8") {
    if (auto np = name_params(e.name); np && static_cast<double>(e.parameters_raw) < 0.6 * *np) {
      e.parameters_raw = static_cast<int64_t>(*np);
      if (!e.quantization.starts_with("INT")) e.quantization = "INT4";
    }
  }
  e.parameter_count = human_params(e.parameters_raw);
  e.context_length = cfg_int(config, {"max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"});
  // Capabilities (llmfit capabilities[]): text always; vision for VLM pipelines / vision towers; MoE from config.
  e.capabilities.push_back("text");
  if (e.pipeline_tag == "image-text-to-text" || config.contains("vision_config")) e.capabilities.push_back("vision");
  if (cfg_int(config, {"num_experts", "n_routed_experts", "num_local_experts"}) > 0) e.capabilities.push_back("moe");
  e.use_case = e.pipeline_tag == "image-text-to-text" ? "multimodal" : "general";
  apply_hygiene(e);
  return e;
}

Json to_json(const CatalogEntry& e, bool with_config) {
  Json j = {{"name", e.name},
            {"provider", e.provider},
            {"parameter_count", e.parameter_count},
            {"parameters_raw", e.parameters_raw},
            {"quantization", e.quantization},
            {"format", e.format},
            {"context_length", e.context_length},
            {"architecture", e.architecture},
            {"pipeline_tag", e.pipeline_tag},
            {"license", e.license},
            {"release_date", e.release_date},
            {"use_case", e.use_case},
            {"capabilities", e.capabilities},
            {"hf_downloads", e.hf_downloads},
            {"hf_likes", e.hf_likes},
            {"demoted", e.demoted},
            {"demote_reason", e.demote_reason}};
  if (with_config) {
    j["config"] = e.config;
    j.erase("demoted");
    j.erase("demote_reason");
  }
  return j;
}

Result<Catalog> load_catalog(const fs::path& p) {
  Catalog c;
  if (!fs::exists(p)) return c;  // an empty catalog is valid: `mqde catalog update` fills it
  std::ifstream in(p);
  Json j;
  try {
    in >> j;
  } catch (const std::exception& ex) {
    return make_error(Errc::kParse, "catalog {}: {}", p.string(), ex.what());
  }
  c.updated_at = j.value("updated_at", "");
  for (const auto& r : j.value("entries", Json::array())) {
    CatalogEntry e;
    e.name = r.value("name", "");
    e.provider = r.value("provider", "");
    e.parameter_count = r.value("parameter_count", "");
    e.parameters_raw = r.value("parameters_raw", int64_t{0});
    e.quantization = r.value("quantization", "");
    e.format = r.value("format", "");
    e.context_length = r.value("context_length", int64_t{0});
    e.architecture = r.value("architecture", "");
    e.pipeline_tag = r.value("pipeline_tag", "");
    e.license = r.value("license", "");
    e.release_date = r.value("release_date", "");
    e.use_case = r.value("use_case", "");
    e.capabilities = r.value("capabilities", std::vector<std::string>{});
    e.hf_downloads = r.value("hf_downloads", int64_t{0});
    e.hf_likes = r.value("hf_likes", int64_t{0});
    e.config = r.value("config", Json::object());
    apply_hygiene(e);
    c.entries.push_back(std::move(e));
  }
  return c;
}

Result<void> save_catalog(const Catalog& c, const fs::path& p) {
  std::error_code ec;
  fs::create_directories(p.parent_path(), ec);
  Json arr = Json::array();
  for (const auto& e : c.entries) arr.push_back(to_json(e, true));
  const fs::path tmp = p.string() + ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) return make_error(Errc::kIo, "cannot write {}", tmp.string());
    out << Json{{"updated_at", c.updated_at}, {"source", "huggingface.co/api/models"}, {"entries", arr}}.dump(1) << "\n";
  }
  fs::rename(tmp, p, ec);
  if (ec) return make_error(Errc::kIo, "rename {}: {}", p.string(), ec.message());
  return {};
}

std::vector<const CatalogEntry*> search(const Catalog& c, const std::string& query, const std::string& provider,
                                        const std::string& capability, size_t limit, bool include_demoted) {
  const std::string q = lower(query), pv = lower(provider), cap = lower(capability);
  std::vector<const CatalogEntry*> out;
  for (const auto& e : c.entries) {
    if (e.demoted && !include_demoted) continue;
    if (!q.empty() && lower(e.name).find(q) == std::string::npos) continue;
    if (!pv.empty() && lower(e.provider) != pv) continue;
    if (!cap.empty()) {
      bool has_cap = false;
      for (const auto& c : e.capabilities)
        if (lower(c) == cap) { has_cap = true; break; }
      if (!has_cap) continue;
    }
    out.push_back(&e);
  }
  std::stable_sort(out.begin(), out.end(), [](auto* a, auto* b) { return a->hf_downloads > b->hf_downloads; });
  if (limit > 0 && out.size() > limit) out.resize(limit);
  return out;
}

const CatalogEntry* find(const Catalog& c, const std::string& name) {
  const std::string n = lower(name.starts_with("hf:") ? name.substr(3) : name);
  for (const auto& e : c.entries)
    if (lower(e.name) == n) return &e;
  return nullptr;
}

Result<modelspec::ModelSpec> to_spec(const CatalogEntry& e) {
  if (e.parameters_raw <= 0) return make_error(Errc::kInvalidArgument, "catalog entry {} has no parameter count", e.name);
  Json cfg = e.config;
  // GPT-2 style names -> the HF-standard keys model_spec.cpp reads.
  for (const auto& [from, to] : std::initializer_list<std::pair<const char*, const char*>>{
           {"n_layer", "num_hidden_layers"}, {"n_embd", "hidden_size"}, {"d_model", "hidden_size"},
           {"n_head", "num_attention_heads"}, {"n_inner", "intermediate_size"}, {"n_positions", "max_position_embeddings"},
           {"n_ctx", "max_position_embeddings"}})
    if (cfg.contains(from) && !cfg.contains(to) && !cfg[from].is_null()) cfg[to] = cfg[from];
  if (!cfg.contains("intermediate_size") && cfg.contains("hidden_size"))
    cfg["intermediate_size"] = 4 * cfg["hidden_size"].get<int64_t>();
  const std::string mt = cfg.value("model_type", "");
  MQDE_ASSIGN_OR_RETURN(auto fam, modelspec::find_model_family(mt));
  // Synthesize tensors whose totals match the Hub's safetensors summary. MoE models get a routed-expert
  // tensor sized from the config so active-parameter math works; the rest is one dense tensor.
  const std::string dtype = e.quantization == "FP8" ? "F8_E4M3" : e.quantization == "F32" ? "F32"
                            : e.quantization == "F16" ? "F16" : "BF16";
  const double bpp = e.quantization == "FP8" ? 1.0 : e.quantization == "F32" ? 4.0
                     : e.quantization.starts_with("INT") ? std::stoi(e.quantization.substr(3)) / 8.0 : 2.0;
  const int64_t hidden = std::max<int64_t>(1, cfg_int(cfg, {"hidden_size", "n_embd", "d_model"}));
  const int64_t layers = cfg_int(cfg, {"num_hidden_layers", "n_layer", "num_layers"});
  const int64_t n_exp = cfg_int(cfg, {"num_experts", "n_routed_experts", "num_local_experts"});
  const int64_t moe_inter = cfg_int(cfg, {"moe_intermediate_size", "intermediate_size"});
  int64_t expert = n_exp > 0 ? 3 * n_exp * hidden * moe_inter * layers : 0;
  if (expert >= e.parameters_raw) expert = static_cast<int64_t>(0.9 * static_cast<double>(e.parameters_raw));
  const int64_t dense = e.parameters_raw - expert;
  std::vector<modelspec::TensorInfo> t;
  auto add = [&](std::string name, int64_t numel) {
    if (numel <= 0) return;
    const int64_t cols = std::max<int64_t>(1, numel / hidden);
    t.push_back({std::move(name), dtype, {hidden, cols}, hidden * cols,
                 static_cast<uint64_t>(static_cast<double>(hidden * cols) * bpp), "catalog", 0});
  };
  add("model.layers.0.mlp.up_proj.weight", dense);
  add("model.layers.0.mlp.experts.0.up_proj.weight", expert);
  MQDE_ASSIGN_OR_RETURN(auto spec, modelspec::build_model_spec(cfg, std::move(t), fam));
  spec.path = "hf:" + e.name;
  spec.warnings.push_back("catalog entry: tensors synthesized from Hub metadata, no local weights");
  return spec;
}

Result<Catalog> update_catalog(const HfOptions& o, const UpdateOptions& u, Catalog base,
                               const std::function<void(size_t, size_t, const std::string&)>& progress) {
  const std::string expand =
      "&expand[]=safetensors&expand[]=cardData&expand[]=downloads&expand[]=likes&expand[]=pipeline_tag"
      "&expand[]=createdAt&expand[]=author";
  std::vector<std::string> sorts;
  if (u.downloads) sorts.push_back("downloads");
  if (u.trending) sorts.push_back("trendingScore");
  if (sorts.empty()) return make_error(Errc::kInvalidArgument, "catalog update: enable --trending and/or --downloads");
  // Over-fetch: gated repos and repos without config.json are skipped, so the list must be longer than limit.
  const size_t per = std::max<size_t>(20, u.limit * 2 / sorts.size());
  std::vector<std::vector<Json>> lists;
  for (const auto& s : sorts)
    for (const char* tag : {"text-generation", "image-text-to-text"}) {
      MQDE_ASSIGN_OR_RETURN(Json l, list_models(o, fmt::format("sort={}&direction=-1&limit={}&pipeline_tag={}{}", s,
                                                               per, tag, expand)));
      lists.emplace_back(l.begin(), l.end());
    }
  // Round-robin merge so downloads and trending, text and vision, all get represented.
  std::vector<Json> cand;
  std::set<std::string> seen;
  for (size_t i = 0;; ++i) {
    bool any = false;
    for (const auto& l : lists)
      if (i < l.size()) {
        any = true;
        const std::string id = l[i].value("id", "");
        if (!id.empty() && seen.insert(id).second) cand.push_back(l[i]);
      }
    if (!any) break;
  }
  std::vector<CatalogEntry> fresh;
  size_t tried = 0;
  for (const auto& info : cand) {
    if (fresh.size() >= u.limit) break;
    const std::string id = info.value("id", "");
    if (tried++ > 0) std::this_thread::sleep_for(std::chrono::milliseconds(u.interval_ms));
    if (progress) progress(fresh.size(), u.limit, id);
    auto cfg = fetch_json_file(o, id, "config.json");
    if (!cfg) continue;  // gated / no config.json: required fields would be missing
    auto e = entry_from_hub(info, *cfg);
    if (e.parameters_raw <= 0 || e.context_length <= 0) continue;  // required fields must be complete
    fresh.push_back(std::move(e));
  }
  // Merge: fresh entries replace same-named old ones; others are kept.
  std::set<std::string> fresh_names;
  for (const auto& e : fresh) fresh_names.insert(lower(e.name));
  Catalog out;
  out.entries = std::move(fresh);
  for (auto& e : base.entries)
    if (!fresh_names.contains(lower(e.name))) out.entries.push_back(std::move(e));
  out.updated_at = now_iso();
  return out;
}

}  // namespace mqde::hub
