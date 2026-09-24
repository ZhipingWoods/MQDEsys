#include "mqde/modelspec/model_spec.h"

#include <algorithm>
#include <map>
#include <set>

namespace mqde::modelspec {
namespace fs = std::filesystem;
namespace {

const Json* walk(const Json& root, const std::vector<std::string>& path) {
  const Json* cur = &root;
  for (const auto& k : path) {
    auto it = cur->find(k);
    if (it == cur->end() || !it->is_object()) return nullptr;
    cur = &*it;
  }
  return cur;
}

TextSpec parse_text(const Json& t) {
  TextSpec s;
  s.n_layers = get_or<int64_t>(t, "num_hidden_layers", 0);
  s.hidden = get_or<int64_t>(t, "hidden_size", 0);
  s.n_heads = get_or<int64_t>(t, "num_attention_heads", 0);
  s.n_kv_heads = get_or<int64_t>(t, "num_key_value_heads", s.n_heads);
  s.head_dim = get_or<int64_t>(t, "head_dim", s.n_heads > 0 ? s.hidden / s.n_heads : 0);
  s.intermediate = get_or<int64_t>(t, "intermediate_size", 0);
  s.vocab = get_or<int64_t>(t, "vocab_size", 0);
  s.max_position_embeddings = get_or<int64_t>(t, "max_position_embeddings", 0);
  s.sliding_window = get_or<int64_t>(t, "sliding_window", 0);
  s.tie_word_embeddings = get_or<bool>(t, "tie_word_embeddings", false);
  s.layer_types = get_or<std::vector<std::string>>(t, "layer_types", {});
  if (s.layer_types.empty()) s.layer_types.assign(static_cast<size_t>(s.n_layers), "full_attention");
  for (const auto& lt : s.layer_types) {
    if (lt == "linear_attention") ++s.n_linear_attention;
    else if (lt == "sliding_attention") ++s.n_sliding_attention;
    else ++s.n_full_attention;
  }
  s.linear_num_key_heads = get_or<int64_t>(t, "linear_num_key_heads", 0);
  s.linear_num_value_heads = get_or<int64_t>(t, "linear_num_value_heads", 0);
  s.linear_key_head_dim = get_or<int64_t>(t, "linear_key_head_dim", 0);
  s.linear_value_head_dim = get_or<int64_t>(t, "linear_value_head_dim", 0);
  s.linear_conv_kernel = get_or<int64_t>(t, "linear_conv_kernel_dim", 0);
  s.linear_state_dtype = get_or<std::string>(t, "mamba_ssm_dtype", "float32");
  return s;
}

}  // namespace

Result<ModelSpec> build_model_spec(const Json& config, std::vector<TensorInfo> tensors, const ModelFamily& family) {
  ModelSpec spec;
  spec.model_type = get_or<std::string>(config, "model_type", "unknown");
  spec.architectures = get_or<std::vector<std::string>>(config, "architectures", {});
  spec.family = family.family;
  spec.modality = family.modality;

  const Json* text = family.text_config_path.empty() ? nullptr : walk(config, family.text_config_path);
  if (!text && config.contains("text_config")) text = &config["text_config"];
  if (!text) text = &config;
  spec.text = parse_text(*text);
  if (spec.text.n_layers == 0) spec.warnings.push_back("could not determine language model layer count");

  if (const Json* v = family.vision_config_path.empty() ? nullptr : walk(config, family.vision_config_path)) {
    VisionSpec vs;
    vs.n_layers = get_or<int64_t>(*v, "num_hidden_layers", 0);
    vs.hidden = get_or<int64_t>(*v, "hidden_size", 0);
    vs.intermediate = get_or<int64_t>(*v, "intermediate_size", 0);
    vs.patch_size = get_or<int64_t>(*v, "patch_size", 0);
    vs.image_size = get_or<int64_t>(*v, "image_size", 0);
    vs.tokens_per_image = family.tokens_per_image;
    spec.vision = vs;
  }
  if (!family.action_horizon_key.empty() && config.contains(family.action_horizon_key)) {
    ActionSpec as;
    as.horizon = get_or<int64_t>(config, family.action_horizon_key, 0);
    as.inference_steps = get_or<int64_t>(config, family.action_steps_key, 1);
    as.action_dim = get_or<int64_t>(config, "action_dim", 0);
    as.query_tokens = family.action_query_tokens;
    spec.action = as;
  }

  std::vector<std::regex> keep_fp, block_res;
  for (const auto& p : family.default_keep_fp) keep_fp.emplace_back(p);
  for (const auto& c : family.components) block_res.emplace_back(c.block_regex.empty() ? "$^" : c.block_regex);
  spec.keep_fp_patterns = family.default_keep_fp;

  std::map<std::string, ComponentStat> stats;
  std::map<std::string, std::set<int64_t>> blocks;
  std::vector<std::string> order;
  int64_t unassigned = 0;
  for (const auto& t : tensors) {
    UnitInfo u;
    u.tensor = t.name;
    for (size_t i = 0; i < family.components.size(); ++i) {
      const auto& c = family.components[i];
      if (!t.name.starts_with(c.prefix)) continue;
      u.component = c.name;
      u.modality = c.modality;
      std::smatch m;
      std::string rest = t.name.substr(c.prefix.size());
      if (!c.block_regex.empty() && std::regex_search(rest, m, block_res[i]) && m.size() > 1)
        u.block = std::stoll(m[1].str());
      break;
    }
    if (u.component.empty()) {
      u.component = "other";
      u.modality = "other";
      ++unassigned;
    }
    const bool is_matrix = t.shape.size() == 2 && t.name.ends_with(".weight");
    const bool kept = std::any_of(keep_fp.begin(), keep_fp.end(),
                                  [&](const std::regex& r) { return std::regex_search(t.name, r); });
    u.quantizable = is_matrix && !kept;

    auto [it, inserted] = stats.try_emplace(u.component);
    if (inserted) order.push_back(u.component);
    auto& s = it->second;
    s.name = u.component;
    s.modality = u.modality;
    s.n_tensors += 1;
    s.params += t.numel;
    s.bytes += t.nbytes;
    if (u.quantizable) {
      s.n_quant_units += 1;
      s.quant_params += t.numel;
    }
    if (u.block >= 0) blocks[u.component].insert(u.block);
    spec.total_params += t.numel;
    spec.total_bytes += t.nbytes;
    spec.units.push_back(std::move(u));
  }
  for (const auto& name : order) {
    auto s = stats[name];
    s.n_blocks = static_cast<int64_t>(blocks[name].size());
    spec.components.push_back(s);
  }
  if (unassigned > 0)
    spec.warnings.push_back(fmt::format("{} tensors did not match any component of family '{}'", unassigned, family.family));
  spec.tensors = std::move(tensors);
  return spec;
}

Result<ModelSpec> inspect_local(const fs::path& model_dir) {
  const auto cfg_path = model_dir / "config.json";
  auto cfg = load_structured_file(cfg_path);
  if (!cfg) return std::unexpected(cfg.error().with_context("reading model config"));
  auto tensors = read_model_tensors(model_dir);
  if (!tensors) return std::unexpected(tensors.error().with_context("reading safetensors headers"));
  const auto model_type = get_or<std::string>(*cfg, "model_type", "");
  MQDE_ASSIGN_OR_RETURN(ModelFamily fam, find_model_family(model_type));
  auto spec = build_model_spec(*cfg, std::move(*tensors), fam);
  if (spec) spec->path = fs::absolute(model_dir).string();
  return spec;
}

Json to_json(const ModelSpec& s, bool include_tensors) {
  Json j;
  j["apiVersion"] = "mqde/v1";
  j["path"] = s.path;
  j["model_type"] = s.model_type;
  j["family"] = s.family;
  j["modality"] = s.modality;
  j["architectures"] = s.architectures;
  j["total_params"] = s.total_params;
  j["total_bytes"] = s.total_bytes;
  const auto& t = s.text;
  j["text"] = {{"n_layers", t.n_layers},
               {"hidden", t.hidden},
               {"n_heads", t.n_heads},
               {"n_kv_heads", t.n_kv_heads},
               {"head_dim", t.head_dim},
               {"intermediate", t.intermediate},
               {"vocab", t.vocab},
               {"max_position_embeddings", t.max_position_embeddings},
               {"n_full_attention", t.n_full_attention},
               {"n_linear_attention", t.n_linear_attention},
               {"n_sliding_attention", t.n_sliding_attention},
               {"sliding_window", t.sliding_window},
               {"tie_word_embeddings", t.tie_word_embeddings}};
  if (s.vision)
    j["vision"] = {{"n_layers", s.vision->n_layers},
                   {"hidden", s.vision->hidden},
                   {"patch_size", s.vision->patch_size},
                   {"image_size", s.vision->image_size},
                   {"tokens_per_image", s.vision->tokens_per_image}};
  if (s.action)
    j["action"] = {{"horizon", s.action->horizon},
                   {"action_dim", s.action->action_dim},
                   {"inference_steps", s.action->inference_steps},
                   {"query_tokens", s.action->query_tokens}};
  j["components"] = Json::array();
  for (const auto& c : s.components)
    j["components"].push_back({{"name", c.name},
                               {"modality", c.modality},
                               {"n_tensors", c.n_tensors},
                               {"params", c.params},
                               {"bytes", c.bytes},
                               {"n_blocks", c.n_blocks},
                               {"n_quant_units", c.n_quant_units},
                               {"quant_params", c.quant_params}});
  j["keep_fp_patterns"] = s.keep_fp_patterns;
  j["warnings"] = s.warnings;
  if (include_tensors) {
    j["tensors"] = Json::array();
    for (size_t i = 0; i < s.tensors.size(); ++i) {
      const auto& ti = s.tensors[i];
      const auto& u = s.units[i];
      j["tensors"].push_back({{"name", ti.name},
                              {"dtype", ti.dtype},
                              {"shape", ti.shape},
                              {"component", u.component},
                              {"block", u.block},
                              {"quantizable", u.quantizable}});
    }
  }
  return j;
}

}  // namespace mqde::modelspec
