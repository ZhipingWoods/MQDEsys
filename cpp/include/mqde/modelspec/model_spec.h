#pragma once
// ModelSpec: everything the estimator / planner / UI need to know about a model, derived from
// config.json + safetensors headers + a data-driven architecture template (data/model_families).

#include <cstdint>
#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "mqde/common/config.h"
#include "mqde/common/result.h"
#include "mqde/modelspec/safetensors.h"

namespace mqde::modelspec {

struct ComponentRule {
  std::string name;
  std::string modality;  // vision | language | action | ...
  std::string prefix;
  std::string block_regex;  // optional, first capture group = block index
};

struct ModelFamily {
  std::string family;
  std::string modality;  // llm | vlm | vla | world
  std::vector<std::string> match_model_types;
  std::vector<std::string> text_config_path;
  std::vector<std::string> vision_config_path;
  int64_t tokens_per_image = 0;
  std::string action_horizon_key;
  std::string action_steps_key;
  int64_t action_query_tokens = 0;
  std::vector<ComponentRule> components;
  std::vector<std::string> default_keep_fp;
};

Result<ModelFamily> load_model_family(const std::filesystem::path& file);
// Picks the family whose match.model_type contains `model_type`, else generic_decoder.
Result<ModelFamily> find_model_family(std::string_view model_type);

// Language-model shape relevant to KV cache / recurrent state sizing.
struct TextSpec {
  int64_t n_layers = 0;
  int64_t hidden = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t head_dim = 0;
  int64_t intermediate = 0;
  int64_t vocab = 0;
  int64_t max_position_embeddings = 0;
  int64_t sliding_window = 0;  // 0 = none
  bool tie_word_embeddings = false;
  std::vector<std::string> layer_types;  // per layer: full_attention | linear_attention | sliding_attention
  int64_t n_full_attention = 0;
  int64_t n_linear_attention = 0;
  int64_t n_sliding_attention = 0;
  // Gated DeltaNet / linear attention recurrent state
  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel = 0;
  std::string linear_state_dtype = "float32";
  // Mixture-of-experts (0 = dense)
  int64_t n_experts = 0;
  int64_t experts_per_tok = 0;
};

struct VisionSpec {
  int64_t n_layers = 0;
  int64_t hidden = 0;
  int64_t intermediate = 0;
  int64_t patch_size = 0;
  int64_t image_size = 0;
  int64_t tokens_per_image = 0;  // tokens fed to the language model per image slice
};

struct ActionSpec {
  int64_t horizon = 0;
  int64_t action_dim = 0;
  int64_t inference_steps = 0;
  int64_t query_tokens = 0;
};

struct UnitInfo {  // one quantization candidate or kept-FP tensor, grouped for QIR / UI
  std::string tensor;
  std::string component;
  std::string modality;
  int64_t block = -1;
  bool quantizable = false;  // 2-D weight not matched by default_keep_fp
  // Unpacked view of the tensor. For pre-quantized checkpoints (U8/I32 packed int4, FP8) this is the
  // logical [out, in] shape and element count; `aux` marks scale/zero/g_idx tensors that disappear when
  // the model is re-quantized from its logical BF16 weights.
  std::vector<int64_t> logical_shape;
  int64_t logical_numel = 0;
  bool aux = false;
};

// Quantization already present in the checkpoint (config.json quantization_config).
struct CheckpointQuant {
  std::string method;  // fp8 | modelopt | gptq | awq | compressed-tensors | ...
  std::string algo;    // e.g. W4A16_AWQ, e4m3
  int bits = 16;
  int64_t group_size = 0;
  std::vector<std::string> ignored;  // modules_to_not_convert / ignore
};

struct ComponentStat {
  std::string name;
  std::string modality;
  int64_t n_tensors = 0;
  int64_t params = 0;
  uint64_t bytes = 0;
  int64_t n_blocks = 0;
  int64_t n_quant_units = 0;
  int64_t quant_params = 0;
};

struct ModelSpec {
  std::string path;
  std::string model_type;
  std::string family;
  std::string modality;
  std::vector<std::string> architectures;
  TextSpec text;
  std::optional<VisionSpec> vision;
  std::optional<ActionSpec> action;
  std::vector<TensorInfo> tensors;
  std::vector<UnitInfo> units;  // parallel to tensors
  std::vector<ComponentStat> components;
  int64_t total_params = 0;
  uint64_t total_bytes = 0;
  std::vector<std::string> keep_fp_patterns;
  std::optional<CheckpointQuant> checkpoint_quant;
  // Parameter count before packing: packed int weights (U8/I32) are counted as logical params.
  int64_t logical_params = 0;
  // Params touched per token: dense params + (experts_per_tok / n_experts) of routed-expert params.
  int64_t active_params = 0;
  int64_t expert_params = 0;
  std::vector<std::string> warnings;
};

// Inspects a local model directory (no weight payload is read).
Result<ModelSpec> inspect_local(const std::filesystem::path& model_dir);

// Builds a spec from already-parsed inputs (used by tests and remote inspection).
Result<ModelSpec> build_model_spec(const Json& config, std::vector<TensorInfo> tensors, const ModelFamily& family);

Json to_json(const ModelSpec& spec, bool include_tensors = false);

}  // namespace mqde::modelspec
