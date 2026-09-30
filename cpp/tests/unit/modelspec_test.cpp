#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "mqde/modelspec/model_spec.h"

namespace fs = std::filesystem;
using namespace mqde::modelspec;

namespace {

// Writes a header-only safetensors file (the payload is never read by the inspector).
void write_safetensors(const fs::path& p, const mqde::Json& header) {
  const std::string h = header.dump();
  std::ofstream out(p, std::ios::binary);
  uint64_t n = h.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xff));
  out << h;
}

mqde::Json entry(const std::string& dtype, std::vector<int64_t> shape, uint64_t& off) {
  uint64_t n = 1;
  for (auto d : shape) n *= static_cast<uint64_t>(d);
  const uint64_t bytes = n * (dtype == "F32" ? 4 : 2);
  mqde::Json e = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {off, off + bytes}}};
  off += bytes;
  return e;
}

fs::path make_fixture_model() {
  const auto dir = fs::temp_directory_path() / "mqde_tiny_vla";
  fs::create_directories(dir);
  fs::copy_file(fs::path(MQDE_TEST_FIXTURES) / "minicpm_vla" / "config.json", dir / "config.json",
                fs::copy_options::overwrite_existing);
  uint64_t off = 0;
  mqde::Json h;
  h["__metadata__"] = {{"format", "pt"}};
  h["vlm.model.language_model.embed_tokens.weight"] = entry("BF16", {248094, 1024}, off);
  h["vlm.model.language_model.layers.0.mlp.up_proj.weight"] = entry("BF16", {3584, 1024}, off);
  h["vlm.model.language_model.layers.3.self_attn.k_proj.weight"] = entry("BF16", {512, 1024}, off);
  h["vlm.model.language_model.layers.0.linear_attn.in_proj_a.weight"] = entry("BF16", {16, 1024}, off);
  h["vlm.model.language_model.norm.weight"] = entry("BF16", {1024}, off);
  h["vlm.model.vision_tower.encoder.layers.26.mlp.fc1.weight"] = entry("BF16", {4304, 1152}, off);
  h["vlm.model.merger.mlp.0.linear_1.weight"] = entry("BF16", {4608, 4608}, off);
  h["action_head.model.transformer_blocks.15.ff.net.2.weight"] = entry("F32", {768, 3072}, off);
  h["action_head.action_decoder.layer1.W"] = entry("F32", {32, 1024, 1024}, off);
  write_safetensors(dir / "model.safetensors", h);
  return dir;
}

const UnitInfo& unit(const ModelSpec& s, const std::string& name) {
  for (const auto& u : s.units)
    if (u.tensor == name) return u;
  throw std::runtime_error("unit not found: " + name);
}

}  // namespace

TEST(modelspec_safetensors, header_roundtrip) {
  const auto dir = make_fixture_model();
  auto t = read_model_tensors(dir);
  ASSERT_TRUE(t) << t.error().to_string();
  ASSERT_EQ(t->size(), 9u);
  for (const auto& ti : *t)
    EXPECT_EQ(ti.nbytes, static_cast<uint64_t>(ti.numel * dtype_bytes(ti.dtype))) << ti.name;
}

TEST(modelspec_safetensors, rejects_garbage) {
  const auto p = fs::temp_directory_path() / "mqde_bad.safetensors";
  std::ofstream(p, std::ios::binary) << "not a safetensors file at all";
  auto t = read_safetensors_header(p);
  ASSERT_FALSE(t);
  EXPECT_EQ(t.error().code, mqde::Errc::kParse);
}

TEST(modelspec_vla, minicpm_vla_structure) {
  auto spec = inspect_local(make_fixture_model());
  ASSERT_TRUE(spec) << spec.error().to_string();
  EXPECT_EQ(spec->family, "minicpm_vla");
  EXPECT_EQ(spec->modality, "vla");
  EXPECT_EQ(spec->text.n_layers, 24);
  EXPECT_EQ(spec->text.n_full_attention, 6);
  EXPECT_EQ(spec->text.n_linear_attention, 18);
  EXPECT_EQ(spec->text.n_kv_heads, 2);
  EXPECT_EQ(spec->text.head_dim, 256);
  ASSERT_TRUE(spec->vision.has_value());
  EXPECT_EQ(spec->vision->n_layers, 27);
  EXPECT_EQ(spec->vision->tokens_per_image, 64);
  ASSERT_TRUE(spec->action.has_value());
  EXPECT_EQ(spec->action->horizon, 30);
  EXPECT_EQ(spec->action->inference_steps, 4);

  const auto& up = unit(*spec, "vlm.model.language_model.layers.0.mlp.up_proj.weight");
  EXPECT_EQ(up.component, "language");
  EXPECT_EQ(up.block, 0);
  EXPECT_TRUE(up.quantizable);
  EXPECT_FALSE(unit(*spec, "vlm.model.language_model.embed_tokens.weight").quantizable);
  EXPECT_FALSE(unit(*spec, "vlm.model.language_model.layers.0.linear_attn.in_proj_a.weight").quantizable);
  EXPECT_FALSE(unit(*spec, "vlm.model.language_model.norm.weight").quantizable) << "1-D";
  EXPECT_FALSE(unit(*spec, "action_head.action_decoder.layer1.W").quantizable) << "3-D category weight";
  const auto& fc1 = unit(*spec, "vlm.model.vision_tower.encoder.layers.26.mlp.fc1.weight");
  EXPECT_EQ(fc1.component, "vision");
  EXPECT_EQ(fc1.block, 26);
  EXPECT_EQ(unit(*spec, "vlm.model.merger.mlp.0.linear_1.weight").component, "merger");
  const auto& dit = unit(*spec, "action_head.model.transformer_blocks.15.ff.net.2.weight");
  EXPECT_EQ(dit.modality, "action");
  EXPECT_EQ(dit.block, 15);
  EXPECT_TRUE(spec->warnings.empty());
}
