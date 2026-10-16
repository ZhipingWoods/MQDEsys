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

namespace {

// A real F32 payload (not header-only like make_fixture_model() above): value[row][col] = row*1000 + col,
// so a correct whole-row read is easy to tell apart from an element-scattered one by eye.
fs::path write_numbered_matrix(const std::string& stem, int64_t rows, int64_t cols) {
  const auto dir = fs::temp_directory_path() / ("mqde_rows_" + stem);
  fs::create_directories(dir);
  const auto file = dir / "model.safetensors";
  std::vector<float> data(static_cast<size_t>(rows * cols));
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) data[static_cast<size_t>(r * cols + c)] = static_cast<float>(r * 1000 + c);
  mqde::Json header = {{"w", {{"dtype", "F32"}, {"shape", {rows, cols}}, {"data_offsets", {0, data.size() * 4}}}}};
  const std::string h = header.dump();
  std::ofstream out(file, std::ios::binary);
  uint64_t n = h.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xff));
  out << h;
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 4));
  return dir;
}

TensorInfo w_tensor(const fs::path& dir, int64_t rows, int64_t cols) {
  // data_offset is 8 (length prefix) + header JSON length + the tensor's own data_offsets[0] -- computing it
  // by hand here would just be re-deriving read_safetensors_header's own logic and risks getting it wrong
  // (as a first draft of this fixture did); reading it back for real is both simpler and the real contract.
  auto hdr = read_safetensors_header(dir / "model.safetensors");
  if (!hdr || hdr->empty()) throw std::runtime_error("fixture header unreadable");
  TensorInfo t = hdr->front();
  if (t.shape != std::vector<int64_t>{rows, cols}) throw std::runtime_error("fixture shape mismatch");
  return t;
}

}  // namespace

// T-C: quant_preview's actual bug. read_tensor_f32's element-level stride sampling is correct for histograms
// (order doesn't matter there) but wrong input for fake_quant, which needs a real row-major submatrix (its
// per-group granularity groups along columns). read_tensor_rows_f32 samples whole rows instead, so what it
// returns is always a valid (if possibly smaller) matrix with the tensor's real `cols`.
TEST(modelspec_safetensors, read_tensor_rows_f32_returns_a_structurally_valid_submatrix) {
  const auto dir = write_numbered_matrix("big", 100, 8);
  const auto t = w_tensor(dir, 100, 8);

  // cap large enough for the whole tensor: every row, bit-for-bit the same as a direct full read would give
  auto all = read_tensor_rows_f32(dir, t, 100 * 8);
  ASSERT_TRUE(all) << all.error().to_string();
  EXPECT_EQ(all->second, 100) << "rows_sampled should equal rows_total when the cap covers everything";
  ASSERT_EQ(all->first.size(), static_cast<size_t>(100 * 8));
  for (int64_t r = 0; r < 100; ++r)
    for (int64_t c = 0; c < 8; ++c)
      EXPECT_FLOAT_EQ(all->first[static_cast<size_t>(r * 8 + c)], static_cast<float>(r * 1000 + c))
          << "row " << r << " col " << c;

  // cap forcing a subset: rows_sampled must be smaller, but still a whole number of complete 8-column rows
  // (never a fractional row -- that would misalign fake_quant's column-grouped granularity)
  auto some = read_tensor_rows_f32(dir, t, 30);  // room for well under 100 rows of 8 columns each
  ASSERT_TRUE(some) << some.error().to_string();
  EXPECT_LT(some->second, 100);
  EXPECT_GT(some->second, 0);
  ASSERT_EQ(some->first.size(), static_cast<size_t>(some->second * 8));
  // every row actually returned must be one of the real rows of the matrix: its 8 values must be
  // {k*1000, k*1000+1, ..., k*1000+7} for some row index k -- not an element-scattered mixture of rows
  int64_t max_row_seen = -1;
  for (int64_t r = 0; r < some->second; ++r) {
    const float first_val = some->first[static_cast<size_t>(r * 8)];
    const int64_t k = static_cast<int64_t>(first_val) / 1000;
    max_row_seen = std::max(max_row_seen, k);
    for (int64_t c = 0; c < 8; ++c)
      EXPECT_FLOAT_EQ(some->first[static_cast<size_t>(r * 8 + c)], static_cast<float>(k * 1000 + c))
          << "returned row " << r << " is not a real, complete row of the source matrix";
  }
  // the sample must actually be spread across the matrix, not just its first few rows -- a 100-row matrix
  // sampled down to 3 rows should include something well past the start
  EXPECT_GT(max_row_seen, 50) << "rows should be spread across the whole height, not clustered at the start";
}

TEST(modelspec_safetensors, read_tensor_rows_f32_rejects_non_2d_tensors) {
  const auto dir = write_numbered_matrix("flat", 1, 8);  // a [1, 8] matrix...
  TensorInfo t = w_tensor(dir, 1, 8);
  t.shape = {8};  // ...reported here as the 1-D tensor it would be if squeezed, which is what must be rejected
  auto r = read_tensor_rows_f32(dir, t, 1000);
  EXPECT_FALSE(r);
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
