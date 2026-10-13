// GGUF header reader (T-R05): written against byte-exact files built here, and the same parser is checked against real
// llama.cpp output on the dev machine (see docs/TASKS.md).
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>

#include "gguf.h"

using namespace mqde;
using namespace mqde::modelspec;
namespace fs = std::filesystem;

namespace {

struct W {  // little-endian byte writer
  std::string b;
  template <class T> W& n(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); return *this; }
  W& s(const std::string& x) { n<uint64_t>(x.size()); b += x; return *this; }
  W& raw(const std::string& x) { b += x; return *this; }
};

enum VT : uint32_t { U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 };
enum TY : uint32_t { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q8_0 = 8, T_Q4_K = 12, T_BF16 = 30 };

struct T { std::string name; uint32_t type; std::vector<uint64_t> ne; };

// kv: already-encoded "key, type, value" blobs
std::string build(uint32_t version, const std::vector<std::string>& kv, const std::vector<T>& ts, uint64_t align = 32, bool write_data = true,
                  const std::vector<uint64_t>& nbytes = {}) {
  W h;
  h.raw("GGUF").n<uint32_t>(version).n<uint64_t>(ts.size()).n<uint64_t>(kv.size());
  for (const auto& e : kv) h.raw(e);
  uint64_t off = 0;
  for (size_t i = 0; i < ts.size(); ++i) {
    h.s(ts[i].name).n<uint32_t>(static_cast<uint32_t>(ts[i].ne.size()));
    for (auto d : ts[i].ne) h.n<uint64_t>(d);
    h.n<uint32_t>(ts[i].type).n<uint64_t>(off);
    off += (nbytes.empty() ? 0 : nbytes[i]) + 0;
  }
  while (h.b.size() % align) h.b.push_back('\0');
  if (write_data) h.b.append(off, '\x7f');
  return h.b;
}
std::string kv_str(const std::string& k, const std::string& v) { return W().s(k).n<uint32_t>(STR).s(v).b; }
template <class V> std::string kv_n(const std::string& k, uint32_t t, V v) { return W().s(k).n<uint32_t>(t).n<V>(v).b; }

fs::path write_file(const std::string& bytes, const char* tag) {
  const auto p = fs::temp_directory_path() / (std::string("mqde_gguf_") + tag + "_" + std::to_string(::getpid()) + ".gguf");
  std::ofstream(p, std::ios::binary) << bytes;
  return p;
}

// the five tensors of the sample model with their exact byte sizes
const std::vector<T> kTensors = {{"token_embd.weight", T_F16, {8, 4}}, {"blk.0.attn_q.weight", T_Q4_K, {256, 2}},
                                  {"blk.0.ffn_down.weight", T_Q8_0, {64, 3}}, {"blk.1.attn_norm.weight", T_F32, {8}},
                                  {"output_norm.weight", T_F32, {8}}};
const std::vector<uint64_t> kBytes = {64, 288, 204, 32, 32};

std::string sample(uint64_t align = 64, uint32_t version = 3) {
  return build(version, {kv_str("general.architecture", "llama"), kv_str("general.name", "tiny"), kv_n<uint32_t>("general.alignment", U32, static_cast<uint32_t>(align)),
                          kv_n<uint32_t>("llama.block_count", U32, 2)}, kTensors, align, true, kBytes);
}

}  // namespace

TEST(gguf_read, header_alignment_and_the_data_section_start) {
  const auto p = write_file(sample(64), "hdr");
  auto g = read_gguf(p);
  ASSERT_TRUE(g) << g.error().message;
  EXPECT_EQ(g->version, 3u);
  EXPECT_EQ(g->alignment, 64u);
  EXPECT_EQ(g->data_offset % 64, 0u);
  EXPECT_EQ(g->file_bytes, fs::file_size(p));
  EXPECT_EQ(g->tensors.size(), 5u);
  EXPECT_EQ(g->file_bytes - g->data_offset, 64u + 288 + 204 + 32 + 32) << "the payload is exactly the tensors";
  fs::remove(p);
}

TEST(gguf_read, alignment_defaults_to_32_and_version_2_is_accepted) {
  W w;
  const auto bytes = build(2, {kv_str("general.architecture", "llama")}, kTensors, 32, true, kBytes);
  const auto p = write_file(bytes, "v2");
  auto g = read_gguf(p);
  ASSERT_TRUE(g) << g.error().message;
  EXPECT_EQ(g->version, 2u);
  EXPECT_EQ(g->alignment, 32u);
  EXPECT_EQ(g->data_offset % 32, 0u);
  fs::remove(p);
}

TEST(gguf_read, tensors_get_row_major_shapes_exact_sizes_and_absolute_offsets) {
  const auto p = write_file(sample(), "ten");
  auto g = read_gguf(p);
  ASSERT_TRUE(g);
  const auto& t = g->tensors;
  EXPECT_EQ(t[0].name, "token_embd.weight");
  EXPECT_EQ(t[0].shape, (std::vector<int64_t>{4, 8})) << "GGUF stores ne = [fastest, ...]; torch order is the reverse";
  EXPECT_EQ(t[0].type, "F16");
  EXPECT_EQ(t[0].numel, 32);
  EXPECT_EQ(t[0].nbytes, 64u);
  EXPECT_EQ(t[1].type, "Q4_K");
  EXPECT_EQ(t[1].shape, (std::vector<int64_t>{2, 256}));
  EXPECT_EQ(t[1].nbytes, 2u * 144) << "256-element blocks of 144 bytes, one per row";
  EXPECT_EQ(t[2].nbytes, 3u * 2 * 34) << "Q8_0: 32-element blocks of 34 bytes";
  EXPECT_EQ(t[3].shape, (std::vector<int64_t>{8}));
  EXPECT_EQ(t[0].offset, g->data_offset);
  EXPECT_EQ(t[1].offset, g->data_offset + 64);
  EXPECT_EQ(t[4].offset, g->data_offset + 64 + 288 + 204 + 32);
  fs::remove(p);
}

TEST(gguf_read, every_block_size_matches_gguf_py) {
  struct C { uint32_t id; const char* name; int64_t blck; uint64_t bytes; };
  // GGML_QUANT_SIZES of gguf-py (QK_K = 256), id -> (elements per block, bytes per block)
  const std::vector<C> cases = {{0, "F32", 1, 4}, {1, "F16", 1, 2}, {2, "Q4_0", 32, 18}, {3, "Q4_1", 32, 20}, {6, "Q5_0", 32, 22}, {7, "Q5_1", 32, 24},
                                {8, "Q8_0", 32, 34}, {9, "Q8_1", 32, 36}, {10, "Q2_K", 256, 84}, {11, "Q3_K", 256, 110}, {12, "Q4_K", 256, 144},
                                {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210}, {15, "Q8_K", 256, 292}, {16, "IQ2_XXS", 256, 66}, {17, "IQ2_XS", 256, 74},
                                {18, "IQ3_XXS", 256, 98}, {19, "IQ1_S", 256, 50}, {20, "IQ4_NL", 32, 18}, {21, "IQ3_S", 256, 110}, {22, "IQ2_S", 256, 82},
                                {23, "IQ4_XS", 256, 136}, {24, "I8", 1, 1}, {25, "I16", 1, 2}, {26, "I32", 1, 4}, {27, "I64", 1, 8}, {28, "F64", 1, 8},
                                {29, "IQ1_M", 256, 56}, {30, "BF16", 1, 2}, {34, "TQ1_0", 256, 54}, {35, "TQ2_0", 256, 66}, {39, "MXFP4", 32, 17},
                                {40, "NVFP4", 64, 36}, {41, "Q1_0", 128, 18}, {42, "Q2_0", 64, 18}};
  for (const auto& c : cases) {
    const std::vector<T> ts = {{"w", c.id, {static_cast<uint64_t>(c.blck * 3), 2}}};  // 3 blocks per row, 2 rows
    const auto p = write_file(build(3, {kv_str("general.architecture", "x")}, ts, 32, true, {6 * c.bytes}), "bs");
    auto g = read_gguf(p);
    ASSERT_TRUE(g) << c.name << ": " << g.error().message;
    EXPECT_EQ(g->tensors[0].type, c.name);
    EXPECT_EQ(g->tensors[0].nbytes, 6 * c.bytes) << c.name;
    fs::remove(p);
  }
}

TEST(gguf_read, scalar_metadata_of_every_type_and_summarised_arrays) {
  std::vector<std::string> kv = {kv_str("general.architecture", "llama"), kv_n<uint8_t>("m.u8", U8, 200), kv_n<int8_t>("m.i8", I8, -5),
                                  kv_n<uint16_t>("m.u16", U16, 65000), kv_n<int16_t>("m.i16", I16, -300), kv_n<uint32_t>("m.u32", U32, 4000000000u),
                                  kv_n<int32_t>("m.i32", I32, -7), kv_n<float>("m.f32", F32, 0.5f), kv_n<uint8_t>("m.bool", BOOL, 1),
                                  kv_n<uint64_t>("m.u64", U64, 1ull << 40), kv_n<int64_t>("m.i64", I64, -(1ll << 40)), kv_n<double>("m.f64", F64, 2.25)};
  W sa;  // string array of 3
  sa.s("tokenizer.ggml.tokens").n<uint32_t>(ARR).n<uint32_t>(STR).n<uint64_t>(3).s("<s>").s("hello").s("</s>");
  W fa;  // float array of 20
  fa.s("tokenizer.ggml.scores").n<uint32_t>(ARR).n<uint32_t>(F32).n<uint64_t>(20);
  for (int i = 0; i < 20; ++i) fa.n<float>(static_cast<float>(i));
  kv.push_back(sa.b);
  kv.push_back(fa.b);
  kv.push_back(kv_n<uint32_t>("after", U32, 99));  // a key after the arrays proves they were skipped correctly
  const auto p = write_file(build(3, kv, kTensors, 32, true, kBytes), "kv");
  auto g = read_gguf(p);
  ASSERT_TRUE(g) << g.error().message;
  const auto& m = g->metadata;
  EXPECT_EQ(m["m.u8"], 200);
  EXPECT_EQ(m["m.i8"], -5);
  EXPECT_EQ(m["m.u16"], 65000);
  EXPECT_EQ(m["m.i16"], -300);
  EXPECT_EQ(m["m.u32"], 4000000000u);
  EXPECT_EQ(m["m.i32"], -7);
  EXPECT_DOUBLE_EQ(m["m.f32"].get<double>(), 0.5);
  EXPECT_EQ(m["m.bool"], true);
  EXPECT_EQ(m["m.u64"].get<uint64_t>(), 1ull << 40);
  EXPECT_EQ(m["m.i64"].get<int64_t>(), -(1ll << 40));
  EXPECT_DOUBLE_EQ(m["m.f64"].get<double>(), 2.25);
  EXPECT_EQ(m["tokenizer.ggml.tokens"]["array_of"], "string");
  EXPECT_EQ(m["tokenizer.ggml.tokens"]["length"], 3);
  EXPECT_EQ(m["tokenizer.ggml.tokens"]["head"], (Json::array({"<s>", "hello", "</s>"})));
  EXPECT_EQ(m["tokenizer.ggml.scores"]["array_of"], "float32");
  EXPECT_EQ(m["tokenizer.ggml.scores"]["length"], 20);
  EXPECT_EQ(m["tokenizer.ggml.scores"]["head"].size(), 8u) << "only the first few are kept";
  EXPECT_EQ(m["after"], 99) << "the reader is positioned right after the arrays";
  fs::remove(p);
}

TEST(gguf_read, a_file_that_cannot_be_a_valid_gguf_is_an_error_with_a_reason) {
  auto err = [&](const std::string& bytes, const char* tag) {
    const auto p = write_file(bytes, tag);
    auto g = read_gguf(p);
    fs::remove(p);
    EXPECT_FALSE(g) << tag;
    return g ? std::string() : g.error().message;
  };
  auto good = sample();
  EXPECT_NE(err("GGML" + good.substr(4), "magic").find("not a GGUF file"), std::string::npos);
  {  // version 1 (32-bit counts), 4 (unknown), and a big-endian file (version bytes swapped)
    auto v = good;
    uint32_t one = 1, four = 4, be3 = 0x03000000;
    std::memcpy(v.data() + 4, &one, 4);
    EXPECT_NE(err(v, "v1").find("version 1"), std::string::npos);
    std::memcpy(v.data() + 4, &four, 4);
    EXPECT_NE(err(v, "v4").find("version 4"), std::string::npos);
    std::memcpy(v.data() + 4, &be3, 4);
    EXPECT_NE(err(v, "be").find("big-endian"), std::string::npos);
  }
  EXPECT_NE(err(good.substr(0, 6), "short").find("truncated"), std::string::npos);
  EXPECT_NE(err(good.substr(0, 40), "cut_in_kv").find("truncated"), std::string::npos);
  {  // payload cut short: a half-downloaded file must not be passed off as complete
    const auto cut = good.substr(0, good.size() - 100);
    EXPECT_NE(err(cut, "cutdata").find("beyond the end of the file"), std::string::npos);
  }
  {  // absurd counts must not allocate
    auto v = good;
    uint64_t huge = 1ull << 50;
    std::memcpy(v.data() + 8, &huge, 8);
    EXPECT_NE(err(v, "huge_t").find("tensor count"), std::string::npos);
    v = good;
    std::memcpy(v.data() + 16, &huge, 8);
    EXPECT_NE(err(v, "huge_kv").find("metadata count"), std::string::npos);
  }
  {  // a row that is not a whole number of quantization blocks
    const auto bad = build(3, {kv_str("general.architecture", "x")}, {{"w", T_Q4_K, {100, 2}}}, 32, false);
    EXPECT_NE(err(bad, "rowblk").find("not a multiple of the block size"), std::string::npos);
  }
  {  // unknown tensor type id
    const auto bad = build(3, {kv_str("general.architecture", "x")}, {{"w", 99, {8}}}, 32, false);
    EXPECT_NE(err(bad, "type").find("unknown tensor type 99"), std::string::npos);
  }
  {  // too many dimensions
    const auto bad = build(3, {kv_str("general.architecture", "x")}, {{"w", T_F32, {1, 1, 1, 1, 1, 1, 1, 1, 1}}}, 32, false);
    EXPECT_NE(err(bad, "dims").find("dimensions"), std::string::npos);
  }
  {  // a string length that runs past the file
    W w;
    w.raw("GGUF").n<uint32_t>(3).n<uint64_t>(0).n<uint64_t>(1).n<uint64_t>(1ull << 40);
    EXPECT_NE(err(w.b, "strlen").find("truncated"), std::string::npos);
  }
  EXPECT_FALSE(read_gguf("/nonexistent/x.gguf"));
}

TEST(gguf_summary, groups_blocks_counts_types_and_names_the_model) {
  const auto p = write_file(sample(), "sum");
  auto g = read_gguf(p);
  ASSERT_TRUE(g);
  const auto s = gguf_summary(*g);
  EXPECT_EQ(s["architecture"], "llama");
  EXPECT_EQ(s["name"], "tiny");
  EXPECT_EQ(s["n_tensors"], 5);
  EXPECT_EQ(s["n_params"], 32 + 512 + 192 + 8 + 8);
  EXPECT_EQ(s["tensor_bytes"], 64 + 288 + 204 + 32 + 32);
  EXPECT_EQ(s["block_count"], 2) << "from <arch>.block_count";
  ASSERT_EQ(s["blocks"].size(), 2u);
  EXPECT_EQ(s["blocks"][0]["index"], 0);
  EXPECT_EQ(s["blocks"][0]["tensors"], 2);
  EXPECT_EQ(s["blocks"][0]["bytes"], 288 + 204);
  EXPECT_EQ(s["blocks"][0]["types"]["Q4_K"], 1);
  EXPECT_EQ(s["blocks"][0]["types"]["Q8_0"], 1);
  EXPECT_EQ(s["blocks"][1]["index"], 1);
  EXPECT_EQ(s["blocks"][1]["types"]["F32"], 1);
  EXPECT_EQ(s["global"], (Json::array({"token_embd.weight", "output_norm.weight"})));
  EXPECT_EQ(s["quant_types"]["Q4_K"]["tensors"], 1);
  EXPECT_EQ(s["quant_types"]["Q4_K"]["params"], 512);
  EXPECT_EQ(s["quant_types"]["Q4_K"]["bytes"], 288);
  EXPECT_EQ(s["quant_types"]["F32"]["tensors"], 2);
  EXPECT_EQ(s["tensors"].size(), 5u);
  EXPECT_EQ(s["tensors"][1]["name"], "blk.0.attn_q.weight");
  EXPECT_EQ(s["tensors"][1]["shape"], (Json::array({2, 256})));
  EXPECT_FALSE(gguf_summary(*g, false).contains("tensors"));
  fs::remove(p);
}

TEST(gguf_summary, blocks_are_ordered_numerically_not_by_text) {
  std::vector<T> ts;
  std::vector<uint64_t> nb;
  for (int i : {10, 2, 0, 1}) {
    ts.push_back({"blk." + std::to_string(i) + ".attn_norm.weight", T_F32, {8}});
    nb.push_back(32);
  }
  const auto p = write_file(build(3, {kv_str("general.architecture", "llama")}, ts, 32, true, nb), "ord");
  auto g = read_gguf(p);
  ASSERT_TRUE(g);
  const auto s = gguf_summary(*g);
  std::vector<int> idx;
  for (const auto& b : s["blocks"]) idx.push_back(b["index"]);
  EXPECT_EQ(idx, (std::vector<int>{0, 1, 2, 10}));
  EXPECT_FALSE(s.contains("block_count")) << "absent when the metadata does not say";
  fs::remove(p);
}

TEST(gguf_read, a_long_string_array_is_walked_to_its_end_not_just_its_head) {
  W sa;  // 12 strings: only 8 are kept, but all 12 must be stepped over
  sa.s("tokenizer.ggml.tokens").n<uint32_t>(ARR).n<uint32_t>(STR).n<uint64_t>(12);
  for (int i = 0; i < 12; ++i) sa.s("token-" + std::to_string(i) + std::string(50 * i, 'x'));
  const auto p = write_file(build(3, {kv_str("general.architecture", "llama"), sa.b, kv_n<uint32_t>("after", U32, 7)}, kTensors, 32, true, kBytes), "longarr");
  auto g = read_gguf(p);
  ASSERT_TRUE(g) << g.error().message;
  EXPECT_EQ(g->metadata["tokenizer.ggml.tokens"]["length"], 12);
  EXPECT_EQ(g->metadata["tokenizer.ggml.tokens"]["head"].size(), 8u);
  EXPECT_EQ(g->metadata["after"], 7) << "positioned right after the 12th string";
  EXPECT_EQ(g->tensors.size(), 5u);
  fs::remove(p);
}

TEST(gguf_summary, only_names_that_start_with_blk_are_text_blocks) {
  // an mmproj file keeps its vision tower under v.blk.N: that is not decoder block N
  const std::vector<T> ts = {{"blk.0.attn_q.weight", T_F32, {8}}, {"v.blk.0.attn_q.weight", T_F32, {8}}, {"mm.blk.3.weight", T_F32, {8}}};
  const auto p = write_file(build(3, {kv_str("general.architecture", "clip")}, ts, 32, true, {32, 32, 32}), "mmproj");
  auto g = read_gguf(p);
  ASSERT_TRUE(g);
  const auto s = gguf_summary(*g);
  ASSERT_EQ(s["blocks"].size(), 1u);
  EXPECT_EQ(s["blocks"][0]["tensors"], 1);
  EXPECT_EQ(s["global"], (Json::array({"v.blk.0.attn_q.weight", "mm.blk.3.weight"})));
  fs::remove(p);
}

TEST(gguf_read, an_alignment_that_is_not_a_power_of_two_is_refused) {
  for (uint32_t a : {0u, 48u, 100u}) {
    const auto p = write_file(build(3, {kv_str("general.architecture", "x"), kv_n<uint32_t>("general.alignment", U32, a)}, {}, 32, false), "align");
    auto g = read_gguf(p);
    EXPECT_FALSE(g) << a;
    if (!g) {
      EXPECT_NE(g.error().message.find("power of two"), std::string::npos) << a;
    }
    fs::remove(p);
  }
  const auto p = write_file(build(3, {kv_str("general.architecture", "x"), kv_n<uint32_t>("general.alignment", U32, 128)}, {}, 128, false), "align128");
  EXPECT_TRUE(read_gguf(p)) << "128 is fine";
  fs::remove(p);
}
