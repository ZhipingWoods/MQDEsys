#include <gtest/gtest.h>

#include <filesystem>

#include "mqde/hub/catalog.h"
#include "mqde/hub/hf.h"

using namespace mqde;
using namespace mqde::hub;

namespace {

// Shapes copied from real huggingface.co/api/models + config.json responses (Qwen/Qwen2.5-7B-Instruct).
Json qwen7_info() {
  return Json::parse(R"({"id":"Qwen/Qwen2.5-7B-Instruct","author":"Qwen","downloads":1000,"likes":50,
    "pipeline_tag":"text-generation","createdAt":"2024-09-16T11:55:40.000Z","cardData":{"license":"apache-2.0"},
    "safetensors":{"parameters":{"BF16":7615616512},"total":7615616512}})");
}
Json qwen7_config() {
  return Json::parse(R"({"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","hidden_size":3584,
    "intermediate_size":18944,"max_position_embeddings":32768,"num_attention_heads":28,"num_hidden_layers":28,
    "num_key_value_heads":4,"vocab_size":152064,"torch_dtype":"bfloat16","rope_theta":1000000.0})");
}

}  // namespace

TEST(hub, parse_size_units) {
  EXPECT_EQ(parse_size("800"), 800u);
  EXPECT_EQ(parse_size("1K"), 1024u);
  EXPECT_EQ(parse_size("12G"), 12ull << 30);
  EXPECT_EQ(parse_size("1.5GB"), static_cast<uint64_t>(1.5 * (1ull << 30)));
  EXPECT_FALSE(parse_size("twelve"));
}

TEST(hub, group_files_keeps_shards_together) {
  std::vector<RepoFile> f = {{"model-Q4_K_M-00001-of-00002.gguf", 5},
                             {"model-Q4_K_M-00002-of-00002.gguf", 5},
                             {"model-Q8_0.gguf", 18},
                             {"README.md", 1}};
  const auto g = group_files(f);
  ASSERT_EQ(g.size(), 2u);
  const auto q4 = std::find_if(g.begin(), g.end(), [](const auto& x) { return x.key == "Q4_K_M"; });
  ASSERT_NE(q4, g.end());
  EXPECT_EQ(q4->files.size(), 2u);
  EXPECT_EQ(q4->bytes, 10u);
}

TEST(hub, group_files_fp16_tag) {
  const auto g =
      group_files({{"qwen2.5-0.5b-instruct-fp16.gguf", 9}, {"qwen2.5-0.5b-instruct-q8_0.gguf", 5}});
  ASSERT_EQ(g.size(), 2u);
  EXPECT_EQ(g[0].key, "F16");
  EXPECT_EQ(g[1].key, "Q8_0");
}

TEST(hub, choose_group_budget_and_fallback) {
  std::vector<RepoFile> f = {{"m-Q4_K_M.gguf", 10}, {"m-Q8_0.gguf", 18}, {"m-Q2_K.gguf", 6}};
  const auto g = group_files(f);
  EXPECT_EQ(choose_group(g, 0)->key, "Q8_0");     // unlimited: best quality
  EXPECT_EQ(choose_group(g, 12)->key, "Q4_K_M");  // best within budget
  EXPECT_EQ(choose_group(g, 3)->key, "Q2_K");     // nothing fits: smallest
  EXPECT_EQ(choose_group(g, 0, "q4_k_m")->key, "Q4_K_M");
  EXPECT_FALSE(choose_group(g, 0, "Q5_K_M"));
}

TEST(hub, safetensors_group_includes_config) {
  std::vector<RepoFile> f = {{"model-00001-of-00002.safetensors", 4},
                             {"model-00002-of-00002.safetensors", 4},
                             {"config.json", 1},
                             {"tokenizer.json", 1},
                             {"README.md", 1}};
  const auto g = group_files(f);
  ASSERT_EQ(g.size(), 1u);
  EXPECT_EQ(g[0].key, "safetensors");
  EXPECT_EQ(g[0].files.size(), 4u);
}

TEST(hub, entry_from_hub_maps_required_fields) {
  const auto e = entry_from_hub(qwen7_info(), qwen7_config());
  EXPECT_EQ(e.name, "Qwen/Qwen2.5-7B-Instruct");
  EXPECT_EQ(e.provider, "Qwen");
  EXPECT_EQ(e.parameter_count, "7.6B");
  EXPECT_EQ(e.quantization, "BF16");
  EXPECT_EQ(e.format, "safetensors");
  EXPECT_EQ(e.context_length, 32768);
  EXPECT_EQ(e.architecture, "Qwen2ForCausalLM");
  EXPECT_EQ(e.license, "apache-2.0");
  EXPECT_EQ(e.release_date, "2024-09-16");
  EXPECT_FALSE(e.demoted) << e.demote_reason;
  EXPECT_FALSE(e.config.contains("rope_theta"));  // only estimator keys are kept
}

TEST(hub, hygiene_demotes_draft_mismatch_and_missing) {
  auto e = entry_from_hub(qwen7_info(), qwen7_config());
  e.name = "someone/Qwen2.5-7B-eagle3";
  apply_hygiene(e);
  EXPECT_TRUE(e.demoted);
  e = entry_from_hub(qwen7_info(), qwen7_config());
  e.name = "someone/Qwen2.5-72B-Instruct";  // name says 72B, weights are 7.6B
  apply_hygiene(e);
  EXPECT_TRUE(e.demoted);
  e = entry_from_hub(qwen7_info(), qwen7_config());
  e.context_length = 0;
  apply_hygiene(e);
  EXPECT_TRUE(e.demoted);
}

// ---- classify (T-1303: category/family_group, called from entry_from_hub) ----

TEST(hub, classify_maps_pipeline_tag_to_category) {
  auto llm = entry_from_hub(qwen7_info(), qwen7_config());
  EXPECT_EQ(llm.category, "LLM");
  auto vlm_info = qwen7_info();
  vlm_info["pipeline_tag"] = "image-text-to-text";
  auto vlm = entry_from_hub(vlm_info, qwen7_config());
  EXPECT_EQ(vlm.category, "VLM");
  auto other_info = qwen7_info();
  other_info["pipeline_tag"] = "text-to-image";
  auto diffusion = entry_from_hub(other_info, qwen7_config());
  EXPECT_EQ(diffusion.category, "Diffusion");
  auto unknown_info = qwen7_info();
  unknown_info["pipeline_tag"] = "something-nobody-registered";
  auto other = entry_from_hub(unknown_info, qwen7_config());
  EXPECT_EQ(other.category, "Other");
}

TEST(hub, classify_collapses_same_series_different_size_variants_into_one_family_group) {
  auto make = [](const std::string& name) {
    auto e = entry_from_hub(qwen7_info(), qwen7_config());
    e.name = name;
    classify(e);
    return e;
  };
  const auto a = make("Qwen/Qwen2.5-7B-Instruct");
  const auto b = make("Qwen/Qwen2.5-0.5B-Instruct");
  const auto c = make("Qwen/Qwen2.5-1.5B-Instruct");
  EXPECT_EQ(a.family_group, "Qwen2.5");
  EXPECT_EQ(a.family_group, b.family_group);
  EXPECT_EQ(a.family_group, c.family_group);
}

TEST(hub, classify_keeps_a_name_with_no_size_token_as_its_own_singleton_group) {
  auto e = entry_from_hub(qwen7_info(), qwen7_config());
  e.name = "hf-internal-testing/tiny-random-gpt2";
  classify(e);
  EXPECT_EQ(e.family_group, "tiny-random-gpt2");
}

TEST(hub, classify_ignores_a_bare_version_number_with_no_size_suffix) {
  // "gemma-4-26B-A4B-it": "4" alone (no B/M/K suffix) must not be mistaken for the size token; "26B" is.
  auto e = entry_from_hub(qwen7_info(), qwen7_config());
  e.name = "google/gemma-4-26B-A4B-it";
  classify(e);
  EXPECT_EQ(e.family_group, "gemma-4");
}

TEST(hub, to_spec_matches_hub_param_count) {
  const auto e = entry_from_hub(qwen7_info(), qwen7_config());
  auto s = to_spec(e);
  ASSERT_TRUE(s) << s.error().to_string();
  EXPECT_EQ(s->path, "hf:Qwen/Qwen2.5-7B-Instruct");
  EXPECT_NEAR(static_cast<double>(s->total_params), 7615616512.0, 7615616512.0 * 0.001);
  EXPECT_EQ(s->text.n_layers, 28);
  EXPECT_EQ(s->text.n_kv_heads, 4);
}

TEST(hub, catalog_roundtrip_and_search) {
  Catalog c;
  c.updated_at = "2026-09-28T00:00:00Z";
  c.entries.push_back(entry_from_hub(qwen7_info(), qwen7_config()));
  auto draft = entry_from_hub(qwen7_info(), qwen7_config());
  draft.name = "x/Qwen2.5-7B-draft";
  c.entries.push_back(draft);
  const auto p = std::filesystem::temp_directory_path() / "mqde_catalog_test.json";
  ASSERT_TRUE(save_catalog(c, p));
  auto l = load_catalog(p);
  ASSERT_TRUE(l);
  ASSERT_EQ(l->entries.size(), 2u);
  EXPECT_EQ(l->entries[0].config.value("num_hidden_layers", 0), 28);
  EXPECT_EQ(search(*l, "qwen2.5", "", "", 10).size(), 1u);  // draft is demoted
  EXPECT_EQ(search(*l, "qwen2.5", "", "", 10, true).size(), 2u);
  EXPECT_NE(find(*l, "hf:qwen/qwen2.5-7b-instruct"), nullptr);
  std::filesystem::remove(p);
}

// T-B: a query no longer has to be a contiguous substring of the name -- its characters just have to occur
// in order (fzf-style subsequence matching), so a quick abbreviated search like "qwen25" finds
// "Qwen2.5-7B-Instruct" even though "qwen25" is not literally inside that string.
TEST(hub, search_is_fuzzy_a_subsequence_of_characters_in_order_not_only_a_contiguous_substring) {
  Catalog c;
  auto mk = [](std::string name, int64_t downloads) {
    CatalogEntry e;
    e.name = name;
    e.provider = "test";
    e.parameter_count = "7B";
    e.quantization = "BF16";
    e.format = "safetensors";
    e.hf_downloads = downloads;
    return e;
  };
  c.entries.push_back(mk("Qwen/Qwen2.5-7B-Instruct", 100));
  c.entries.push_back(mk("meta-llama/Llama-3.1-8B-Instruct", 50));
  c.entries.push_back(mk("mistralai/Mistral-7B-v0.1", 10));

  // "qwen25" is not a contiguous substring of "Qwen/Qwen2.5-7B-Instruct" (there's a '.' in the way), but
  // every one of its characters occurs in that order -- a plain substring search would find nothing.
  auto hits = search(c, "qwen25", "", "", 10);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0]->name, "Qwen/Qwen2.5-7B-Instruct");

  // a subsequence that could match more than one entry (both names contain an 'l', 'a', 'm' in order) must
  // not swallow everything -- "llama31" is specific enough that only Llama should come back
  auto hits2 = search(c, "llama31", "", "", 10);
  ASSERT_EQ(hits2.size(), 1u);
  EXPECT_EQ(hits2[0]->name, "meta-llama/Llama-3.1-8B-Instruct");

  // no query still returns everything, unaffected by the new scoring
  EXPECT_EQ(search(c, "", "", "", 10).size(), 3u);

  // a query matching nothing at all (character not present in any name) returns nothing, not everything
  EXPECT_TRUE(search(c, "zzzzz", "", "", 10).empty());
}

TEST(hub, search_still_puts_an_exact_substring_hit_first_and_falls_back_to_downloads_on_ties) {
  Catalog c;
  auto mk = [](std::string name, int64_t downloads) {
    CatalogEntry e;
    e.name = name;
    e.provider = "test";
    e.parameter_count = "7B";
    e.quantization = "BF16";
    e.format = "safetensors";
    e.hf_downloads = downloads;
    return e;
  };
  // "qwen" is a fuzzy subsequence of both, but a literal substring of only the first -- becoming fuzzy must
  // not make an exact hit rank behind (or even tie with) a merely-fuzzy one
  c.entries.push_back(mk("some/qwen-fork", 5));
  c.entries.push_back(mk("other/Quietly-When-Everyone-Naps", 1000));
  auto hits = search(c, "qwen", "", "", 10);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0]->name, "some/qwen-fork") << "the exact substring hit must rank first despite far fewer downloads";

  // same score (both exact substring hits) -> download count still breaks the tie, as before
  c.entries.push_back(mk("a/qwen-small", 2));
  c.entries.push_back(mk("b/qwen-big", 2000));
  auto tie = search(c, "qwen", "", "", 10);
  const auto idx = [&](const char* n) {
    for (size_t i = 0; i < tie.size(); ++i)
      if (tie[i]->name == n) return i;
    return tie.size();
  };
  EXPECT_LT(idx("b/qwen-big"), idx("a/qwen-small"));
}

// A short query is the sharpest version of "exact must still win": with only two characters, a fuzzy hit
// where both characters happen to start their own word (maximum boundary bonus, no penalty for not being
// consecutive) can out-score an exact substring match that happens to sit mid-word in its name -- unless
// the exact-match bonus is there specifically to prevent that.
TEST(hub, a_short_exact_hit_still_beats_a_short_fuzzy_hit_with_favorable_word_boundaries) {
  Catalog c;
  auto mk = [](std::string name) {
    CatalogEntry e;
    e.name = name;
    e.provider = "test";
    e.parameter_count = "7B";
    e.quantization = "BF16";
    e.format = "safetensors";
    e.hf_downloads = 0;
    return e;
  };
  c.entries.push_back(mk("x/crab-model"));   // "ab" is an exact substring, but mid-word (not at a boundary)
  c.entries.push_back(mk("x/a-b-corp"));     // "ab" only matches as a fuzzy subsequence, one word-start each
  auto hits = search(c, "ab", "", "", 10);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0]->name, "x/crab-model")
      << "an exact (if mid-word) substring hit must still outrank a fuzzy hit that only wins on word-boundary bonuses";
}

TEST(hub, packed_int_checkpoint_uses_name_size) {
  auto info = qwen7_info();
  info["id"] = "someone/Model-27B-AWQ";
  info["safetensors"] =
      Json::parse(R"({"parameters":{"I32":3000000000,"F16":800000000},"total":3800000000})");
  auto cfg = qwen7_config();
  cfg["quantization_config"] =
      Json::parse(R"({"quant_method":"awq","bits":4,"group_size":128,"modules_to_not_convert":["a","b"]})");
  const auto e = entry_from_hub(info, cfg);
  EXPECT_EQ(e.quantization, "INT4");
  EXPECT_EQ(e.parameters_raw, 27000000000);
  EXPECT_FALSE(e.config["quantization_config"].contains("modules_to_not_convert"));
  EXPECT_FALSE(e.demoted) << e.demote_reason;
}

TEST(hub, gpt2_style_keys_are_kept) {
  auto info = qwen7_info();
  info["id"] = "openai-community/gpt2";
  info["safetensors"] = Json::parse(R"({"parameters":{"F32":137022720},"total":137022720})");
  const auto cfg = Json::parse(
      R"({"model_type":"gpt2","n_layer":12,"n_embd":768,"n_head":12,"n_ctx":1024,"n_positions":1024,"vocab_size":50257})");
  const auto e = entry_from_hub(info, cfg);
  EXPECT_FALSE(e.demoted) << e.demote_reason;
  EXPECT_EQ(e.context_length, 1024);
  auto sp = to_spec(e);
  ASSERT_TRUE(sp) << sp.error().to_string();
  EXPECT_EQ(sp->text.n_layers, 12);
  EXPECT_EQ(sp->text.hidden, 768);
}
