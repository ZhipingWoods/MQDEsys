// T-0210: compatibility matrix (export-format packing rules, backend known issues) and JobSpec validation.
#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include "mqde/planner/validate.h"

using namespace mqde;
using namespace mqde::planner;
using quant::Granularity;
using quant::NumFormat;
using quant::QuantSpec;

namespace {

QuantSpec w_int(int bits, Granularity g = Granularity::kPerGroup, bool sym = false, int64_t gs = 128) {
  QuantSpec s;
  s.fmt = NumFormat::kInt;
  s.bits = bits;
  s.gran = g;
  s.symmetric = sym;
  s.group_size = gs;
  return s;
}
QuantSpec w_fp8(Granularity g = Granularity::kPerChannel) {
  QuantSpec s;
  s.fmt = NumFormat::kFp8E4M3;
  s.bits = 8;
  s.gran = g;
  return s;
}
QuantSpec none() { return QuantSpec{}; }

estimator::HardwareProfile hw4090() {
  auto r = estimator::load_hardware("nvidia/rtx4090");
  EXPECT_TRUE(r) << (r ? "" : r.error().message);
  return r ? *r : estimator::HardwareProfile{};
}

}  // namespace

// ---------------------------------------------------------------- algorithm registry / resolve

TEST(planner_algorithms, lists_the_real_catalog_with_the_right_metadata) {
  auto all = list_algorithms();
  ASSERT_TRUE(all) << all.error().message;
  std::set<std::string> names;
  for (const auto& a : *all) names.insert(a.name);
  for (const char* n : {"rtn", "gptq", "awq", "smoothquant", "awq+gptq", "smoothquant+gptq"})
    EXPECT_TRUE(names.count(n)) << n;
  auto gptq = load_algorithm("gptq");
  ASSERT_TRUE(gptq);
  EXPECT_EQ(gptq->kind, "quantizer");
  EXPECT_TRUE(gptq->needs_calibration);
  auto rtn = load_algorithm("rtn");
  ASSERT_TRUE(rtn);
  EXPECT_FALSE(rtn->needs_calibration);
  auto bad = load_algorithm("nonsense");
  ASSERT_FALSE(bad);
  EXPECT_EQ(bad.error().code, Errc::kNotFound);
  EXPECT_NE(bad.error().message.find("unknown algorithm 'nonsense'"), std::string::npos);
}

TEST(planner_resolve, matches_the_python_resolve_exactly_same_parts_same_order_same_rejections) {
  auto a = resolve_algorithm("awq+gptq");
  ASSERT_TRUE(a);
  EXPECT_EQ(*a, (std::vector<std::string>{"awq", "gptq"}));
  EXPECT_EQ(*resolve_algorithm("smoothquant+gptq"), (std::vector<std::string>{"smoothquant", "gptq"}));
  EXPECT_EQ(*resolve_algorithm("rtn"), (std::vector<std::string>{"rtn"}));
  EXPECT_EQ(*resolve_algorithm("none"), (std::vector<std::string>{"rtn"}));
  EXPECT_EQ(*resolve_algorithm(""), (std::vector<std::string>{"rtn"}));
  EXPECT_EQ(*resolve_algorithm("AWQ"), (std::vector<std::string>{"awq", "rtn"}))
      << "case-insensitive, and a lone transform gets the implicit rtn quantizer";
  EXPECT_EQ(*resolve_algorithm(" gptq "), (std::vector<std::string>{"gptq"}));
  auto bad_name = resolve_algorithm("made_up");
  ASSERT_FALSE(bad_name);
  EXPECT_NE(bad_name.error().message.find("unknown quantization algorithm 'made_up'"), std::string::npos);
  auto two_transforms = resolve_algorithm("awq+smoothquant");
  ASSERT_FALSE(two_transforms);
  EXPECT_NE(two_transforms.error().message.find("at most one transform"), std::string::npos);
  auto two_quantizers = resolve_algorithm("rtn+gptq");
  ASSERT_FALSE(two_quantizers);
  EXPECT_NE(two_quantizers.error().message.find("at most one transform"), std::string::npos);
}

// ---------------------------------------------------------------- check_export: recipe-level (no shape)

TEST(planner_check_export, the_five_shipped_recipes_against_their_own_default_algorithm_export) {
  // w4a16 (int4, per_group, asymmetric) -> gptq and awq both accept it
  EXPECT_EQ(check_export("gptq", w_int(4), none()).verdict, Verdict::kNative);
  EXPECT_EQ(check_export("awq", w_int(4), none()).verdict, Verdict::kNative);
  // w2a16 (int2) -> gptq accepts (bits_ok includes 2), awq does not (bits_ok = {4} only)
  EXPECT_EQ(check_export("gptq", w_int(2), none()).verdict, Verdict::kNative);
  auto awq2 = check_export("awq", w_int(2), none());
  EXPECT_EQ(awq2.verdict, Verdict::kUnsupported);
  EXPECT_NE(awq2.reason.find("2-bit weights are not supported by awq"), std::string::npos) << awq2.reason;
  // fp8 recipe -> compressed-tensors-fp8 accepts it, gptq/awq (integer-only containers) do not
  EXPECT_EQ(check_export("compressed-tensors", w_fp8(), none()).verdict, Verdict::kNative);
  auto gptq_fp8 = check_export("gptq", w_fp8(), none());
  EXPECT_EQ(gptq_fp8.verdict, Verdict::kUnsupported);
  EXPECT_NE(gptq_fp8.reason.find("needs int weights"), std::string::npos) << gptq_fp8.reason;
}

TEST(planner_check_export, per_tensor_weight_scales_are_rejected_by_gptq_and_awq) {
  auto g = check_export("gptq", w_int(4, Granularity::kPerTensor), none());
  EXPECT_EQ(g.verdict, Verdict::kUnsupported);
  EXPECT_NE(g.reason.find("per-tensor"), std::string::npos);
  EXPECT_EQ(check_export("awq", w_int(4, Granularity::kPerTensor), none()).verdict, Verdict::kUnsupported);
}

TEST(planner_check_export, fp8_per_group_is_rejected_but_per_channel_and_per_tensor_are_not) {
  EXPECT_EQ(check_export("compressed-tensors", w_fp8(Granularity::kPerGroup), none()).verdict,
            Verdict::kUnsupported);
  EXPECT_EQ(check_export("compressed-tensors", w_fp8(Granularity::kPerChannel), none()).verdict,
            Verdict::kNative);
  EXPECT_EQ(check_export("compressed-tensors", w_fp8(Granularity::kPerTensor), none()).verdict,
            Verdict::kNative);
}

TEST(planner_check_export,
     integer_weights_with_quantized_activations_are_not_exported_by_compressed_tensors) {
  QuantSpec a = w_int(8, Granularity::kPerChannel);  // any enabled act spec
  auto r = check_export("compressed-tensors", w_int(4), a);
  EXPECT_EQ(r.verdict, Verdict::kUnsupported);
  EXPECT_NE(r.reason.find("W4A8/W8A8-int"), std::string::npos) << r.reason;
  EXPECT_EQ(check_export("compressed-tensors", w_int(4), none()).verdict, Verdict::kNative)
      << "disabled activations are fine";
}

TEST(planner_check_export, fp8_weights_need_matching_fp8_activations_when_activations_are_quantized_at_all) {
  QuantSpec bf16_act = w_int(16, Granularity::kPerChannel);
  bf16_act.fmt = NumFormat::kInt;
  bf16_act.bits = 16;
  auto mismatched = check_export("compressed-tensors", w_fp8(), bf16_act);
  EXPECT_EQ(mismatched.verdict, Verdict::kUnsupported);
  EXPECT_NE(mismatched.reason.find("int16"), std::string::npos) << mismatched.reason;
  // "per_token" (the fp8 recipe's own dynamic activation granularity) maps to kPerChannel in this QuantSpec
  // (see parse_quant_spec); any fp8 activation granularity is accepted here since "must be per-tensor" only
  // applies to a *static* scale, which is a per-job fact this recipe-level check cannot see (see
  // check_export's comment).
  for (auto g : {Granularity::kPerChannel, Granularity::kPerGroup, Granularity::kPerTensor})
    EXPECT_EQ(check_export("compressed-tensors", w_fp8(), w_fp8(g)).verdict, Verdict::kNative)
        << static_cast<int>(g);
  EXPECT_EQ(check_export("compressed-tensors", w_fp8(), none()).verdict, Verdict::kNative)
      << "activations not quantized at all";
}

// ---------------------------------------------------------------- check_export: shape-dependent packing

TEST(planner_check_export, awq_needs_in_features_divisible_by_group_size_gptq_does_not_check_that) {
  // 4104 is a multiple of 8 (the 4-bit pack divisor, so GPTQ's own packing check passes) but not of 128
  // (group_size).
  LayerShape ok{4096, 4096}, bad{4096, 4104};
  EXPECT_EQ(check_export("awq", w_int(4, Granularity::kPerGroup, false, 128), none(), ok).verdict,
            Verdict::kNative);
  auto r = check_export("awq", w_int(4, Granularity::kPerGroup, false, 128), none(), bad);
  EXPECT_EQ(r.verdict, Verdict::kUnsupported);
  EXPECT_NE(r.reason.find("not divisible by group_size"), std::string::npos) << r.reason;
  EXPECT_EQ(check_export("gptq", w_int(4, Granularity::kPerGroup, false, 128), none(), bad).verdict,
            Verdict::kNative)
      << "GPTQ's container layout does not require in_features % group_size == 0";
}

TEST(planner_check_export, gptq_checks_both_dims_pack_into_32_bit_words_awq_only_checks_out_features) {
  // 3-bit weights: 32-bit word holds 10.67 codes, so dims not a multiple of... use a dim not covered by
  // 32/bits.
  LayerShape odd_in{512, 511};  // out divisible, in not (at 4 bit: pf=8, 511 % 8 != 0)
  auto g = check_export("gptq", w_int(4, Granularity::kPerGroup, false, 511), none(), odd_in);
  EXPECT_EQ(g.verdict, Verdict::kUnsupported);
  EXPECT_NE(g.reason.find("in_features 511"), std::string::npos) << g.reason;
  LayerShape odd_out{511, 512};
  EXPECT_EQ(check_export("gptq", w_int(4, Granularity::kPerGroup, false, 512), none(), odd_out).verdict,
            Verdict::kUnsupported);
  // AWQ's check_layer only tests out_features for the generic pack divisor (in_features is covered separately
  // by the group check)
  QuantSpec awq_spec = w_int(4, Granularity::kPerGroup, false, 511);
  EXPECT_EQ(check_export("awq", awq_spec, none(), {{512, 511}}).verdict, Verdict::kNative)
      << "in_features 511 not a multiple of 8 but AWQ does not check that dim this way";
}

TEST(planner_check_export, compressed_tensors_int_needs_out_features_packing_only_when_asymmetric) {
  LayerShape odd_out{511, 512};
  EXPECT_EQ(
      check_export("compressed-tensors", w_int(4, Granularity::kPerChannel, true), none(), odd_out).verdict,
      Verdict::kNative)
      << "symmetric: no zero-point packing needed";
  auto r = check_export("compressed-tensors", w_int(4, Granularity::kPerChannel, false), none(), odd_out);
  EXPECT_EQ(r.verdict, Verdict::kUnsupported);
  EXPECT_NE(r.reason.find("zero points"), std::string::npos) << r.reason;
}

TEST(planner_check_export, an_unknown_format_is_unsupported_with_a_clear_message) {
  auto r = check_export("made_up_format", w_int(4), none());
  EXPECT_EQ(r.verdict, Verdict::kUnsupported);
  EXPECT_NE(r.reason.find("unknown export format"), std::string::npos);
}

// ---------------------------------------------------------------- check_deploy

TEST(planner_check_deploy, asymmetric_gptq_on_vllm_is_the_specific_known_issue_symmetric_is_not) {
  estimator::Recipe r;
  r.weight = w_int(4, Granularity::kPerGroup, /*sym=*/false);
  auto bad = check_deploy("gptq", r, "vllm", hw4090());
  EXPECT_EQ(bad.verdict, Verdict::kUnsupported);
  EXPECT_NE(bad.reason.find("vLLM 0.30.0 rejects asymmetric GPTQ"), std::string::npos) << bad.reason;
  estimator::Recipe r2;
  r2.weight = w_int(4, Granularity::kPerGroup, /*sym=*/true);
  auto ok = check_deploy("gptq", r2, "vllm", hw4090());
  EXPECT_EQ(ok.verdict, Verdict::kNative) << ok.reason;
}

TEST(planner_check_deploy, awq_and_compressed_tensors_load_on_vllm_with_no_known_issue) {
  estimator::Recipe r;
  r.weight = w_int(4, Granularity::kPerGroup, false);
  EXPECT_EQ(check_deploy("awq", r, "vllm", hw4090()).verdict, Verdict::kNative);
  estimator::Recipe fp8;
  fp8.weight = w_fp8();
  EXPECT_EQ(check_deploy("compressed-tensors", fp8, "vllm", hw4090()).verdict, Verdict::kNative);
}

// T-0704: every claim below was measured on a real SGLang (0.5.22.dev, 86624f14eb) serving Qwen2.5-0.5B on an
// RTX 4090, served PPL within 0.05% of the real-weight PPL; the asymmetric-GPTQ failure is SGLang's own error
// text.
TEST(planner_check_deploy, awq_gptq_and_both_compressed_tensors_kinds_load_on_sglang) {
  estimator::Recipe r;
  r.weight = w_int(4, Granularity::kPerGroup, false);
  EXPECT_EQ(check_deploy("awq", r, "sglang", hw4090()).verdict, Verdict::kNative);
  estimator::Recipe fp8;
  fp8.weight = w_fp8();
  EXPECT_EQ(check_deploy("compressed-tensors", fp8, "sglang", hw4090()).verdict, Verdict::kNative);
  EXPECT_EQ(check_deploy("compressed-tensors", r, "sglang", hw4090()).verdict, Verdict::kNative);
}

TEST(planner_check_deploy, asymmetric_gptq_on_sglang_is_the_specific_known_issue_symmetric_is_not) {
  estimator::Recipe asym;
  asym.weight = w_int(4, Granularity::kPerGroup, /*sym=*/false);
  auto bad = check_deploy("gptq", asym, "sglang", hw4090());
  EXPECT_EQ(bad.verdict, Verdict::kUnsupported);
  EXPECT_NE(bad.reason.find("gptq_marlin"), std::string::npos) << bad.reason;
  estimator::Recipe sym;
  sym.weight = w_int(4, Granularity::kPerGroup, /*sym=*/true);
  EXPECT_EQ(check_deploy("gptq", sym, "sglang", hw4090()).verdict, Verdict::kNative);
}

TEST(planner_preferred_export_format, sglang_gets_the_same_weight_driven_format_as_vllm) {
  estimator::Recipe w4;
  w4.weight = w_int(4, Granularity::kPerGroup, false);
  EXPECT_EQ(preferred_export_format(w4.weight, "sglang"), "awq");
  EXPECT_EQ(preferred_export_format(w4.weight, "sglang"), preferred_export_format(w4.weight, "vllm"));
}

TEST(planner_check_deploy, hf_transformers_is_never_a_deployment_target) {
  estimator::Recipe r;
  r.weight = w_int(4);
  auto x = check_deploy("awq", r, "hf_transformers", hw4090());
  EXPECT_EQ(x.verdict, Verdict::kUnsupported);
  EXPECT_NE(x.reason.find("does not load awq"), std::string::npos) << x.reason;
}

TEST(planner_check_deploy, a_format_outside_the_hardwares_native_formats_is_emulated_not_rejected) {
  estimator::HardwareProfile weak;
  weak.id = "weak/gpu";
  weak.name = "Weak";
  weak.arch = "x";
  weak.native_formats = {"bf16"};  // no int4
  estimator::Recipe r;
  r.weight = w_int(4, Granularity::kPerGroup, true);
  auto x = check_deploy("gptq", r, "vllm", weak);
  EXPECT_EQ(x.verdict, Verdict::kEmulated);
  EXPECT_NE(x.reason.find("no native kernel"), std::string::npos) << x.reason;
}

TEST(planner_check_deploy, an_unknown_backend_is_unsupported) {
  estimator::Recipe r;
  r.weight = w_int(4);
  EXPECT_EQ(check_deploy("gptq", r, "made_up_backend", hw4090()).verdict, Verdict::kUnsupported);
}

// GGUF re-quantizes from the original FP weights with llama.cpp's own numerics (python/mqde/pipeline/
// llm_job.py's _llamacpp, independent of quant.recipe) -- so unlike gptq/awq/compressed-tensors, whether it
// "fits" has nothing to do with the recipe's weight format, and checking hw.native_formats against the
// recipe's int/fp8 spec would be checking the wrong thing entirely.
TEST(planner_check_deploy, gguf_on_llamacpp_is_always_native_regardless_of_the_recipes_weight_format) {
  estimator::Recipe int4;
  int4.weight = w_int(4, Granularity::kPerGroup, true);
  EXPECT_EQ(check_deploy("gguf", int4, "llamacpp", hw4090()).verdict, Verdict::kNative);
  estimator::Recipe fp8r;
  fp8r.weight = w_fp8();
  EXPECT_EQ(check_deploy("gguf", fp8r, "llamacpp", hw4090()).verdict, Verdict::kNative)
      << "gguf ignores the recipe's weight spec entirely, so an fp8 recipe must not be rejected either";
  estimator::Recipe bf16r;  // no quantization at all (weight disabled)
  EXPECT_EQ(check_deploy("gguf", bf16r, "llamacpp", hw4090()).verdict, Verdict::kNative);
}

TEST(planner_check_deploy, gguf_does_not_load_on_backends_other_than_llamacpp) {
  estimator::Recipe r;
  r.weight = w_int(4);
  EXPECT_EQ(check_deploy("gguf", r, "vllm", hw4090()).verdict, Verdict::kUnsupported);
}

// ---------------------------------------------------------------- validate_jobspec

TEST(planner_validate_jobspec, a_valid_llm_job_has_no_problems) {
  Json job = {{"model", {{"type", "llm"}}},
              {"quant", {{"recipe", Json::array({"w4a16"})}, {"algorithm", "recipe"}}},
              {"deploy", {{"device", "gpu"}}}};
  EXPECT_TRUE(validate_jobspec(job).empty());
}

TEST(planner_validate_jobspec, an_empty_recipe_list_is_one_problem_with_a_suggestion) {
  Json job = {{"quant", {{"recipe", Json::array()}}}};
  auto p = validate_jobspec(job);
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0].field, "/quant/recipe");
  EXPECT_FALSE(p[0].suggestion.empty());
}

TEST(planner_validate_jobspec, every_problem_is_reported_not_just_the_first) {
  Json job = {{"model", {{"type", "robot"}}},
              {"quant", {{"recipe", Json::array({"not-a-recipe", "w4a16"})}, {"algorithm", "not-an-algo"}}},
              {"deploy", {{"device", "cpu"}}}};
  auto p = validate_jobspec(job);
  std::set<std::string> fields;
  for (const auto& x : p) fields.insert(x.field);
  EXPECT_TRUE(fields.count("/quant/recipe"));
  EXPECT_TRUE(fields.count("/quant/algorithm"));
  EXPECT_TRUE(fields.count("/model/type"));
  EXPECT_TRUE(fields.count("/deploy/device"))
      << "cpu is invalid for a non-llm job, reported alongside the others";
  EXPECT_GE(p.size(), 4u);
}

TEST(planner_validate_jobspec, an_algorithm_needing_calibration_without_samples_is_a_problem) {
  Json job = {{"model", {{"type", "llm"}}},
              {"quant", {{"recipe", Json::array({"w4a16"})}, {"algorithm", "gptq"}}},
              {"calibration", {{"n_samples", 0}}}};
  auto p = validate_jobspec(job);
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0].field, "/calibration/n_samples");
  Json ok = job;
  ok["calibration"]["n_samples"] = 32;
  EXPECT_TRUE(validate_jobspec(ok).empty());
}

TEST(planner_validate_jobspec, rtn_never_needs_calibration_samples) {
  Json job = {{"model", {{"type", "llm"}}},
              {"quant", {{"recipe", Json::array({"w4a16"})}, {"algorithm", "rtn"}}}};
  EXPECT_TRUE(validate_jobspec(job).empty());
}

namespace {
Json llm_with_samples(Json samples) {
  return {{"model", {{"type", "llm"}}},
          {"quant", {{"recipe", Json::array({"w4a16"})}, {"algorithm", "rtn"}}},
          {"calibration", {{"custom_samples", std::move(samples)}}}};
}
}  // namespace

TEST(planner_validate_jobspec, pasted_calibration_samples_are_a_list_of_text) {
  EXPECT_TRUE(validate_jobspec(llm_with_samples(Json::array({"some real text to calibrate on", "and a second sample"}))).empty());
  EXPECT_TRUE(validate_jobspec(llm_with_samples(Json::array())).empty()) << "an empty list is the same as not giving any";
}

TEST(planner_validate_jobspec, pasted_samples_that_are_not_a_list_of_strings_are_refused_with_the_field) {
  for (const Json& bad : {Json("one long string"), Json(5), Json::object({{"a", "b"}})}) {
    auto p = validate_jobspec(llm_with_samples(bad));
    ASSERT_EQ(p.size(), 1u) << bad.dump();
    EXPECT_EQ(p[0].field, "/calibration/custom_samples");
    EXPECT_NE(p[0].message.find("list of strings"), std::string::npos) << p[0].message;
  }
  auto p = validate_jobspec(llm_with_samples(Json::array({"fine text", 7, "also fine"})));
  ASSERT_EQ(p.size(), 1u);
  EXPECT_NE(p[0].message.find("#1"), std::string::npos) << "names which entry (counting from 0): " << p[0].message;
}

TEST(planner_validate_jobspec, pasted_samples_that_are_all_blank_are_refused) {
  auto p = validate_jobspec(llm_with_samples(Json::array({"", "   ", "\n\t"})));
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0].field, "/calibration/custom_samples");
  EXPECT_NE(p[0].message.find("blank"), std::string::npos) << p[0].message;
  EXPECT_TRUE(validate_jobspec(llm_with_samples(Json::array({"", "one real one", " "}))).empty()) << "blank ones among real ones are dropped later";
}

TEST(planner_validate_jobspec, pasted_samples_are_only_for_llm_jobs) {
  for (const char* type : {"vla", "vlm"}) {
    Json job = {{"model", {{"type", type}}},
                {"quant", {{"recipe", Json::array({"w4a16"})}}},
                {"calibration", {{"dataset", "synthetic"}, {"custom_samples", Json::array({"text"})}}}};
    auto p = validate_jobspec(job);
    bool found = false;
    for (const auto& x : p) found |= x.field == "/calibration/custom_samples";
    EXPECT_TRUE(found) << type << " jobs would silently ignore the text";
  }
}

TEST(planner_validate_jobspec, an_export_format_incompatible_with_the_recipe_is_reported) {
  Json job = {{"model", {{"type", "llm"}}},
              {"quant", {{"recipe", Json::array({"w2a16"})}}},
              {"export", {{"formats", Json::array({"awq"})}}}};
  auto p = validate_jobspec(job);
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0].field, "/export/formats");
  EXPECT_NE(p[0].message.find("W2A16"), std::string::npos) << p[0].message;
  Json ok = job;
  ok["export"]["formats"] = Json::array({"gptq"});
  EXPECT_TRUE(validate_jobspec(ok).empty());
}

TEST(planner_validate_jobspec, export_compatibility_is_only_checked_for_llm_jobs) {
  Json job = {{"model", {{"type", "vla"}}},
              {"quant", {{"recipe", Json::array({"w2a16"})}}},
              {"export", {{"formats", Json::array({"awq"})}}}};
  EXPECT_TRUE(validate_jobspec(job).empty()) << "VLA jobs have no exporter yet; nothing to check";
}

TEST(planner_validate_jobspec, missing_fields_default_sensibly_and_do_not_crash) {
  EXPECT_FALSE(validate_jobspec(Json::object()).empty());  // no recipe at all
  EXPECT_NO_FATAL_FAILURE(validate_jobspec({{"quant", "not an object"}}));
  EXPECT_NO_FATAL_FAILURE(validate_jobspec({{"quant", {{"recipe", "not an array"}}}}));
}

// ---- domain (T-1305, uses T-1301's data/domains/*.yaml registry) ----

TEST(planner_validate_jobspec, an_unknown_domain_id_is_a_problem) {
  Json job = {{"domain", "not-a-real-domain"},
              {"model", {{"type", "llm"}}},
              {"quant", {{"recipe", Json::array({"w4a16"})}}}};
  auto p = validate_jobspec(job);
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0].field, "/domain");
}

TEST(planner_validate_jobspec, a_domain_with_no_real_pipeline_is_rejected_with_a_clear_reason) {
  for (const char* id : {"world", "ewm"}) {
    Json job = {
        {"domain", id}, {"model", {{"type", "llm"}}}, {"quant", {{"recipe", Json::array({"w4a16"})}}}};
    auto p = validate_jobspec(job);
    ASSERT_EQ(p.size(), 1u) << id;
    EXPECT_EQ(p[0].field, "/domain") << id;
    EXPECT_NE(p[0].message.find("no"), std::string::npos) << id << ": " << p[0].message;
  }
}

TEST(planner_validate_jobspec, llm_and_vla_domains_have_a_real_pipeline_and_add_no_problem) {
  Json llm_job = {
      {"domain", "llm"}, {"model", {{"type", "llm"}}}, {"quant", {{"recipe", Json::array({"w4a16"})}}}};
  EXPECT_TRUE(validate_jobspec(llm_job).empty());
  Json vla_job = {{"domain", "vla"},
                  {"model", {{"type", "vla"}}},
                  {"quant", {{"recipe", Json::array({"w4a16"})}}},
                  {"deploy", {{"device", "gpu"}}}};
  EXPECT_TRUE(validate_jobspec(vla_job).empty());
}

TEST(planner_validate_jobspec, vlm_is_a_real_domain_and_model_type_and_needs_a_gpu) {
  Json vlm = {{"domain", "vlm"},
              {"model", {{"type", "vlm"}}},
              {"quant", {{"recipe", Json::array({"w4a16"})}}},
              {"deploy", {{"device", "gpu"}}}};
  EXPECT_TRUE(validate_jobspec(vlm).empty());
  vlm["deploy"]["device"] = "cpu";
  auto p = validate_jobspec(vlm);
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0].field, "/deploy/device");
  Json bogus = {{"model", {{"type", "vlmm"}}}, {"quant", {{"recipe", Json::array({"w4a16"})}}}};
  ASSERT_EQ(validate_jobspec(bogus).size(), 1u);
  EXPECT_NE(validate_jobspec(bogus)[0].message.find("vlm"), std::string::npos)
      << "the message lists the valid types";
}

TEST(planner_validate_jobspec, domain_is_entirely_optional_and_omitting_it_adds_no_problem) {
  Json job = {{"model", {{"type", "llm"}}}, {"quant", {{"recipe", Json::array({"w4a16"})}}}};
  EXPECT_TRUE(validate_jobspec(job).empty());
}

// ---- quantize-only (evaluation.skip, T-1304/T-1306): conflicts with anything that needs an evaluated
// baseline, matching llm_job.py's / vla_job.py's own quantize_only branches exactly (see their comments).

TEST(planner_validate_jobspec, quantize_only_alone_is_fine_for_llm_and_vla) {
  Json llm_job = {{"model", {{"type", "llm"}}},
                  {"quant", {{"recipe", Json::array({"w4a16"})}}},
                  {"evaluation", {{"skip", true}}}};
  EXPECT_TRUE(validate_jobspec(llm_job).empty());
  Json vla_job = {{"model", {{"type", "vla"}}},
                  {"quant", {{"recipe", Json::array({"w4a16"})}}},
                  {"deploy", {{"device", "gpu"}}},
                  {"evaluation", {{"skip", true}, {"attribution", false}}}};
  EXPECT_TRUE(validate_jobspec(vla_job).empty());
}

TEST(planner_validate_jobspec, quantize_only_conflicts_with_export_vllm_llamacpp_and_analysis_for_llm) {
  Json base = {{"model", {{"type", "llm"}}},
               {"quant", {{"recipe", Json::array({"w4a16"})}}},
               {"evaluation", {{"skip", true}}}};
  Json with_export = base;
  with_export["export"] = {{"formats", Json::array({"awq"})}};
  EXPECT_EQ(validate_jobspec(with_export).size(), 1u);

  Json with_vllm = base;
  with_vllm["deploy"] = {{"vllm", Json::object()}};
  EXPECT_EQ(validate_jobspec(with_vllm).size(), 1u);

  Json with_sglang = base;
  with_sglang["deploy"] = {{"sglang", Json::object()}};
  EXPECT_EQ(validate_jobspec(with_sglang).size(), 1u);

  Json with_llamacpp = base;
  with_llamacpp["deploy"] = {{"llamacpp", {{"types", Json::array({"q4_k"})}}}};
  EXPECT_EQ(validate_jobspec(with_llamacpp).size(), 1u);

  Json with_sensitivity = base;
  with_sensitivity["analysis"] = {{"sensitivity", "block"}};
  EXPECT_EQ(validate_jobspec(with_sensitivity).size(), 1u);

  Json with_logits_probe = base;
  with_logits_probe["analysis"] = {{"logits_probe", Json::object()}};
  EXPECT_EQ(validate_jobspec(with_logits_probe).size(), 1u);

  for (const auto& j :
       {with_export, with_vllm, with_sglang, with_llamacpp, with_sensitivity, with_logits_probe}) {
    auto p = validate_jobspec(j);
    ASSERT_EQ(p.size(), 1u);
    EXPECT_EQ(p[0].field, "/evaluation/skip");
  }
}

TEST(planner_validate_jobspec, quantize_only_conflicts_with_vlm_attribution_too_including_the_default) {
  Json base = {{"model", {{"type", "vlm"}}},
               {"quant", {{"recipe", Json::array({"w4a16"})}}},
               {"deploy", {{"device", "gpu"}}},
               {"evaluation", {{"skip", true}}}};
  auto p = validate_jobspec(base);  // attribution omitted: defaults to true, needs the reference this skips
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0].field, "/evaluation/skip");
  base["evaluation"]["attribution"] = false;
  EXPECT_TRUE(validate_jobspec(base).empty());
}

TEST(planner_validate_jobspec, quantize_only_conflicts_with_vla_attribution_true_including_the_default) {
  Json explicit_true = {{"model", {{"type", "vla"}}},
                        {"quant", {{"recipe", Json::array({"w4a16"})}}},
                        {"deploy", {{"device", "gpu"}}},
                        {"evaluation", {{"skip", true}, {"attribution", true}}}};
  ASSERT_EQ(validate_jobspec(explicit_true).size(), 1u);
  // attribution defaults to true in vla_job.py's JobSpec dataclass when the field is absent entirely, so
  // the check must use the same default -- omitting it must be flagged exactly like setting it explicitly.
  Json omitted = {{"model", {{"type", "vla"}}},
                  {"quant", {{"recipe", Json::array({"w4a16"})}}},
                  {"deploy", {{"device", "gpu"}}},
                  {"evaluation", {{"skip", true}}}};
  EXPECT_EQ(validate_jobspec(omitted).size(), 1u);
}

// ---------------------------------------------------------------- recommend

namespace {

// A 7B-parameter dense llama, built the same way cpp/tests/unit/concurrency_test.cpp's tiny_dense() does (one
// real weight tensor sized to the target parameter count; estimate() only needs total bytes for the memory
// numbers used here, not real per-layer shapes).
modelspec::ModelSpec dense_7b() {
  Json cfg = {{"model_type", "llama"},      {"num_hidden_layers", 32},  {"hidden_size", 4096},
              {"num_attention_heads", 32},  {"num_key_value_heads", 8}, {"head_dim", 128},
              {"intermediate_size", 14336}, {"vocab_size", 128256},     {"max_position_embeddings", 131072}};
  const int64_t rows = 4096, cols = static_cast<int64_t>(7e9 / rows);
  std::vector<modelspec::TensorInfo> t = {{"model.layers.0.mlp.up_proj.weight",
                                           "BF16",
                                           {rows, cols},
                                           rows * cols,
                                           static_cast<uint64_t>(rows * cols * 2),
                                           "m"}};
  auto fam = modelspec::find_model_family("llama");
  auto spec = modelspec::build_model_spec(cfg, t, *fam);
  EXPECT_TRUE(spec) << (spec ? "" : std::string(spec.error().message));
  spec->path = "test/dense7b";
  return *spec;
}

estimator::HardwareProfile gpu24() {
  estimator::HardwareProfile h;
  h.id = "test/24g";
  h.name = "Test 24GB";
  h.arch = "sm_89";
  h.mem_bytes = 24ull << 30;
  h.mem_bw_gbps = 1000;
  h.peak_tflops = {{"bf16", 100}, {"fp8", 200}, {"int8", 200}, {"int4", 400}};
  h.native_formats = {"bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8", "int4"};
  return h;
}

estimator::BackendProfile vllm_be() {
  estimator::BackendProfile b;
  b.id = "vllm";
  b.runtime_overhead_bytes = 400 << 20;
  b.allocator_factor = 1.0;
  b.activation_model = "batched";
  b.max_batched_tokens = 8192;
  return b;
}

}  // namespace

TEST(planner_recommend,
     a_7b_model_on_24gb_at_8k_context_and_concurrency_8_recommends_w4a16_or_fp8_and_excludes_bf16) {
  estimator::RuntimeParams rt;
  rt.context_tokens = 8192;
  rt.concurrency = 8;
  rt.gpu_memory_utilization = 0.9;
  auto r = recommend(dense_7b(), {vllm_be()}, {gpu24()}, rt, 5);
  ASSERT_FALSE(r.top.empty()) << "excluded " << r.excluded;
  std::set<std::string> names;
  for (const auto& rec : r.top) names.insert(rec.recipe);
  EXPECT_TRUE(names.count("W4A16") || names.count("FP8")) << fmt::format("{}", fmt::join(names, ", "));
  for (const auto& rec : r.top) {
    EXPECT_TRUE(rec.feasible) << rec.recipe;
    EXPECT_NE(rec.recipe, "BF16") << "16 bit x 7B does not fit in 24 GiB at this context and concurrency";
    EXPECT_FALSE(rec.reasons.empty());
  }
  EXPECT_GT(r.excluded, 0)
      << "bf16 (and the 2-bit recipes being dominated) should be excluded, not silently omitted";
}

TEST(planner_recommend, every_recommendation_actually_fits_and_has_a_native_or_emulated_kernel) {
  estimator::RuntimeParams rt;
  rt.context_tokens = 4096;
  rt.concurrency = 4;
  rt.gpu_memory_utilization = 0.9;
  auto r = recommend(dense_7b(), {vllm_be()}, {gpu24()}, rt, 10);
  for (const auto& rec : r.top) {
    auto recipe = estimator::load_recipe(rec.recipe);
    ASSERT_TRUE(recipe);
    const auto est = estimator::estimate(dense_7b(), *recipe, gpu24(), vllm_be(), rt);
    EXPECT_TRUE(est.feasible) << rec.recipe;
    EXPECT_EQ(rec.peak_bytes, est.peak_bytes);
    EXPECT_NE(rec.deploy_verdict, Verdict::kUnsupported) << rec.recipe;
  }
}

TEST(planner_recommend, results_are_ranked_feasible_and_native_first_then_by_higher_concurrency) {
  estimator::RuntimeParams rt;
  rt.context_tokens = 2048;
  rt.concurrency = 1;
  rt.gpu_memory_utilization = 0.9;
  auto r = recommend(dense_7b(), {vllm_be()}, {gpu24()}, rt, 10);
  ASSERT_GE(r.top.size(), 2u);
  for (size_t i = 1; i < r.top.size(); ++i) {
    const bool prev_native = r.top[i - 1].deploy_verdict == Verdict::kNative;
    const bool cur_native = r.top[i].deploy_verdict == Verdict::kNative;
    EXPECT_TRUE(prev_native || !cur_native) << "a native-kernel result must not come after a non-native one";
    if (prev_native == cur_native) {
      EXPECT_GE(r.top[i - 1].max_concurrency, r.top[i].max_concurrency);
    }
  }
}

TEST(planner_recommend, top_k_actually_limits_the_count_and_a_tiny_hardware_excludes_everything) {
  estimator::RuntimeParams rt;
  rt.context_tokens = 2048;
  rt.concurrency = 1;
  rt.gpu_memory_utilization = 0.9;
  auto r3 = recommend(dense_7b(), {vllm_be()}, {gpu24()}, rt, 3);
  EXPECT_LE(r3.top.size(), 3u);
  estimator::HardwareProfile tiny = gpu24();
  tiny.mem_bytes = 1ull << 20;  // 1 MiB: nothing fits
  auto none = recommend(dense_7b(), {vllm_be()}, {tiny}, rt, 5);
  EXPECT_TRUE(none.top.empty());
  EXPECT_GT(none.excluded, 0);
}

TEST(planner_preferred_export_format, matches_what_check_deploy_and_check_export_were_written_for) {
  EXPECT_EQ(preferred_export_format(none()), "");
  EXPECT_EQ(preferred_export_format(w_int(4)), "awq");
  EXPECT_EQ(preferred_export_format(w_int(2)), "gptq");
  EXPECT_EQ(preferred_export_format(w_int(8)), "gptq");
  EXPECT_EQ(preferred_export_format(w_fp8()), "compressed-tensors");
}

TEST(planner_preferred_export_format, llamacpp_always_suggests_gguf_regardless_of_weight_format) {
  EXPECT_EQ(preferred_export_format(none(), "llamacpp"), "gguf");
  EXPECT_EQ(preferred_export_format(w_int(4), "llamacpp"), "gguf");
  EXPECT_EQ(preferred_export_format(w_fp8(), "llamacpp"), "gguf");
}

TEST(planner_preferred_export_format, other_backends_are_unaffected_by_the_new_backend_parameter) {
  EXPECT_EQ(preferred_export_format(w_int(4), "vllm"), "awq");
  EXPECT_EQ(preferred_export_format(w_int(4), ""), "awq");
}

// ---- B2 (作业派生): an `artifact` job serves a checkpoint a finished llm job exported, without quantizing
// ----
namespace {
Json artifact_job(const Json& artifact, const Json& deploy) {
  return {{"model", {{"type", "llm"}}}, {"artifact", artifact}, {"deploy", deploy}};
}
std::vector<std::string> fields(const std::vector<Problem>& ps) {
  std::vector<std::string> out;
  for (const auto& p : ps) out.push_back(p.field);
  return out;
}
}  // namespace

TEST(planner_validate_artifact, an_awq_checkpoint_on_sglang_and_a_gguf_on_llamacpp_are_valid) {
  EXPECT_TRUE(validate_jobspec(
                  artifact_job({{"job", "20261009-225558-b43b47"}, {"recipe", "W4A16"}, {"format", "awq"}},
                               {{"device", "gpu"}, {"sglang", Json::object()}}))
                  .empty());
  EXPECT_TRUE(validate_jobspec(artifact_job({{"job", "j1"}, {"format", "gguf"}, {"type", "q4_k"}},
                                            {{"device", "cpu"}, {"llamacpp", {{"gpu_layers", 0}}}}))
                  .empty())
      << "a GGUF on the CPU is a real device change";
}

TEST(planner_validate_artifact, each_malformed_piece_is_its_own_problem) {
  auto f = fields(validate_jobspec(
      artifact_job({{"job", "../x"}, {"format", "safetensors"}}, {{"vllm", Json::object()}})));
  EXPECT_NE(std::find(f.begin(), f.end(), "/artifact/job"), f.end());
  EXPECT_NE(std::find(f.begin(), f.end(), "/artifact/format"), f.end());
  f = fields(validate_jobspec(artifact_job({{"job", "j1"}, {"format", "awq"}}, {{"vllm", Json::object()}})));
  EXPECT_NE(std::find(f.begin(), f.end(), "/artifact/recipe"), f.end())
      << "an HF export is named by its recipe";
  f = fields(
      validate_jobspec(artifact_job({{"job", "j1"}, {"format", "gguf"}}, {{"llamacpp", Json::object()}})));
  EXPECT_NE(std::find(f.begin(), f.end(), "/artifact/type"), f.end()) << "a GGUF is named by its type";
  f = fields(validate_jobspec(artifact_job("w4a16-awq", {{"vllm", Json::object()}})));
  EXPECT_NE(std::find(f.begin(), f.end(), "/artifact"), f.end());
}

TEST(planner_validate_artifact, the_backend_must_be_able_to_load_the_format) {
  auto p = validate_jobspec(
      artifact_job({{"job", "j1"}, {"format", "gguf"}, {"type", "q4_k"}}, {{"vllm", Json::object()}}));
  ASSERT_EQ(fields(p), std::vector<std::string>{"/deploy"});
  p = validate_jobspec(
      artifact_job({{"job", "j1"}, {"recipe", "W4A16"}, {"format", "awq"}}, {{"llamacpp", Json::object()}}));
  ASSERT_EQ(fields(p), std::vector<std::string>{"/deploy"});
  p = validate_jobspec(
      artifact_job({{"job", "j1"}, {"recipe", "W4A16"}, {"format", "awq"}}, {{"device", "gpu"}}));
  ASSERT_EQ(fields(p), std::vector<std::string>{"/deploy"}) << "no serving backend at all";
}

TEST(planner_validate_artifact, vllm_and_sglang_need_a_gpu) {
  auto p = validate_jobspec(artifact_job({{"job", "j1"}, {"recipe", "W4A16"}, {"format", "awq"}},
                                         {{"device", "cpu"}, {"vllm", Json::object()}}));
  ASSERT_EQ(fields(p), std::vector<std::string>{"/deploy/device"});
}

TEST(planner_validate_artifact, quantization_side_options_do_not_mix_with_serving_an_existing_export) {
  auto j = artifact_job({{"job", "j1"}, {"recipe", "W4A16"}, {"format", "awq"}},
                        {{"device", "gpu"}, {"vllm", Json::object()}});
  j["export"] = {{"formats", {"gptq"}}};
  j["analysis"] = {{"sensitivity", "block"}};
  j["evaluation"] = {{"skip", true}};
  auto f = fields(validate_jobspec(j));
  for (const char* want : {"/export", "/analysis", "/evaluation/skip"})
    EXPECT_NE(std::find(f.begin(), f.end(), want), f.end()) << want;
}

TEST(planner_validate_artifact, only_llm_jobs_have_exports_to_reuse) {
  auto j = artifact_job({{"job", "j1"}, {"recipe", "W4A16"}, {"format", "awq"}}, {{"vllm", Json::object()}});
  j["model"]["type"] = "vlm";
  auto f = fields(validate_jobspec(j));
  EXPECT_NE(std::find(f.begin(), f.end(), "/artifact"), f.end());
}

TEST(planner_validate_artifact, it_quantizes_nothing_so_it_has_no_quant_section_and_needs_no_calibration) {
  auto j = artifact_job({{"job", "j1"}, {"recipe", "W4A8"}, {"format", "awq"}},
                        {{"device", "gpu"}, {"sglang", Json::object()}});
  EXPECT_TRUE(validate_jobspec(j).empty()) << "no quant.recipe, no calibration: fine";
  j["quant"] = {{"recipe", {"w4a16"}},
                {"algorithm", "gptq"}};  // the wizard used to send the parent's, contradicting the artifact
  EXPECT_EQ(fields(validate_jobspec(j)), std::vector<std::string>{"/quant"})
      << "one problem, not a calibration complaint too";
}
