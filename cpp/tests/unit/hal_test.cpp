// T-1101: HAL plugin ABI, loader and probe integration. Every scenario is a real shared library built from
// cpp/tests/plugins/<case>.c (or the shipped template plugin) and loaded with dlopen: ABI mismatch, missing
// symbol, NULL mandatory functions, a probe that fails or lies about its count, duplicate names, and the
// allow / deny lists.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include <gtest/gtest.h>

#include "mqde/hal/mqde_hal.h"
#include "mqde_probe.h"

#ifndef _WIN32
namespace {

struct Env {
  Env(const char* k, const char* v) : key(k) {
    const char* old = ::getenv(k);
    if (old) saved = old, had = true;
    if (v)
      ::setenv(k, v, 1);
    else
      ::unsetenv(k);
  }
  ~Env() {
    if (had)
      ::setenv(key.c_str(), saved.c_str(), 1);
    else
      ::unsetenv(key.c_str());
  }
  std::string key, saved;
  bool had = false;
};

// Plugins are listed in file-name order, so tests look one up by name, never by position.
mqde_hal_plugin_info plugin_info(const char* name) {
  for (int i = 0; i < mqde_hal_plugin_count(); ++i) {
    mqde_hal_plugin_info p;
    if (mqde_hal_plugin_get(i, &p) == 0 && std::string(p.name) == name) return p;
  }
  ADD_FAILURE() << "plugin " << name << " not in the registry";
  return {};
}

std::map<std::string, mqde_hal_plugin_info> load_cases() {
  EXPECT_GT(mqde_hal_load_all(), 0);
  std::map<std::string, mqde_hal_plugin_info> m;
  for (int i = 0; i < mqde_hal_plugin_count(); ++i) {
    mqde_hal_plugin_info p;
    EXPECT_EQ(mqde_hal_plugin_get(i, &p), 0);
    std::string file = p.path;
    file = file.substr(file.rfind('/') + 1);
    m[file.substr(0, file.size() - 3)] = p;  // keyed by the library's file name: good, bad_abi, ...
  }
  return m;
}

}  // namespace

class hal_cases : public ::testing::Test {
 protected:
  Env dir{"MQDE_PLUGIN_DIR", MQDE_HAL_CASE_DIR};
  Env only{"MQDE_HAL_PLUGINS", nullptr};
  Env off{"MQDE_HAL_DISABLE", nullptr};
  Env home{"MQDE_HOME", nullptr};
};

TEST_F(hal_cases, a_good_plugin_loads_and_reports_its_device) {
  auto m = load_cases();
  ASSERT_TRUE(m.count("good"));
  EXPECT_EQ(m["good"].status, MQDE_HAL_LOADED);
  EXPECT_STREQ(m["good"].name, "good");
  EXPECT_EQ(m["good"].devices, 1);
  EXPECT_STREQ(m["good"].reason, "");
  const auto* api = mqde_hal_plugin_by_name("good");
  ASSERT_NE(api, nullptr);
  mqde_hal_device d[4];
  int n = 0;
  ASSERT_EQ(api->device_probe(d, 4, &n), 0);
  ASSERT_EQ(n, 1);
  EXPECT_STREQ(d[0].name, "Acme One");
}

TEST_F(hal_cases, each_kind_of_broken_plugin_is_rejected_with_a_reason_naming_the_problem) {
  auto m = load_cases();
  struct Want {
    const char* file;
    const char* reason;
  };
  for (const Want& w :
       {Want{"bad_abi", "ABI version 2"}, Want{"small_struct", "struct_size 24"},
        Want{"no_symbol", "mqde_hal_get_plugin_v1"}, Want{"returns_null", "returned NULL"},
        Want{"null_probe", "device_probe"}, Want{"null_last_error", "last_error"},
        Want{"bad_name", "invalid plugin name"}, Want{"probe_fails", "device_probe failed: boom"},
        Want{"too_many", "1000 devices"}, Want{"negative_n", "-3 devices"}}) {
    ASSERT_TRUE(m.count(w.file)) << w.file;
    EXPECT_EQ(m[w.file].status, MQDE_HAL_REJECTED) << w.file;
    EXPECT_NE(std::string(m[w.file].reason).find(w.reason), std::string::npos)
        << w.file << ": " << m[w.file].reason;
    EXPECT_EQ(mqde_hal_plugin_by_name(m[w.file].name), nullptr) << w.file << " must not be usable";
  }
}

TEST_F(hal_cases, a_broken_plugin_does_not_stop_the_good_ones) {
  auto m = load_cases();
  EXPECT_EQ(m["good"].status, MQDE_HAL_LOADED);
  int rejected = 0, loaded = 0;
  for (auto& [k, p] : m) {
    rejected += p.status == MQDE_HAL_REJECTED;
    loaded += p.status == MQDE_HAL_LOADED;
  }
  EXPECT_GE(rejected, 10);
  EXPECT_GE(loaded, 2);  // good, plus the first of the duplicates
}

TEST_F(hal_cases, zero_devices_is_inactive_not_an_error) {
  auto m = load_cases();
  EXPECT_EQ(m["zero_devices"].status, MQDE_HAL_INACTIVE);
  EXPECT_EQ(m["zero_devices"].devices, 0);
  EXPECT_STREQ(m["zero_devices"].reason, "");
  EXPECT_NE(mqde_hal_plugin_by_name("zero_devices"), nullptr) << "still loaded: it may find devices later";
}

TEST_F(hal_cases, duplicate_names_keep_the_first_in_file_order_and_say_why) {
  auto m = load_cases();
  EXPECT_EQ(m["dup_a"].status, MQDE_HAL_LOADED);
  EXPECT_EQ(m["dup_b"].status, MQDE_HAL_REJECTED);
  EXPECT_NE(std::string(m["dup_b"].reason).find("duplicate plugin name"), std::string::npos)
      << m["dup_b"].reason;
  EXPECT_NE(std::string(m["dup_b"].reason).find("dup_a.so"), std::string::npos) << "names the file that won";
}

TEST_F(hal_cases, the_allow_list_loads_only_the_named_plugins) {
  Env only2("MQDE_HAL_PLUGINS", "good");
  auto m = load_cases();
  EXPECT_EQ(m["good"].status, MQDE_HAL_LOADED);
  EXPECT_EQ(m["zero_devices"].status, MQDE_HAL_DISABLED);
  EXPECT_NE(std::string(m["zero_devices"].reason).find("MQDE_HAL_PLUGINS"), std::string::npos);
  EXPECT_EQ(m["bad_abi"].status, MQDE_HAL_REJECTED)
      << "a plugin that is broken is reported as broken, not as 'disabled'";
}

TEST_F(hal_cases, the_deny_list_skips_the_named_plugins) {
  Env off2("MQDE_HAL_DISABLE", "good,dup");
  auto m = load_cases();
  EXPECT_EQ(m["good"].status, MQDE_HAL_DISABLED);
  EXPECT_EQ(m["dup_a"].status, MQDE_HAL_DISABLED);
  EXPECT_EQ(m["zero_devices"].status, MQDE_HAL_INACTIVE);
  EXPECT_EQ(mqde_hal_plugin_by_name("good"), nullptr);
  Env off3("MQDE_HAL_DISABLE", "goo");  // a prefix is not a match
  EXPECT_EQ(load_cases()["good"].status, MQDE_HAL_LOADED);
}

TEST_F(hal_cases, loading_again_rescans_and_a_missing_directory_is_not_an_error) {
  EXPECT_GT(load_cases().size(), 10u);
  Env gone("MQDE_PLUGIN_DIR", "/definitely/not/a/directory");
  Env noexe("MQDE_HOME", "/also/not/there");
  EXPECT_EQ(mqde_hal_load_all(),
            0);  // (the executable's own plugins/ directory does not exist next to the test binary)
  EXPECT_EQ(mqde_hal_plugin_count(), 0);
  mqde_hal_plugin_info p;
  EXPECT_NE(mqde_hal_plugin_get(0, &p), 0);
  EXPECT_NE(mqde_hal_plugin_get(-1, &p), 0);
  EXPECT_EQ(mqde_hal_plugin_by_name("good"), nullptr);
  EXPECT_EQ(mqde_hal_plugin_by_name(nullptr), nullptr);
}

TEST_F(hal_cases, devices_of_loaded_plugins_are_listed_with_their_plugin_names) {
  load_cases();
  mqde_hal_device d[16];
  const char* names[16];
  const int n = mqde_hal_devices(d, 16, names);
  ASSERT_GE(n, 1);
  bool found = false;
  for (int i = 0; i < n; ++i)
    if (std::string(names[i]) == "good") found = std::string(d[i].name) == "Acme One";
  EXPECT_TRUE(found);
  EXPECT_EQ(mqde_hal_devices(d, 0, nullptr), 0) << "capacity zero writes nothing";
}

// ---- the shipped template plugin, through the real probe
// ----------------------------------------------------------

class hal_template : public ::testing::Test {
 protected:
  Env dir{"MQDE_PLUGIN_DIR", MQDE_TEMPLATE_PLUGIN_DIR};
  Env only{"MQDE_HAL_PLUGINS", nullptr};
  Env off{"MQDE_HAL_DISABLE", nullptr};
};

TEST_F(hal_template, is_inactive_by_default_so_it_never_fakes_hardware_in_a_normal_probe) {
  Env none("MQDE_TEMPLATE_DEVICES", nullptr);
  ASSERT_GE(mqde_hal_load_all(), 1);
  mqde_hal_plugin_info p = plugin_info("template");
  EXPECT_STREQ(p.name, "template");
  EXPECT_EQ(p.status, MQDE_HAL_INACTIVE);
  mqde_device_info devs[32];
  int n = 0;
  ASSERT_EQ(mqde_probe_all(devs, 32, &n), 0);
  for (int i = 0; i < n; ++i) EXPECT_STRNE(devs[i].vendor, "template");
}

TEST_F(hal_template, its_devices_appear_in_mqde_probe_all_without_any_change_above_the_loader) {
  Env two("MQDE_TEMPLATE_DEVICES", "2");
  ASSERT_GE(mqde_hal_load_all(), 1);
  mqde_device_info devs[32];
  int n = 0;
  ASSERT_EQ(mqde_probe_all(devs, 32, &n), 0);
  int found = 0;
  for (int i = 0; i < n; ++i)
    if (std::string(devs[i].vendor) == "template") {
      EXPECT_EQ(devs[i].kind, MQDE_DEV_NPU);
      EXPECT_EQ(devs[i].index, found);
      EXPECT_EQ(devs[i].mem_total_bytes, 16ull << 30);
      EXPECT_STREQ(devs[i].arch, "tmpl1");
      EXPECT_EQ(std::string(devs[i].name), "MQDE Template Accelerator " + std::to_string(found));
      ++found;
    }
  EXPECT_EQ(found, 2);
  EXPECT_EQ(devs[0].kind, MQDE_DEV_CPU) << "built-in devices still come first";
}

TEST_F(hal_template, a_device_the_builtin_probe_already_found_is_not_listed_twice) {
  Env one("MQDE_TEMPLATE_DEVICES", "1");
  ASSERT_GE(mqde_hal_load_all(), 1);
  mqde_device_info a[32], b[32];
  int na = 0, nb = 0;
  ASSERT_EQ(mqde_probe_all(a, 32, &na), 0);
  ASSERT_EQ(mqde_probe_all(b, 32, &nb), 0);
  EXPECT_EQ(na, nb) << "probing twice does not accumulate devices";
}

TEST_F(hal_template, capabilities_memory_and_profile_work_and_bad_requests_say_what_is_wrong) {
  Env two("MQDE_TEMPLATE_DEVICES", "2");
  ASSERT_GE(mqde_hal_load_all(), 1);
  const auto* api = mqde_hal_plugin_by_name("template");
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->compile, nullptr) << "optional functions may be NULL in v1";
  mqde_hal_capability_query q{};
  q.device = 1;
  std::strcpy(q.dtype, "int8");
  mqde_hal_capability_result r{};
  ASSERT_EQ(api->supports(&q, &r), 0);
  EXPECT_EQ(r.supported, 1);
  EXPECT_EQ(r.native, 1);
  EXPECT_DOUBLE_EQ(r.peak_tflops, 200.0);
  std::strcpy(q.dtype, "fp8_e4m3");
  ASSERT_EQ(api->supports(&q, &r), 0);
  EXPECT_EQ(r.supported, 1);
  EXPECT_EQ(r.native, 0) << "emulated, not native";
  std::strcpy(q.dtype, "float128");
  ASSERT_EQ(api->supports(&q, &r), 0);
  EXPECT_EQ(r.supported, 0);
  q.device = 5;
  EXPECT_NE(api->supports(&q, &r), 0);
  EXPECT_STREQ(api->last_error(), "no such device");

  void* p = nullptr;
  ASSERT_EQ(api->alloc(0, 1 << 20, &p), 0);
  ASSERT_NE(p, nullptr);
  std::memset(p, 0xAB, 1 << 20);
  EXPECT_EQ(api->free(0, p), 0);

  mqde_hal_profile_request req{};
  req.device = 0;
  std::strcpy(req.kind, "bandwidth");
  mqde_hal_profile_result res{};
  ASSERT_EQ(api->profile(0, &req, &res), 0);
  EXPECT_GT(res.value, 0.0);
  EXPECT_STREQ(res.unit, "GB/s");
  std::strcpy(req.kind, "teleport");
  EXPECT_NE(api->profile(0, &req, &res), 0);
  EXPECT_NE(std::string(api->last_error()).find("unknown profile kind"), std::string::npos);
}

// ---- ABI layout: the numbers scripts/check_abi.sh compares against abi_v1.golden
// ----------------------------------

TEST(hal_abi, version_and_struct_sizes_are_the_published_ones) {
  EXPECT_EQ(MQDE_HAL_ABI_VERSION, 1);
  // 4+4 (kind, index) + 32 + 128 + 32 strings = 200, + 8+8 memory sizes = 216, + 4+4 = 224, + 64 driver
  // string
  EXPECT_EQ(sizeof(mqde_hal_device), 288u);
  EXPECT_EQ(offsetof(mqde_hal_plugin_v1, device_probe), 32u);
}
#endif

// ---- the shipped CPU plugin: a real device and real micro-benchmarks, on any machine
// --------------------------------

#ifndef _WIN32
class hal_cpu : public ::testing::Test {
 protected:
  Env dir{"MQDE_PLUGIN_DIR", MQDE_TEMPLATE_PLUGIN_DIR};
  Env only{"MQDE_HAL_PLUGINS", "cpu"};
  Env off{"MQDE_HAL_DISABLE", nullptr};
};

TEST_F(hal_cpu, is_the_same_device_the_builtin_probe_found_so_the_cpu_is_listed_once) {
  ASSERT_GE(mqde_hal_load_all(), 1);
  mqde_device_info devs[32];
  int n = 0;
  ASSERT_EQ(mqde_probe_all(devs, 32, &n), 0);
  int cpus = 0;
  for (int i = 0; i < n; ++i) cpus += devs[i].kind == MQDE_DEV_CPU;
  EXPECT_EQ(cpus, 1) << "the plugin's CPU matches the built-in one by vendor, name and index";
}

TEST_F(hal_cpu, bandwidth_and_gemm_return_plausible_positive_numbers_with_units) {
  ASSERT_GE(mqde_hal_load_all(), 1);
  const auto* api = mqde_hal_plugin_by_name("cpu");
  ASSERT_NE(api, nullptr);
  mqde_hal_profile_request req{};
  req.device = 0;
  std::strcpy(req.kind, "bandwidth");
  req.bytes = 64u << 20;
  req.iterations = 4;
  mqde_hal_profile_result res{};
  ASSERT_EQ(api->profile(0, &req, &res), 0) << api->last_error();
  EXPECT_STREQ(res.unit, "GB/s");
  EXPECT_GT(res.value, 0.5) << "a memcpy slower than 0.5 GB/s would mean the timing is broken";
  EXPECT_LT(res.value, 5000.0) << "faster than 5 TB/s on a CPU would mean the copy was optimised away";
  EXPECT_GT(res.seconds, 0.0);

  std::strcpy(req.kind, "gemm");
  std::strcpy(req.dtype, "fp32");
  req.m = req.n = req.k = 256;
  req.iterations = 2;
  ASSERT_EQ(api->profile(0, &req, &res), 0) << api->last_error();
  EXPECT_STREQ(res.unit, "TFLOPS");
  EXPECT_GT(res.value, 1e-5);
  EXPECT_LT(res.value, 50.0);
  // the number is derived from the work and the time: 2*m*n*k flops over the reported seconds
  EXPECT_NEAR(res.value, 2.0 * 256 * 256 * 256 / res.seconds / 1e12, res.value * 1e-6);
}

TEST_F(hal_cpu, a_bigger_gemm_does_not_report_a_wildly_different_rate_but_takes_longer) {
  ASSERT_GE(mqde_hal_load_all(), 1);
  const auto* api = mqde_hal_plugin_by_name("cpu");
  ASSERT_NE(api, nullptr);
  mqde_hal_profile_request small{}, big{};
  std::strcpy(small.kind, "gemm"), std::strcpy(small.dtype, "fp32");
  big = small;
  small.m = small.n = small.k = 128;
  big.m = big.n = big.k = 384;
  small.iterations = big.iterations = 2;
  mqde_hal_profile_result a{}, b{};
  ASSERT_EQ(api->profile(0, &small, &a), 0);
  ASSERT_EQ(api->profile(0, &big, &b), 0);
  EXPECT_GT(b.seconds, a.seconds) << "27 times the flops must take longer";
}

TEST_F(hal_cpu, capability_answers_and_bad_requests) {
  ASSERT_GE(mqde_hal_load_all(), 1);
  const auto* api = mqde_hal_plugin_by_name("cpu");
  ASSERT_NE(api, nullptr);
  mqde_hal_capability_query q{};
  mqde_hal_capability_result r{};
  std::strcpy(q.dtype, "fp32");
  ASSERT_EQ(api->supports(&q, &r), 0);
  EXPECT_EQ(r.supported, 1);
  EXPECT_EQ(r.native, 1);
  std::strcpy(q.dtype, "fp8_e4m3");
  ASSERT_EQ(api->supports(&q, &r), 0);
  EXPECT_EQ(r.supported, 1);
  EXPECT_EQ(r.native, 0) << "emulated";
  std::strcpy(q.dtype, "float128");
  ASSERT_EQ(api->supports(&q, &r), 0);
  EXPECT_EQ(r.supported, 0);
  q.device = 3;
  EXPECT_NE(api->supports(&q, &r), 0);
  mqde_hal_profile_request req{};
  std::strcpy(req.kind, "gemm");
  std::strcpy(req.dtype, "fp16");
  mqde_hal_profile_result res{};
  EXPECT_NE(api->profile(0, &req, &res), 0);
  EXPECT_NE(std::string(api->last_error()).find("fp32 only"), std::string::npos);
  std::strcpy(req.kind, "teleport");
  EXPECT_NE(api->profile(0, &req, &res), 0);
}
#endif

// ---- vendor plugins against a stand-in vendor runtime
// --------------------------------------------------------------- The ROCm plugin's real code runs: dlopen of
// the runtime, symbol lookup, attribute ids, memory info, timed copies and the error paths. Only the runtime
// behind it is fake (cpp/tests/fakes/fake_hip.c), so these tests prove the plugin's logic but cannot prove
// that real HIP behaves the same way.
#ifndef _WIN32
class hal_rocm : public ::testing::Test {
 protected:
  Env dir{"MQDE_PLUGIN_DIR", MQDE_TEMPLATE_PLUGIN_DIR};
  Env only{"MQDE_HAL_PLUGINS", "rocm"};
  Env off{"MQDE_HAL_DISABLE", nullptr};
  Env lib{"MQDE_ROCM_LIB_DIR", MQDE_FAKE_RUNTIME_DIR};
  Env fdev{"FAKE_HIP_DEVICES", nullptr};
  Env fmaj{"FAKE_HIP_MAJOR", nullptr};
  Env ffail{"FAKE_HIP_FAIL", nullptr};
  Env fmem{"FAKE_HIP_MEM_MB", nullptr};
};

// The plugin keeps its resolved runtime for the life of the process, so every scenario that changes the
// fake's environment runs in a child process (EXPECT_EXIT) and the parent stays clean.
#define IN_CHILD(body)                                  \
  EXPECT_EXIT(                                          \
      {                                                 \
        body;                                           \
        std::exit(testing::Test::HasFailure() ? 1 : 0); \
      },                                                \
      ::testing::ExitedWithCode(0), "")

TEST_F(hal_rocm, lists_the_fake_devices_with_the_attributes_the_plugin_reads) {
  IN_CHILD({
    Env n("FAKE_HIP_DEVICES", "3");
    ASSERT_GE(mqde_hal_load_all(), 1);
    mqde_hal_plugin_info p = plugin_info("rocm");
    EXPECT_EQ(p.status, MQDE_HAL_LOADED);
    EXPECT_EQ(p.devices, 3);
    mqde_device_info devs[32];
    int cnt = 0;
    ASSERT_EQ(mqde_probe_all(devs, 32, &cnt), 0);
    int found = 0;
    for (int i = 0; i < cnt; ++i)
      if (std::string(devs[i].vendor) == "amd") {
        EXPECT_EQ(devs[i].kind, MQDE_DEV_GPU);
        EXPECT_EQ(devs[i].index, found);
        EXPECT_EQ(std::string(devs[i].name), "Fake AMD GPU " + std::to_string(found));
        EXPECT_EQ(devs[i].compute_units, 96u) << "the CU count comes from attribute id 63";
        EXPECT_STREQ(devs[i].arch, "hip11.4") << "major from attribute 23, minor from 61";
        EXPECT_EQ(devs[i].mem_total_bytes, 8192ull << 20);
        ++found;
      }
    EXPECT_EQ(found, 3);
  });
}

TEST_F(hal_rocm, without_the_runtime_it_is_inactive_not_an_error) {
  IN_CHILD({
    Env nowhere("MQDE_ROCM_LIB_DIR", "/nonexistent");
    ASSERT_GE(mqde_hal_load_all(), 1);
    mqde_hal_plugin_info p = plugin_info("rocm");
    EXPECT_EQ(p.status, MQDE_HAL_INACTIVE) << p.reason;
    EXPECT_EQ(p.devices, 0);
  });
}

TEST_F(hal_rocm, a_runtime_with_no_gpu_or_a_failing_init_gives_zero_devices) {
  IN_CHILD({
    Env fail("FAKE_HIP_FAIL", "hipInit");
    ASSERT_GE(mqde_hal_load_all(), 1);
    mqde_hal_plugin_info p = plugin_info("rocm");
    EXPECT_EQ(p.status, MQDE_HAL_INACTIVE);
  });
  IN_CHILD({
    Env zero("FAKE_HIP_DEVICES", "0");
    ASSERT_GE(mqde_hal_load_all(), 1);
    mqde_hal_plugin_info p = plugin_info("rocm");
    EXPECT_EQ(p.status, MQDE_HAL_INACTIVE);
  });
}

TEST_F(hal_rocm, capabilities_follow_the_architecture_generation) {
  struct Case {
    const char* major;
    const char* dtype;
    int supported;
    int native;
  };
  for (const Case& c : {Case{"9", "fp8_e4m3", 1, 1}, Case{"9", "fp16", 1, 1}, Case{"11", "fp8_e4m3", 0, 0},
                        Case{"11", "bf16", 1, 1}, Case{"10", "fp16", 1, 0}, Case{"10", "int8", 1, 0},
                        Case{"11", "int4", 0, 0}, Case{"11", "fp32", 1, 1}}) {
    IN_CHILD({
      Env m("FAKE_HIP_MAJOR", c.major);
      ASSERT_GE(mqde_hal_load_all(), 1);
      const auto* api = mqde_hal_plugin_by_name("rocm");
      ASSERT_NE(api, nullptr);
      mqde_hal_capability_query q{};
      std::strcpy(q.dtype, c.dtype);
      mqde_hal_capability_result r{};
      ASSERT_EQ(api->supports(&q, &r), 0);
      EXPECT_EQ(r.supported, c.supported) << "gfx major " << c.major << " " << c.dtype;
      EXPECT_EQ(r.native, c.native) << "gfx major " << c.major << " " << c.dtype;
    });
  }
}

TEST_F(hal_rocm, bandwidth_runs_the_real_copy_loop_and_reports_what_it_measured) {
  IN_CHILD({
    ASSERT_GE(mqde_hal_load_all(), 1);
    const auto* api = mqde_hal_plugin_by_name("rocm");
    ASSERT_NE(api, nullptr);
    mqde_hal_profile_request req{};
    std::strcpy(req.kind, "bandwidth");
    req.bytes = 64u << 20;
    req.iterations = 5;
    mqde_hal_profile_result res{};
    ASSERT_EQ(api->profile(0, &req, &res), 0) << api->last_error();
    EXPECT_STREQ(res.unit, "GB/s");
    EXPECT_GT(res.seconds, 0.0);
    // 5 copies of 64 MiB, read plus write counted: the value must equal that work over the reported time
    EXPECT_NEAR(res.value, 2.0 * (64u << 20) * 5 / res.seconds / 1e9, res.value * 1e-6);
    EXPECT_GT(res.value, 0.5);
  });
}

TEST_F(hal_rocm, runtime_errors_surface_with_their_message_and_do_not_leak_or_crash) {
  for (const char* fn : {"hipMalloc", "hipMemcpyAsync", "hipSetDevice"}) {
    IN_CHILD({
      Env f("FAKE_HIP_FAIL", fn);
      ASSERT_GE(mqde_hal_load_all(), 0);
      const auto* api = mqde_hal_plugin_by_name("rocm");
      ASSERT_NE(api, nullptr);
      mqde_hal_profile_request req{};
      std::strcpy(req.kind, "bandwidth");
      req.bytes = 8u << 20;
      mqde_hal_profile_result res{};
      EXPECT_NE(api->profile(0, &req, &res), 0) << fn;
      EXPECT_NE(std::string(api->last_error()).find("fake)"), std::string::npos)
          << fn << ": " << api->last_error();
    });
  }
}

TEST_F(hal_rocm, gemm_says_it_is_not_implemented_instead_of_guessing) {
  IN_CHILD({
    ASSERT_GE(mqde_hal_load_all(), 1);
    const auto* api = mqde_hal_plugin_by_name("rocm");
    ASSERT_NE(api, nullptr);
    mqde_hal_profile_request req{};
    std::strcpy(req.kind, "gemm");
    std::strcpy(req.dtype, "fp16");
    mqde_hal_profile_result res{};
    EXPECT_NE(api->profile(0, &req, &res), 0);
    EXPECT_NE(std::string(api->last_error()).find("not implemented"), std::string::npos);
    std::strcpy(req.kind, "nonsense");
    EXPECT_NE(api->profile(0, &req, &res), 0);
    req.device = 0;
    EXPECT_NE(api->profile(7, &req, &res), 0);  // no such device
  });
}

TEST_F(hal_rocm, alloc_and_free_go_through_the_runtime) {
  IN_CHILD({
    ASSERT_GE(mqde_hal_load_all(), 1);
    const auto* api = mqde_hal_plugin_by_name("rocm");
    ASSERT_NE(api, nullptr);
    void* p = nullptr;
    ASSERT_EQ(api->alloc(0, 4096, &p), 0) << api->last_error();
    ASSERT_NE(p, nullptr);
    std::memset(p, 7, 4096);
    EXPECT_EQ(api->free(0, p), 0);
    EXPECT_NE(api->alloc(9, 16, &p), 0) << "a device that does not exist";
    EXPECT_NE(api->alloc(0, 16, nullptr), 0);
  });
}
#endif

// ---- Metal plugin on a real Apple GPU (skipped where there is none)
// --------------------------------------------------
#ifdef __APPLE__
class hal_metal : public ::testing::Test {
 protected:
  Env dir{"MQDE_PLUGIN_DIR", MQDE_TEMPLATE_PLUGIN_DIR};
  Env only{"MQDE_HAL_PLUGINS", "metal"};
  Env off{"MQDE_HAL_DISABLE", nullptr};
  const mqde_hal_plugin_v1* api = nullptr;
  void SetUp() override {
    ASSERT_GE(mqde_hal_load_all(), 1);
    if (plugin_info("metal").devices == 0) GTEST_SKIP() << "no Metal device on this machine";
    api = mqde_hal_plugin_by_name("metal");
    ASSERT_NE(api, nullptr);
  }
};

TEST_F(hal_metal, lists_the_gpu_with_unified_memory_figures_and_it_is_not_duplicated) {
  mqde_device_info devs[32];
  int n = 0;
  ASSERT_EQ(mqde_probe_all(devs, 32, &n), 0);
  int gpus = 0;
  for (int i = 0; i < n; ++i)
    if (std::string(devs[i].vendor) == "apple" && devs[i].kind == MQDE_DEV_GPU) {
      ++gpus;
      EXPECT_GT(devs[i].mem_total_bytes, 1ull << 30);
      EXPECT_LE(devs[i].mem_free_bytes, devs[i].mem_total_bytes);
      EXPECT_NE(std::string(devs[i].name).find("Apple"), std::string::npos);
    }
  EXPECT_GE(gpus, 1);
}

TEST_F(hal_metal, bandwidth_is_in_the_range_of_unified_memory_and_is_stable_between_runs) {
  mqde_hal_profile_request req{};
  std::strcpy(req.kind, "bandwidth");
  req.bytes = 128u << 20;
  req.iterations = 8;
  double v[3];
  for (double& x : v) {
    mqde_hal_profile_result r{};
    ASSERT_EQ(api->profile(0, &req, &r), 0) << api->last_error();
    EXPECT_STREQ(r.unit, "GB/s");
    EXPECT_NEAR(r.value, 2.0 * (128u << 20) * 8 / r.seconds / 1e9, r.value * 1e-6);
    x = r.value;
  }
  for (double x : v) {
    EXPECT_GT(x, 20.0) << "an Apple GPU copy slower than 20 GB/s would be a broken measurement";
    EXPECT_LT(x, 1500.0)
        << "faster than any Apple unified memory (800 GB/s on the largest chips) means the copy was skipped";
  }
  EXPECT_LT(*std::max_element(v, v + 3) / *std::min_element(v, v + 3), 1.6)
      << "three runs should agree within 60%";
}

TEST_F(hal_metal, gemm_rate_is_derived_from_the_work_and_the_time_and_fp16_is_not_slower_than_fp32) {
  mqde_hal_profile_request req{};
  std::strcpy(req.kind, "gemm");
  req.m = req.n = req.k = 1536;
  req.iterations = 6;
  mqde_hal_profile_result f32{}, f16{};
  std::strcpy(req.dtype, "fp32");
  ASSERT_EQ(api->profile(0, &req, &f32), 0) << api->last_error();
  std::strcpy(req.dtype, "fp16");
  ASSERT_EQ(api->profile(0, &req, &f16), 0) << api->last_error();
  EXPECT_STREQ(f32.unit, "TFLOPS");
  const double work = 2.0 * 1536 * 1536 * 1536 * 6;
  EXPECT_NEAR(f32.value, work / f32.seconds / 1e12, f32.value * 1e-6);
  EXPECT_GT(f32.value, 0.05);
  EXPECT_LT(f32.value, 100.0);
  EXPECT_GE(f16.value, f32.value * 0.7) << "half precision should not be much slower than single";
}

TEST_F(hal_metal, errors_name_the_problem) {
  mqde_hal_profile_request req{};
  mqde_hal_profile_result r{};
  std::strcpy(req.kind, "gemm");
  std::strcpy(req.dtype, "int8");
  EXPECT_NE(api->profile(0, &req, &r), 0);
  EXPECT_NE(std::string(api->last_error()).find("fp32, fp16"), std::string::npos);
  std::strcpy(req.kind, "bandwidth");
  EXPECT_NE(api->profile(5, &req, &r), 0);
  EXPECT_STREQ(api->last_error(), "no such device");
  std::strcpy(req.kind, "teleport");
  EXPECT_NE(api->profile(0, &req, &r), 0);
  mqde_hal_capability_query q{};
  mqde_hal_capability_result c{};
  std::strcpy(q.dtype, "fp16");
  ASSERT_EQ(api->supports(&q, &c), 0);
  EXPECT_EQ(c.supported, 1);
  std::strcpy(q.dtype, "fp8_e4m3");
  ASSERT_EQ(api->supports(&q, &c), 0);
  EXPECT_EQ(c.supported, 0);
}
#endif
