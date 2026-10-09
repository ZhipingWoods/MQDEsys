find_package(nlohmann_json 3.11 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(CLI11 REQUIRED)
find_package(httplib REQUIRED)

# ---- probe (pure C99, vendor libraries are dlopen'ed at runtime) ----
# The HAL plugin loader (hal_loader.c) lives here too: plugin devices join the built-in ones inside mqde_probe_all().
add_library(mqde_probe STATIC probe_c/src/probe.c probe_c/src/hal_loader.c)
target_include_directories(mqde_probe PUBLIC probe_c/include include)
find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)
target_link_libraries(mqde_probe PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

# ---- hardware plugins (HAL ABI v1): one directory per vendor under plugins/, built as loadable modules ----
# Loaded at run time from $MQDE_PLUGIN_DIR, $MQDE_HOME/plugins or <executable dir>/plugins.
function(mqde_add_plugin name src)
  add_library(mqde_plugin_${name} MODULE ${src})
  target_include_directories(mqde_plugin_${name} PRIVATE include)
  set_target_properties(mqde_plugin_${name} PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME ${name}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins C_VISIBILITY_PRESET hidden)
endfunction()
if(NOT WIN32)
  mqde_add_plugin(template plugins/template/template.c)
  mqde_add_plugin(cpu plugins/cpu/cpu.c)
  mqde_add_plugin(cuda plugins/cuda/cuda.c)
  mqde_add_plugin(rocm plugins/rocm/rocm.c)
  target_link_libraries(mqde_plugin_rocm PRIVATE ${CMAKE_DL_LIBS})
endif()
if(APPLE)
  # Metal is Objective-C and macOS only: built on the Mac, absent elsewhere.
  enable_language(OBJC)
  add_library(mqde_plugin_metal MODULE plugins/metal/metal.m)
  target_include_directories(mqde_plugin_metal PRIVATE include)
  target_compile_options(mqde_plugin_metal PRIVATE -fobjc-arc)
  target_link_libraries(mqde_plugin_metal PRIVATE "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
  set_target_properties(mqde_plugin_metal PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME metal
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins)
endif()
if(NOT WIN32)
  target_link_libraries(mqde_plugin_cuda PRIVATE ${CMAKE_DL_LIBS})
endif()

# ---- common ----
add_library(mqde_common STATIC src/common/log.cpp src/common/config.cpp src/common/hash.cpp src/common/thread_pool.cpp)
target_include_directories(mqde_common PUBLIC include)
target_link_libraries(mqde_common PUBLIC nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp fmt::fmt Threads::Threads)
target_compile_definitions(mqde_common PUBLIC MQDE_DEFAULT_DATA_DIR="${MQDE_DATA_DIR}"
                                              MQDE_VERSION="${PROJECT_VERSION}")

# ---- modelspec (config.json + safetensors header -> ModelSpec) ----
add_library(mqde_modelspec STATIC src/modelspec/safetensors.cpp src/modelspec/model_spec.cpp
                                  src/modelspec/model_family.cpp src/modelspec/graph.cpp)
target_link_libraries(mqde_modelspec PUBLIC mqde_common)

# ---- quant (numerical core, single source of truth for quantization semantics) ----
add_library(mqde_quant STATIC src/quant/quant_spec.cpp src/quant/ops.cpp src/quant/metrics.cpp src/quant/pack.cpp)
target_link_libraries(mqde_quant PUBLIC mqde_common)
# Bit-exactness contract: no FMA contraction / fast-math in the numerical core.
target_compile_options(mqde_quant PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off -fno-fast-math>)

# ---- estimator (memory / KV / state / concurrency) ----
add_library(mqde_estimator STATIC src/estimator/recipe.cpp src/estimator/hardware.cpp
                                  src/estimator/estimator.cpp)
target_link_libraries(mqde_estimator PUBLIC mqde_modelspec mqde_quant)

# ---- fit (llmfit-style deployment pre-evaluation) ----
add_library(mqde_fit STATIC src/fit/fit.cpp src/fit/benchmarks.cpp src/fit/concurrency.cpp src/fit/claim.cpp
                             src/fit/installed.cpp)
target_link_libraries(mqde_fit PUBLIC mqde_estimator PRIVATE httplib::httplib)

# ---- storage: SQLite metadata + content-addressed cache ----
add_library(mqde_storage STATIC src/storage/db.cpp src/storage/cas.cpp src/storage/calibration.cpp)
if(TARGET SQLite3::SQLite3)
  target_link_libraries(mqde_storage PUBLIC mqde_common SQLite3::SQLite3)
else()
  target_link_libraries(mqde_storage PUBLIC mqde_common SQLite::SQLite3)
endif()

# ---- hub (Hugging Face client + model catalog) ----
add_library(mqde_hub STATIC src/hub/hf.cpp src/hub/catalog.cpp)
target_link_libraries(mqde_hub PUBLIC mqde_modelspec PRIVATE httplib::httplib)

# ---- server (REST + SSE, job runner) ----
add_library(mqde_server STATIC src/server/server.cpp src/server/job_runner.cpp)
target_link_libraries(mqde_server PUBLIC mqde_fit mqde_hub mqde_estimator mqde_probe PRIVATE httplib::httplib)

# ---- CLI ----
add_executable(mqde apps/mqde_cli/main.cpp)
target_link_libraries(mqde PRIVATE mqde_server mqde_fit mqde_hub mqde_estimator mqde_probe CLI11::CLI11)
set_target_properties(mqde PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Python extension mqde._core ----
if(MQDE_WITH_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core bindings/python/core_module.cpp)
  target_link_libraries(_core PRIVATE mqde_estimator mqde_probe)
  target_compile_definitions(_core PRIVATE MQDE_VERSION_INFO="${PROJECT_VERSION}")
  if(SKBUILD)
    install(TARGETS _core DESTINATION mqde)
    install(TARGETS mqde DESTINATION mqde/bin)
  else()
    # Dev build: drop the extension next to the Python sources for `PYTHONPATH=python`.
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/mqde)
  endif()
endif()

if(MQDE_BUILD_TESTS)
  find_package(GTest REQUIRED)
  add_executable(mqde_unit_tests
    tests/unit/common_test.cpp
    tests/unit/common_registry_test.cpp
    tests/unit/common_hash_test.cpp
    tests/unit/common_thread_pool_test.cpp
    tests/unit/storage_db_test.cpp
    tests/unit/storage_cas_test.cpp
    tests/unit/storage_calibration_test.cpp
    tests/unit/estimator_perf_test.cpp
    tests/unit/modelspec_test.cpp
    tests/unit/quant_test.cpp
    tests/unit/pack_test.cpp
    tests/unit/estimator_test.cpp
    tests/unit/probe_test.cpp
    tests/unit/fit_test.cpp
    tests/unit/benchmarks_test.cpp
    tests/unit/concurrency_test.cpp
    tests/unit/installed_test.cpp
    tests/unit/hub_test.cpp
    tests/unit/hub_remote_test.cpp
    tests/unit/hal_test.cpp)
  add_library(mqde_registry_probe STATIC tests/registry_probe/shapes.cpp)
  target_include_directories(mqde_registry_probe PUBLIC tests/registry_probe)
  target_link_libraries(mqde_registry_probe PUBLIC mqde_common)
  target_link_libraries(mqde_unit_tests PRIVATE mqde_storage mqde_registry_probe mqde_fit mqde_hub mqde_estimator mqde_probe httplib::httplib GTest::gtest_main)
  target_compile_definitions(mqde_unit_tests PRIVATE MQDE_TEST_FIXTURES="${CMAKE_SOURCE_DIR}/tests/fixtures")
  # Deliberately broken plugins for the loader tests, one directory per scenario group (see hal_test.cpp).
  if(NOT WIN32)
    set(MQDE_HAL_CASES good dup_a dup_b bad_abi small_struct no_symbol returns_null null_probe null_last_error bad_name
        probe_fails too_many negative_n zero_devices)
    set(MQDE_HAL_CASE_DIR ${CMAKE_BINARY_DIR}/test_plugins)
    foreach(c ${MQDE_HAL_CASES})
      add_library(hal_case_${c} MODULE tests/plugins/${c}.c)
      target_include_directories(hal_case_${c} PRIVATE include)
      set_target_properties(hal_case_${c} PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME ${c}
        LIBRARY_OUTPUT_DIRECTORY ${MQDE_HAL_CASE_DIR} C_VISIBILITY_PRESET hidden)
      add_dependencies(mqde_unit_tests hal_case_${c})
    endforeach()
    add_dependencies(mqde_unit_tests mqde_plugin_template)
    # Stand-in vendor runtimes (see tests/fakes): lets the vendor plugins run their real code without the hardware.
    add_library(fake_hip MODULE tests/fakes/fake_hip.c)
    set_target_properties(fake_hip PROPERTIES PREFIX "" SUFFIX ".so" OUTPUT_NAME libamdhip64
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fake_runtimes C_VISIBILITY_PRESET default)
    add_dependencies(mqde_unit_tests fake_hip)
    target_compile_definitions(mqde_unit_tests PRIVATE MQDE_FAKE_RUNTIME_DIR="${CMAKE_BINARY_DIR}/fake_runtimes")
    target_compile_definitions(mqde_unit_tests PRIVATE MQDE_HAL_CASE_DIR="${MQDE_HAL_CASE_DIR}"
      MQDE_TEMPLATE_PLUGIN_DIR="${CMAKE_BINARY_DIR}/bin/plugins")
  endif()
  include(GoogleTest)
  gtest_discover_tests(mqde_unit_tests)
endif()

