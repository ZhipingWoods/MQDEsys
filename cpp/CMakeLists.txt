find_package(nlohmann_json 3.11 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(CLI11 REQUIRED)
find_package(httplib REQUIRED)

# ---- probe (pure C99, vendor libraries are dlopen'ed at runtime) ----
add_library(mqde_probe STATIC probe_c/src/probe.c)
target_include_directories(mqde_probe PUBLIC probe_c/include)
target_link_libraries(mqde_probe PRIVATE ${CMAKE_DL_LIBS})

# ---- common ----
add_library(mqde_common STATIC src/common/log.cpp src/common/config.cpp)
target_include_directories(mqde_common PUBLIC include)
target_link_libraries(mqde_common PUBLIC nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp fmt::fmt)
target_compile_definitions(mqde_common PUBLIC MQDE_DEFAULT_DATA_DIR="${MQDE_DATA_DIR}"
                                              MQDE_VERSION="${PROJECT_VERSION}")

# ---- modelspec (config.json + safetensors header -> ModelSpec) ----
add_library(mqde_modelspec STATIC src/modelspec/safetensors.cpp src/modelspec/model_spec.cpp
                                  src/modelspec/model_family.cpp)
target_link_libraries(mqde_modelspec PUBLIC mqde_common)

# ---- quant (numerical core, single source of truth for quantization semantics) ----
add_library(mqde_quant STATIC src/quant/quant_spec.cpp src/quant/ops.cpp src/quant/metrics.cpp)
target_link_libraries(mqde_quant PUBLIC mqde_common)
# Bit-exactness contract: no FMA contraction / fast-math in the numerical core.
target_compile_options(mqde_quant PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off -fno-fast-math>)

# ---- estimator (memory / KV / state / concurrency) ----
add_library(mqde_estimator STATIC src/estimator/recipe.cpp src/estimator/hardware.cpp
                                  src/estimator/estimator.cpp)
target_link_libraries(mqde_estimator PUBLIC mqde_modelspec mqde_quant)

# ---- fit (llmfit-style deployment pre-evaluation) ----
add_library(mqde_fit STATIC src/fit/fit.cpp)
target_link_libraries(mqde_fit PUBLIC mqde_estimator)

# ---- server (REST + SSE, job runner) ----
add_library(mqde_server STATIC src/server/server.cpp src/server/job_runner.cpp)
target_link_libraries(mqde_server PUBLIC mqde_fit mqde_estimator mqde_probe PRIVATE httplib::httplib)

# ---- CLI ----
add_executable(mqde apps/mqde_cli/main.cpp)
target_link_libraries(mqde PRIVATE mqde_server mqde_fit mqde_estimator mqde_probe CLI11::CLI11)
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
    tests/unit/modelspec_test.cpp
    tests/unit/quant_test.cpp
    tests/unit/estimator_test.cpp
    tests/unit/probe_test.cpp
    tests/unit/fit_test.cpp)
  target_link_libraries(mqde_unit_tests PRIVATE mqde_fit mqde_estimator mqde_probe GTest::gtest_main)
  target_compile_definitions(mqde_unit_tests PRIVATE MQDE_TEST_FIXTURES="${CMAKE_SOURCE_DIR}/tests/fixtures")
  include(GoogleTest)
  gtest_discover_tests(mqde_unit_tests)
endif()
