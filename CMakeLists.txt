cmake_minimum_required(VERSION 3.25)
project(mqde VERSION 0.1.0 LANGUAGES C CXX)

option(MQDE_BUILD_TESTS "Build unit tests" ON)
option(MQDE_WITH_PYTHON "Build the mqde._core Python extension" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wall> $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra> $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
                      $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>)  # not for nvcc: its -Werror takes a value
endif()

set(MQDE_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Default data directory")

if(MQDE_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(cpp)
