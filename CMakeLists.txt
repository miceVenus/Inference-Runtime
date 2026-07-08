cmake_minimum_required(VERSION 3.20)
project(InferenceRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(runtime_demo src/main.cpp)

target_compile_features(runtime_demo PRIVATE cxx_std_20)

target_include_directories(runtime_demo PRIVATE includes)



add_executable(test_runtime tests/test_runtime.cpp)

target_compile_features(test_runtime PRIVATE cxx_std_20)

target_include_directories(test_runtime PRIVATE includes)

enable_testing()

add_test(NAME runtime_tests COMMAND $<TARGET_FILE:test_runtime>)
