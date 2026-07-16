cmake_minimum_required(VERSION 3.20)
project(InferenceRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(runtime src/tensor.cpp src/shape.cpp src/dtype.cpp)


add_executable(runtime_demo src/main.cpp)

target_compile_features(runtime_demo PRIVATE cxx_std_20)

target_include_directories(runtime PUBLIC includes)

target_link_libraries(runtime_demo PRIVATE runtime)


add_executable(test_runtime tests/test_runtime.cpp)

target_compile_features(test_runtime PRIVATE cxx_std_20)

# target_include_directories(test_runtime PRIVATE includes)

target_link_libraries(test_runtime PRIVATE runtime)

enable_testing()

add_test(NAME runtime_tests COMMAND $<TARGET_FILE:test_runtime>)
