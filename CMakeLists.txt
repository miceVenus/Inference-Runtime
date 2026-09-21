cmake_minimum_required(VERSION 3.20)
project(InferenceRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(PNG REQUIRED)
find_package(CUDAToolkit 13.2 REQUIRED)

message(STATUS "Protobuf version: ${Protobuf_VERSION}")
message(STATUS "protoc: ${Protobuf_PROTOC_EXECUTABLE}")

set(ONNX_PROTO
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/onnx.proto"
)

set(ONNX_PROTO_GEN_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)


file(MAKE_DIRECTORY "${ONNX_PROTO_GEN_DIR}")

protobuf_generate(
    LANGUAGE cpp
    PROTOS "${ONNX_PROTO}"
    PROTOC_OUT_DIR "${ONNX_PROTO_GEN_DIR}"
    OUT_VAR ONNX_PROTO_GENERATED
    APPEND_PATH
)

add_library(onnx_proto STATIC ${ONNX_PROTO_GENERATED})

target_include_directories(onnx_proto
    PUBLIC
        "${ONNX_PROTO_GEN_DIR}"
)

target_link_libraries(onnx_proto
    PUBLIC
        protobuf::libprotobuf
)

# Keep CPU operators in a separate library and optimize their kernels.
add_library(
    operator_cpu
        src/op/cpu/addop.cpp
        src/op/cpu/mulop.cpp
        src/op/cpu/reluop.cpp
        src/op/cpu/matmulop.cpp
        src/op/cpu/elementwise.cpp
        src/op/cpu/cpu_kernels.cpp
        src/broadcast.cpp
)

target_include_directories(operator_cpu PUBLIC includes)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable optimized kernels even when the rest of the project is Debug.
    target_compile_options(operator_cpu PRIVATE -O3)
endif()

add_library(
    runtime 
        src/tensor.cpp 
        src/shape.cpp 
        src/dtype.cpp 
        src/storage.cpp
        src/loader.cpp)

target_include_directories(runtime PUBLIC includes)

target_link_libraries(runtime PUBLIC onnx_proto operator_cpu CUDA::cudart)

add_executable(runtime_demo src/main.cpp)

target_compile_features(runtime_demo PRIVATE cxx_std_20)


target_link_libraries(runtime_demo PRIVATE runtime onnx_proto PNG::PNG)


add_executable(test_runtime tests/test_runtime.cpp)

target_compile_features(test_runtime PRIVATE cxx_std_20)

# target_include_directories(test_runtime PRIVATE includes)

target_link_libraries(test_runtime PRIVATE runtime onnx_proto)

enable_testing()

add_test(NAME runtime_tests COMMAND $<TARGET_FILE:test_runtime>)
