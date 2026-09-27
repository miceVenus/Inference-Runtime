# Inference Runtime

一个使用 C++20 编写的轻量级 ONNX 推理运行时。项目重点是静态执行计划、张量广播、生命周期分析和 CPU 算子优化。

## 项目特点

- **静态执行计划**：加载模型时建立计算图、排序节点并推导张量形状；执行器创建时据此规划中间存储。`run()` 按计划执行节点。
- **固定 batch**：在构建计算图前指定 batch size，使形状推导和内存规划都使用具体的 N。一个执行器在运行期间保持固定形状。
- **广播**：Add、Mul 支持按尾部维度对齐的广播；MatMul 支持批次维度广播。
- **内存规划**：根据张量的生命周期复用大小和设备相同的中间缓冲区。
- **CPU 并行与 SIMD**：使用共享线程池并行处理独立工作块；在支持的 Intel CPU 上运行 AVX2/FMA 内核，否则回退到标量实现。
- **Float32 张量**：当前推理数据路径使用 Float32。

## 执行流程

```text
ONNX 模型
   │
   ▼
Loader：解析节点、权重和输入形状；绑定固定 batch
   │
   ▼
Graph：拓扑排序、形状推导、张量生命周期分析
   │
   ▼
Executor：静态规划并复用中间缓冲区
   │
   ▼
CPU 算子：线程池并行 + AVX2/FMA
```

## 当前算子范围

| 算子 | CPU 支持 | 说明 |
|---|---|---|
| `MatMul` | 已实现 | 4×16 输出分块、AVX2/FMA 微内核、并行执行；长归约可拆分工作 |
| `Add` | 已实现 | 支持广播、AVX2 和并行执行 |
| `Mul` | 已实现 | 支持广播、AVX2 和并行执行 |
| `Relu` | 已实现 | AVX2 和并行执行 |
| `Conv` | 未实现 | 目前是占位实现，不应用于实际推理 |

当前运行路径面向 CPU；尚未接入可用的 CUDA 推理算子。虽然 CMake 配置目前仍依赖 CUDA Toolkit 和 CUDA Runtime，但这不代表模型会使用 GPU 执行。

## 固定 batch 推理

batch size 在加载模型时指定。形状推导和执行器的内存规划会使用同一个 N；更换 N 时，应重新加载并创建对应的 Graph 和 Executor。

```cpp
#include "executor.hpp"
#include "loader.hpp"

#include <filesystem>
#include <utility>
#include <vector>

constexpr long batch_size = 4;

Graph graph = Loader::load("model.onnx", batch_size);
std::vector<std::filesystem::path> image_paths{
    "image_0.png", "image_1.png", "image_2.png", "image_3.png"};
Tensor input = load_image_batch(image_paths);  // [4, 784]

Executor executor(graph);
executor.set_input(graph.inputs().front(), std::move(input));
executor.run();
```

`load_image_batch` 用于 28×28 的灰度 MNIST 图片，并采用训练时相同的归一化方式。通用模型可以自行构造形状与 Graph 输入描述一致的 Tensor。输入不匹配时，`set_input` 会抛出异常。

当前静态绑定主要支持首维 batch。ONNX 符号维度不会在每次 `run()` 时动态解析；需要变更 batch 时，应创建新的执行计划。

## 构建与运行

需要 CMake 3.20 以上、支持 C++20 的编译器、Protobuf 开发库和 `protoc`、libpng，以及当前构建配置要求的 CUDA Toolkit 13.2 以上。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

性能基准程序接收模型路径、MNIST 数据目录，以及可选的 split、batch size 和样本数。数据目录可以是 `MNIST` 或其 `raw` 子目录；默认使用测试集、batch size 16 和整个 split。它读取未压缩的 IDX 文件，并使用与 `load_image_batch` 相同的归一化方式：

```bash
./build/runtime_demo \
  test_samples/mnist_heavy_mlp.onnx \
  experiments/mnist_onnx/data/MNIST \
  test \
  16
```

参数格式为 `MODEL.onnx MNIST_DIR [train|test] [BATCH_SIZE] [MAX_SAMPLES]`。例如把 `test` 改为 `train` 可测训练集；`MAX_SAMPLES` 可限制测量样本数。程序预热 5 次，然后遍历所选数据，输出每批延迟、吞吐、分类准确率、worker 数和 AVX2/FMA 状态。吞吐只按 `Executor::run()` 的累计时间计算，不包含 IDX 读取、归一化、组 batch、设置输入或准确率计算。最后一个 batch 不足时会用最后一张有效图片补齐，准确率只统计真实数据样本。

## MNIST 性能结果

性能测试使用 `mnist_heavy_mlp.onnx`：8 个隐藏层、每层 1024 个单元，包含 9 个 MatMul、9 个 Add 和 8 个 ReLU。下表使用完整的 10,000 张 MNIST 测试集，每个 batch size 单独运行；当前环境使用 16 个 CPU worker，并检测到 AVX2/FMA。延迟是每个固定大小 batch 的 `Executor::run()` 时间，吞吐是有效图片数除以这些运行时间之和。

| Batch | 每批平均延迟 | P95 延迟 | 吞吐 | 测试集准确率 |
|---:|---:|---:|---:|---:|
| 1 | 2.029 ms | 2.832 ms | 493 张/秒 | 98.990% |
| 4 | 2.022 ms | 2.690 ms | 1,978 张/秒 | 98.990% |
| 8 | 2.366 ms | 3.003 ms | 3,381 张/秒 | 98.990% |
| 16 | 2.945 ms | 3.620 ms | 5,434 张/秒 | 98.990% |