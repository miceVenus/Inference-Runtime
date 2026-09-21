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

MNIST 示例程序默认读取以下文件：

- `test_samples/mnist_heavy_mlp.onnx`
- `test_samples/mnist_test_00000_label_7.png`

确保文件位于对应路径后运行：

```bash
./build/runtime_demo
```

## MNIST 效果

性能测试使用 `mnist_heavy_mlp.onnx`：8 个隐藏层、每层 1024 个单元，包含 9 个 MatMul、9 个 Add 和 8 个 ReLU。模型实验记录的 MNIST 测试集准确率为 98.99%。运行时对提供的标签为 7 的样例也预测为 7。

下表是在 Intel Core i7-12700 上的开发机测量结果。运行时使用 16 个 CPU worker，检测到 AVX2/FMA。每个 batch 使用同一张样例图重复填充；每档预热 5 次、测量 100 次。计时使用墙钟，只包含 `Executor::run()`，不含模型加载、图片解码和输入设置。

| Batch | 每次 `run()` 平均延迟 | P95 延迟 | 吞吐 |
|---:|---:|---:|---:|
| 1 | 2.130 ms | 3.184 ms | 469 张/秒 |
| 4 | 2.071 ms | 2.709 ms | 1,932 张/秒 |
| 8 | 2.396 ms | 3.428 ms | 3,339 张/秒 |
| 16 | 3.816 ms | 5.475 ms | 4,193 张/秒 |

Batch 16 的平均单次延迟约 3.8 ms，吞吐约为 batch 1 的 8.9 倍 而未经优化时 每次run()的平均延迟约为30-40ms。