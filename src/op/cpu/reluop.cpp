#include "reluop.hpp"

#include "cpu_kernels.hpp"
#include "cpu_parallel.hpp"

#include <cstddef>

namespace {

constexpr std::size_t relu_grain = 16 * 1024;

}  // namespace

Tensor& ReluOp::forward(
    const std::vector<const Tensor*>& inputs,
    Tensor& output) const {
    const Tensor& input = *inputs.at(0);
    const auto count = static_cast<std::size_t>(input.numel());
    const Float32* input_data = input.raw_data();
    Float32* output_data = output.raw_data();

    // Keep small tensors serial and split larger spans across the pool.
    cpu_runtime::parallel_for(count, relu_grain, [&](std::size_t begin, std::size_t end) {
        cpu_runtime::relu_span(
            input_data + begin,
            output_data + begin,
            end - begin);
    });
    return output;
}
