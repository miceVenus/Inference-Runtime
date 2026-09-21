#include "broadcast.hpp"

#include "cpu_kernels.hpp"
#include "cpu_parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

constexpr std::size_t elementwise_grain = 16 * 1024;

std::pair<long, long> broadcast_row_offsets(
    std::size_t row,
    const std::vector<long>& output_dims,
    const BroadcastPlan& plan) {
    // Map one flattened output row to each input's broadcasted row.
    long lhs_offset = 0;
    long rhs_offset = 0;
    std::size_t remaining = row;

    for (std::size_t axis = output_dims.size(); axis > 1; --axis) {
        const std::size_t index = axis - 2;
        const auto dimension = static_cast<std::size_t>(output_dims[index]);
        const long coordinate = static_cast<long>(remaining % dimension);
        remaining /= dimension;
        lhs_offset += coordinate * plan.lhs_strides[index];
        rhs_offset += coordinate * plan.rhs_strides[index];
    }

    return {lhs_offset, rhs_offset};
}

}  // namespace

Tensor& binary_broadcast_cpu(
    const Tensor& lhs,
    const Tensor& rhs,
    Tensor& output,
    cpu_runtime::BinaryOpKind operation) {
    if (lhs.backend() != Backend::CPU || rhs.backend() != Backend::CPU ||
        output.backend() != Backend::CPU) {
        throw std::runtime_error("CPU binary operation received non-CPU storage");
    }

    const BroadcastPlan plan = make_broadcast_plan(lhs.shape(), rhs.shape());
    if (plan.output_shape != output.shape()) {
        throw std::runtime_error("binary output shape does not match broadcast shape");
    }

    const long numel = output.numel();
    if (numel <= 0) {
        return output;
    }

    const auto count = static_cast<std::size_t>(numel);
    const Float32* lhs_data = lhs.raw_data();
    const Float32* rhs_data = rhs.raw_data();
    Float32* output_data = output.raw_data();
    const bool lhs_is_flat = lhs.numel() == numel;
    const bool rhs_is_flat = rhs.numel() == numel;

    // Contiguous operands need no broadcast index calculation.
    if (lhs_is_flat && rhs_is_flat) {
        cpu_runtime::parallel_for(count, elementwise_grain, [&](std::size_t begin, std::size_t end) {
            cpu_runtime::binary_span(
                lhs_data + begin,
                1,
                rhs_data + begin,
                1,
                output_data + begin,
                end - begin,
                operation);
        });
        return output;
    }

    // A scalar operand is represented by a zero stride in the SIMD kernel.
    if (lhs_is_flat && rhs.numel() == 1) {
        cpu_runtime::parallel_for(count, elementwise_grain, [&](std::size_t begin, std::size_t end) {
            cpu_runtime::binary_span(
                lhs_data + begin,
                1,
                rhs_data,
                0,
                output_data + begin,
                end - begin,
                operation);
        });
        return output;
    }

    if (rhs_is_flat && lhs.numel() == 1) {
        cpu_runtime::parallel_for(count, elementwise_grain, [&](std::size_t begin, std::size_t end) {
            cpu_runtime::binary_span(
                lhs_data,
                0,
                rhs_data + begin,
                1,
                output_data + begin,
                end - begin,
                operation);
        });
        return output;
    }

    // Walk row segments so the innermost dimension stays contiguous.
    const auto& output_dims = plan.output_shape.dims();
    const long inner_size = output_dims.empty() ? 1 : output_dims.back();
    const long lhs_inner_stride = output_dims.empty() ? 0 : plan.lhs_strides.back();
    const long rhs_inner_stride = output_dims.empty() ? 0 : plan.rhs_strides.back();

    cpu_runtime::parallel_for(count, elementwise_grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t linear = begin; linear < end;) {
            const std::size_t row = linear / static_cast<std::size_t>(inner_size);
            const std::size_t inner_index = linear % static_cast<std::size_t>(inner_size);
            const std::size_t row_end = std::min(
                end,
                (row + 1) * static_cast<std::size_t>(inner_size));
            const std::size_t segment_size = row_end - linear;
            const auto [lhs_row_offset, rhs_row_offset] =
                broadcast_row_offsets(row, output_dims, plan);

            const Float32* lhs_segment = lhs_is_flat
                ? lhs_data + linear
                : lhs_data + lhs_row_offset +
                    static_cast<long>(inner_index) * lhs_inner_stride;
            const Float32* rhs_segment = rhs_is_flat
                ? rhs_data + linear
                : rhs_data + rhs_row_offset +
                    static_cast<long>(inner_index) * rhs_inner_stride;

            cpu_runtime::binary_span(
                lhs_segment,
                lhs_is_flat ? 1 : lhs_inner_stride,
                rhs_segment,
                rhs_is_flat ? 1 : rhs_inner_stride,
                output_data + linear,
                segment_size,
                operation);
            linear = row_end;
        }
    });

    return output;
}
