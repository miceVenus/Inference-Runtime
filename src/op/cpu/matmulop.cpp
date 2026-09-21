#include "matmulop.hpp"

#include "cpu_kernels.hpp"
#include "cpu_parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

// Keep short products serial to avoid task scheduling overhead.
constexpr std::size_t matmul_parallel_work_threshold = 32 * 1024;
// Split long reductions only when the output exposes too few tiles.
constexpr long matmul_split_k_threshold = 8 * 1024;
constexpr long matmul_split_k_grain = 4 * 1024;

}  // namespace

Tensor& MatMulOp::forward(
    const std::vector<const Tensor*>& inputs,
    Tensor& output) const {
    const Tensor& lhs = *inputs.at(0);
    const Tensor& rhs = *inputs.at(1);

    std::vector<long> lhs_dims = lhs.shape().dims();
    std::vector<long> rhs_dims = rhs.shape().dims();
    // Normalize vectors to matrices before applying ONNX MatMul rules.
    if (lhs_dims.size() == 1) {
        lhs_dims = {1, lhs_dims[0]};
    }
    if (rhs_dims.size() == 1) {
        rhs_dims = {rhs_dims[0], 1};
    }

    const long M = lhs_dims[lhs_dims.size() - 2];
    const long K = lhs_dims.back();
    const long N = rhs_dims.back();
    const long lhs_matrix_size = M * K;
    const long rhs_matrix_size = K * N;
    const long output_matrix_size = M * N;

    // Leading axes are broadcast batch dimensions for the matrix products.
    Shape lhs_batch_shape(std::vector<long>(lhs_dims.begin(), lhs_dims.end() - 2));
    Shape rhs_batch_shape(std::vector<long>(rhs_dims.begin(), rhs_dims.end() - 2));
    const BroadcastPlan batch_plan =
        make_broadcast_plan(lhs_batch_shape, rhs_batch_shape);
    const auto& batch_dims = batch_plan.output_shape.dims();
    const std::size_t batch_count =
        static_cast<std::size_t>(batch_plan.output_shape.numel());

    const std::size_t row_tiles = static_cast<std::size_t>(
        (M + cpu_runtime::matmul_tile_rows - 1) /
        cpu_runtime::matmul_tile_rows);
    const std::size_t column_tiles = static_cast<std::size_t>(
        (N + cpu_runtime::matmul_tile_columns - 1) /
        cpu_runtime::matmul_tile_columns);
    const std::size_t tiles_per_batch = row_tiles * column_tiles;
    const std::size_t job_count = batch_count * tiles_per_batch;
    if (job_count == 0) {
        return output;
    }

    const Float32* lhs_data = lhs.raw_data();
    const Float32* rhs_data = rhs.raw_data();
    Float32* output_data = output.raw_data();
    const std::size_t output_count =
        static_cast<std::size_t>(output.numel());
    const std::size_t output_matrix_count =
        static_cast<std::size_t>(output_matrix_size);

    const auto run_tile = [&](std::size_t job,
                              Float32* destination,
                              long reduction_begin,
                              long reduction_end) {
        // Decode the flattened batch index and honor broadcasted input axes.
        const std::size_t batch = job / tiles_per_batch;
        const std::size_t tile = job % tiles_per_batch;
        const std::size_t tile_row = tile / column_tiles;
        const std::size_t tile_column = tile % column_tiles;

        std::size_t remaining = batch;
        long lhs_offset = 0;
        long rhs_offset = 0;
        for (std::size_t axis = batch_dims.size(); axis > 0; --axis) {
            const std::size_t index = axis - 1;
            const auto dimension = static_cast<std::size_t>(batch_dims[index]);
            const long coordinate = static_cast<long>(remaining % dimension);
            remaining /= dimension;
            lhs_offset += coordinate * batch_plan.lhs_strides[index] * lhs_matrix_size;
            rhs_offset += coordinate * batch_plan.rhs_strides[index] * rhs_matrix_size;
        }

        cpu_runtime::matmul_tile(
            lhs_data + lhs_offset,
            rhs_data + rhs_offset,
            destination + batch * output_matrix_count,
            M,
            K,
            N,
            static_cast<long>(tile_row) * cpu_runtime::matmul_tile_rows,
            static_cast<long>(tile_column) * cpu_runtime::matmul_tile_columns,
            reduction_begin,
            reduction_end);
    };

    const std::size_t total_work = output_count * static_cast<std::size_t>(K);
    const std::size_t minimum_job_grain =
        total_work < matmul_parallel_work_threshold ? job_count : 1;

    // Split K when output tiles cannot fill the worker pool.
    if (total_work >= matmul_parallel_work_threshold &&
        K >= matmul_split_k_threshold &&
        job_count < cpu_runtime::worker_pool().thread_count()) {
        const std::size_t worker_budget =
            cpu_runtime::worker_pool().thread_count() / job_count;
        const std::size_t reduction_chunks = static_cast<std::size_t>(
            1 + (K - 1) / matmul_split_k_grain);
        const std::size_t split_count =
            std::min(worker_budget, reduction_chunks);

        if (split_count > 1) {
            std::vector<Float32> partial_outputs(output_count * split_count);
            const std::size_t split_jobs = job_count * split_count;

            cpu_runtime::parallel_for(split_jobs, 1, [&](std::size_t begin, std::size_t end) {
                for (std::size_t job = begin; job < end; ++job) {
                    const std::size_t split = job / job_count;
                    const std::size_t tile_job = job % job_count;
                    const long reduction_begin = static_cast<long>(
                        (static_cast<std::size_t>(K) * split) / split_count);
                    const long reduction_end = static_cast<long>(
                        (static_cast<std::size_t>(K) * (split + 1)) / split_count);
                    run_tile(
                        tile_job,
                        partial_outputs.data() + split * output_count,
                        reduction_begin,
                        reduction_end);
                }
            });

            // Combine partial sums after all split-K tasks have completed.
            cpu_runtime::parallel_for(output_count, 16 * 1024, [&](std::size_t begin, std::size_t end) {
                for (std::size_t index = begin; index < end; ++index) {
                    Float32 sum = 0.0f;
                    for (std::size_t split = 0; split < split_count; ++split) {
                        sum += partial_outputs[split * output_count + index];
                    }
                    output_data[index] = sum;
                }
            });
            return output;
        }
    }

    cpu_runtime::parallel_for(job_count, minimum_job_grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t job = begin; job < end; ++job) {
            run_tile(job, output_data, 0, K);
        }
    });
    return output;
}
