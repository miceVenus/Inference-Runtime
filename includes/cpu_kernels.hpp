#ifndef CPU_KERNELS_HPP
#define CPU_KERNELS_HPP

#include "dtype.hpp"

#include <cstddef>

namespace cpu_runtime {

enum class BinaryOpKind {
    Add,
    Mul
};

// Output tiles limit register use and give workers disjoint regions.
inline constexpr long matmul_tile_rows = 4;
inline constexpr long matmul_tile_columns = 16;

// A zero stride broadcasts one value across the span.
void binary_span(
    const Float32* lhs,
    long lhs_stride,
    const Float32* rhs,
    long rhs_stride,
    Float32* output,
    std::size_t count,
    BinaryOpKind operation);

void relu_span(const Float32* input, Float32* output, std::size_t count);

// Compute one output tile over a possibly split reduction range.
void matmul_tile(
    const Float32* lhs,
    const Float32* rhs,
    Float32* output,
    long M,
    long K,
    long N,
    long row_begin,
    long column_begin,
    long reduction_begin,
    long reduction_end);

}  // namespace cpu_runtime

#endif
