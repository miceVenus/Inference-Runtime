#include "cpu_kernels.hpp"

#include "cpu_parallel.hpp"

#include <algorithm>
#include <cstddef>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
// Isolate AVX2 code so runtime dispatch remains safe on older CPUs.
#define CPU_KERNELS_HAS_AVX2_TARGET 1
#define CPU_KERNELS_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define CPU_KERNELS_HAS_AVX2_TARGET 0
#define CPU_KERNELS_AVX2_TARGET
#endif

namespace cpu_runtime {
namespace {

Float32 apply_binary(Float32 lhs, Float32 rhs, BinaryOpKind operation) {
    if (operation == BinaryOpKind::Add) {
        return lhs + rhs;
    }
    return lhs * rhs;
}

#if CPU_KERNELS_HAS_AVX2_TARGET
CPU_KERNELS_AVX2_TARGET
void binary_span_avx2(
    const Float32* lhs,
    long lhs_stride,
    const Float32* rhs,
    long rhs_stride,
    Float32* output,
    std::size_t count,
    BinaryOpKind operation) {
    // Broadcast operands use stride zero; contiguous operands use stride one.
    std::size_t index = 0;

    if (operation == BinaryOpKind::Add) {
        for (; index + 8 <= count; index += 8) {
            const __m256 lhs_values = lhs_stride == 0
                ? _mm256_set1_ps(lhs[0])
                : _mm256_loadu_ps(lhs + index);
            const __m256 rhs_values = rhs_stride == 0
                ? _mm256_set1_ps(rhs[0])
                : _mm256_loadu_ps(rhs + index);
            _mm256_storeu_ps(output + index, _mm256_add_ps(lhs_values, rhs_values));
        }
    } else {
        for (; index + 8 <= count; index += 8) {
            const __m256 lhs_values = lhs_stride == 0
                ? _mm256_set1_ps(lhs[0])
                : _mm256_loadu_ps(lhs + index);
            const __m256 rhs_values = rhs_stride == 0
                ? _mm256_set1_ps(rhs[0])
                : _mm256_loadu_ps(rhs + index);
            _mm256_storeu_ps(output + index, _mm256_mul_ps(lhs_values, rhs_values));
        }
    }

    for (; index < count; ++index) {
        const Float32 lhs_value = lhs_stride == 0 ? lhs[0] : lhs[index];
        const Float32 rhs_value = rhs_stride == 0 ? rhs[0] : rhs[index];
        output[index] = apply_binary(lhs_value, rhs_value, operation);
    }
}

CPU_KERNELS_AVX2_TARGET
void relu_span_avx2(
    const Float32* input,
    Float32* output,
    std::size_t count) {
    const __m256 zero = _mm256_setzero_ps();
    std::size_t index = 0;

    for (; index + 8 <= count; index += 8) {
        const __m256 values = _mm256_loadu_ps(input + index);
        _mm256_storeu_ps(output + index, _mm256_max_ps(values, zero));
    }

    for (; index < count; ++index) {
        output[index] = std::max(0.0f, input[index]);
    }
}

CPU_KERNELS_AVX2_TARGET
Float32 dot_product_avx2(
    const Float32* lhs,
    const Float32* rhs,
    long begin,
    long end) {
    // Multiple accumulators reduce FMA dependency stalls.
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    long index = begin;

    for (; index + 32 <= end; index += 32) {
        sum0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs + index),
            _mm256_loadu_ps(rhs + index),
            sum0);
        sum1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs + index + 8),
            _mm256_loadu_ps(rhs + index + 8),
            sum1);
        sum2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs + index + 16),
            _mm256_loadu_ps(rhs + index + 16),
            sum2);
        sum3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs + index + 24),
            _mm256_loadu_ps(rhs + index + 24),
            sum3);
    }
    for (; index + 8 <= end; index += 8) {
        sum0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(lhs + index),
            _mm256_loadu_ps(rhs + index),
            sum0);
    }

    alignas(32) Float32 lanes[8];
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    _mm256_store_ps(lanes, sum0);
    Float32 result = 0.0f;
    for (Float32 lane : lanes) {
        result += lane;
    }
    for (; index < end; ++index) {
        result += lhs[index] * rhs[index];
    }
    return result;
}

template <int Rows, int Vectors>
CPU_KERNELS_AVX2_TARGET
void matmul_tile_avx2_microkernel(
    const Float32* lhs,
    const Float32* rhs,
    Float32* output,
    long K,
    long N,
    long row_begin,
    long column_begin,
    long reduction_begin,
    long reduction_end) {
    // Keep each output tile's accumulators in AVX registers.
    const __m256 zero = _mm256_setzero_ps();
    __m256 c00 = zero;
    __m256 c01 = zero;
    __m256 c10 = zero;
    __m256 c11 = zero;
    __m256 c20 = zero;
    __m256 c21 = zero;
    __m256 c30 = zero;
    __m256 c31 = zero;

    for (long k = reduction_begin; k < reduction_end; ++k) {
        const __m256 rhs0 = _mm256_loadu_ps(
            rhs + k * N + column_begin);
        __m256 rhs1 = zero;
        if constexpr (Vectors > 1) {
            rhs1 = _mm256_loadu_ps(
                rhs + k * N + column_begin + 8);
        }

        const __m256 lhs0 = _mm256_set1_ps(lhs[row_begin * K + k]);
        c00 = _mm256_fmadd_ps(lhs0, rhs0, c00);
        if constexpr (Vectors > 1) {
            c01 = _mm256_fmadd_ps(lhs0, rhs1, c01);
        }

        if constexpr (Rows > 1) {
            const __m256 lhs1 = _mm256_set1_ps(
                lhs[(row_begin + 1) * K + k]);
            c10 = _mm256_fmadd_ps(lhs1, rhs0, c10);
            if constexpr (Vectors > 1) {
                c11 = _mm256_fmadd_ps(lhs1, rhs1, c11);
            }
        }

        if constexpr (Rows > 2) {
            const __m256 lhs2 = _mm256_set1_ps(lhs[(row_begin + 2) * K + k]);
            c20 = _mm256_fmadd_ps(lhs2, rhs0, c20);
            if constexpr (Vectors > 1) {
                c21 = _mm256_fmadd_ps(lhs2, rhs1, c21);
            }
        }

        if constexpr (Rows > 3) {
            const __m256 lhs3 = _mm256_set1_ps(lhs[(row_begin + 3) * K + k]);
            c30 = _mm256_fmadd_ps(lhs3, rhs0, c30);
            if constexpr (Vectors > 1) {
                c31 = _mm256_fmadd_ps(lhs3, rhs1, c31);
            }
        }
    }

    _mm256_storeu_ps(output + row_begin * N + column_begin, c00);
    if constexpr (Vectors > 1) {
        _mm256_storeu_ps(output + row_begin * N + column_begin + 8, c01);
    }
    if constexpr (Rows > 1) {
        _mm256_storeu_ps(output + (row_begin + 1) * N + column_begin, c10);
        if constexpr (Vectors > 1) {
            _mm256_storeu_ps(output + (row_begin + 1) * N + column_begin + 8, c11);
        }
    }
    if constexpr (Rows > 2) {
        _mm256_storeu_ps(output + (row_begin + 2) * N + column_begin, c20);
        if constexpr (Vectors > 1) {
            _mm256_storeu_ps(output + (row_begin + 2) * N + column_begin + 8, c21);
        }
    }
    if constexpr (Rows > 3) {
        _mm256_storeu_ps(output + (row_begin + 3) * N + column_begin, c30);
        if constexpr (Vectors > 1) {
            _mm256_storeu_ps(output + (row_begin + 3) * N + column_begin + 8, c31);
        }
    }
}

CPU_KERNELS_AVX2_TARGET
void matmul_tile_avx2(
    const Float32* lhs,
    const Float32* rhs,
    Float32* output,
    long M,
    long K,
    long N,
    long row_begin,
    long column_begin,
    long reduction_begin,
    long reduction_end) {
    const long rows = std::min(matmul_tile_rows, M - row_begin);
    const long columns = std::min(matmul_tile_columns, N - column_begin);

    if (N == 1) {
        // Matrix-vector products use a dedicated dot-product path.
        for (long row = 0; row < rows; ++row) {
            output[row_begin + row] = dot_product_avx2(
                lhs + (row_begin + row) * K,
                rhs,
                reduction_begin,
                reduction_end);
        }
        return;
    }

    // Use up to two AVX vectors, then handle any narrow tail scalarly.
    const long vector_count = columns / 8;
    if (vector_count == 2) {
        switch (rows) {
            case 1:
                matmul_tile_avx2_microkernel<1, 2>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
            case 2:
                matmul_tile_avx2_microkernel<2, 2>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
            case 3:
                matmul_tile_avx2_microkernel<3, 2>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
            case 4:
                matmul_tile_avx2_microkernel<4, 2>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
        }
    } else if (vector_count == 1) {
        switch (rows) {
            case 1:
                matmul_tile_avx2_microkernel<1, 1>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
            case 2:
                matmul_tile_avx2_microkernel<2, 1>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
            case 3:
                matmul_tile_avx2_microkernel<3, 1>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
            case 4:
                matmul_tile_avx2_microkernel<4, 1>(lhs, rhs, output, K, N, row_begin, column_begin, reduction_begin, reduction_end);
                break;
        }
    }

    for (long row = 0; row < rows; ++row) {
        for (long column = vector_count * 8; column < columns; ++column) {
            Float32 sum = 0.0f;
            for (long k = reduction_begin; k < reduction_end; ++k) {
                sum += lhs[(row_begin + row) * K + k] *
                    rhs[k * N + column_begin + column];
            }
            output[(row_begin + row) * N + column_begin + column] = sum;
        }
    }
}
#endif

void matmul_tile_scalar(
    const Float32* lhs,
    const Float32* rhs,
    Float32* output,
    long M,
    long K,
    long N,
    long row_begin,
    long column_begin,
    long reduction_begin,
    long reduction_end) {
    // Keep a cache-friendly scalar fallback for CPUs without AVX2+FMA.
    const long rows = std::min(matmul_tile_rows, M - row_begin);
    const long columns = std::min(matmul_tile_columns, N - column_begin);

    for (long row = 0; row < rows; ++row) {
        for (long column = 0; column < columns; ++column) {
            output[(row_begin + row) * N + column_begin + column] = 0.0f;
        }
    }

    for (long k = reduction_begin; k < reduction_end; ++k) {
        for (long row = 0; row < rows; ++row) {
            const Float32 lhs_value = lhs[(row_begin + row) * K + k];
            for (long column = 0; column < columns; ++column) {
                output[(row_begin + row) * N + column_begin + column] +=
                    lhs_value * rhs[k * N + column_begin + column];
            }
        }
    }
}

}  // namespace

void binary_span(
    const Float32* lhs,
    long lhs_stride,
    const Float32* rhs,
    long rhs_stride,
    Float32* output,
    std::size_t count,
    BinaryOpKind operation) {
#if CPU_KERNELS_HAS_AVX2_TARGET
    // Select the vector kernel only when the host CPU supports it.
    static const bool use_avx2 = avx2_fma_available();
    if (use_avx2) {
        binary_span_avx2(lhs, lhs_stride, rhs, rhs_stride, output, count, operation);
        return;
    }
#endif

    for (std::size_t index = 0; index < count; ++index) {
        const Float32 lhs_value = lhs_stride == 0 ? lhs[0] : lhs[index];
        const Float32 rhs_value = rhs_stride == 0 ? rhs[0] : rhs[index];
        output[index] = apply_binary(lhs_value, rhs_value, operation);
    }
}

void relu_span(const Float32* input, Float32* output, std::size_t count) {
#if CPU_KERNELS_HAS_AVX2_TARGET
    // Keep the same runtime feature check for the fused ReLU span.
    static const bool use_avx2 = avx2_fma_available();
    if (use_avx2) {
        relu_span_avx2(input, output, count);
        return;
    }
#endif

    for (std::size_t index = 0; index < count; ++index) {
        output[index] = std::max(0.0f, input[index]);
    }
}

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
    long reduction_end) {
#if CPU_KERNELS_HAS_AVX2_TARGET
    // Fall back to scalar code when AVX2+FMA is unavailable.
    static const bool use_avx2 = avx2_fma_available();
    if (use_avx2) {
        matmul_tile_avx2(
            lhs,
            rhs,
            output,
            M,
            K,
            N,
            row_begin,
            column_begin,
            reduction_begin,
            reduction_end);
        return;
    }
#endif

    matmul_tile_scalar(
        lhs,
        rhs,
        output,
        M,
        K,
        N,
        row_begin,
        column_begin,
        reduction_begin,
        reduction_end);
}

}  // namespace cpu_runtime

#undef CPU_KERNELS_HAS_AVX2_TARGET
#undef CPU_KERNELS_AVX2_TARGET
