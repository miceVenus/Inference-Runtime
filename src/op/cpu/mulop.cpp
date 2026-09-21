#include "mulop.hpp"

Tensor& MulOp::forward(
    const std::vector<const Tensor*>& inputs,
    Tensor& output) const {
    const Tensor& lhs = *inputs.at(0);
    const Tensor& rhs = *inputs.at(1);
    // Share the broadcast-aware SIMD path with addition.
    return binary_broadcast_cpu(
        lhs, rhs, output, cpu_runtime::BinaryOpKind::Mul);
}
