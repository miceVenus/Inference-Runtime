#ifndef BROADCAST_HPP
#define BROADCAST_HPP

#include "shape.hpp"
#include "tensor.hpp"
#include "cpu_kernels.hpp"


struct BroadcastPlan{

    Shape output_shape;
    std::vector<long> lhs_strides;
    std::vector<long> rhs_strides;
};

BroadcastPlan make_broadcast_plan(const Shape& lhs, const Shape& rhs);

Shape broadcast_shape(const Shape& lhs, const Shape& rhs);

Tensor& add_broadcast_cuda();

Tensor& mul_broadcast_cuda();

Tensor& binary_broadcast_cpu(
    const Tensor& lhs,
    const Tensor& rhs,
    Tensor& output,
    cpu_runtime::BinaryOpKind operation);

#endif
