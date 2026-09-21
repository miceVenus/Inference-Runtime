#ifndef BROADCAST_HPP
#define BROADCAST_HPP

#include "shape.hpp"
#include "tensor.hpp"


struct BroadcastPlan{

    Shape output_shape;
    std::vector<long> lhs_strides;
    std::vector<long> rhs_strides;
};

BroadcastPlan make_broadcast_plan(const Shape& lhs, const Shape& rhs);

Shape broadcast_shape(const Shape& lhs, const Shape& rhs);

__global__ void vector_add(const float* lhs, const float* rhs, float* output, int numel, const long* lhs_strides, const long* rhs_strides, const long* output_dims, long axis){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= numel) return;

    long remaining = index;
    long lhs_offset = 0;
    long rhs_offset = 0;

    for (long axi = axis - 1; axi >= 0; axi--) {
        const long coordinate = remaining % output_dims[axi];

        remaining /= output_dims[axi];

        lhs_offset += coordinate * lhs_strides[axi];

        rhs_offset += coordinate * rhs_strides[axi];
    }

    output[index] = lhs[lhs_offset] + rhs[rhs_offset];
}


#pragma once

template <typename BinaryFunction>
Tensor& binary_broadcast_cpu(const Tensor& lhs, const Tensor& rhs, Tensor& output, BinaryFunction function){

        const auto plan = make_broadcast_plan(lhs.shape(), rhs.shape());

        const auto& output_dims = plan.output_shape.dims();

        for (long linear = 0; linear < output.numel(); linear++) {
            long remaining = linear;
            long lhs_offset = 0;
            long rhs_offset = 0;

            for (long axis = output_dims.size() - 1; axis >= 0; axis--) {
                const long coordinate = remaining % output_dims[axis];

                remaining /= output_dims[axis];

                lhs_offset += coordinate * plan.lhs_strides[axis];

                rhs_offset += coordinate * plan.rhs_strides[axis];
            }

            output[linear] = function(lhs[lhs_offset], rhs[rhs_offset]);
        }

        return output;
}

#endif