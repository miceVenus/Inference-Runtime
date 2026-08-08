#ifndef MULOP_HPP
#define MULOP_HPP

#include "operation.hpp"
#include "tensor.hpp"

using MulParam = size_t;

class MulOp : public Operation{

    
    public:

        TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const override{

            const TensorDesc& t1 = inputs.at(0);
            const TensorDesc& t2 = inputs.at(1);

            if(t1.backend_ != t2.backend_) throw std::runtime_error("mul(t1, t2) : error in different device");
            if(t1.backend_ != Backend::CPU) throw std::runtime_error("mul(t1, t2) : error in GPU device");
            if(t1.dtype_ != t2.dtype_) throw std::runtime_error("mul(t1, t2) : error in dtype");

            Shape output = broadcast_shape(t1.shape_, t2.shape_);
            return TensorDesc(std::move(output), t1.backend_, t1.dtype_);
        }

        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override{

            const Tensor & lhs = *inputs.at(0);
            const Tensor & rhs = *inputs.at(1);

            return binary_broadcast_cpu(lhs, rhs, output, [](Float32 a, Float32 b){return a * b;});
        }

};

#endif