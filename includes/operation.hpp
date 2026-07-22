#ifndef OPERATION_HPP
#define OPERATION_HPP

#include "tensor.hpp"
#include <vector>

enum class OP_TYPE{
    AddOp,
    MulOp,
    ReluOp,
    MatMulOp,
};

class Operation{
    public:
        virtual ~Operation() = default;

        // virtual TensorDesc forward_T(const std::vector<const Tensor*> &inputs) const = 0;
        virtual TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const = 0;
        virtual Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const = 0;
};


#endif