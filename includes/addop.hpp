#ifndef ADDOP_HPP
#define ADDOP_HPP

#include "tensor.hpp"
#include "operation.hpp"

using AddParam = size_t;

class AddOp : public Operation{

    public:

        TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const override;
        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override;
};

#endif