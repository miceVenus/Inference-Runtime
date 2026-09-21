#ifndef RELU_HPP
#define RELU_HPP

#include "tensor.hpp"
#include "operation.hpp"

using ReluParam = size_t;

class ReluOp : public Operation{

    public:

        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override;

        TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const override{

            const TensorDesc& t = inputs.at(0);
            if(t.backend_ != Backend::CPU) throw std::runtime_error("relu(t1) : error in GPU device");
            return t;
        }

};

#endif
