#ifndef RELU_HPP
#define RELU_HPP

#include "tensor.hpp"
#include "operation.hpp"
#include <algorithm>

class ReluOp : public Operation{

    public:

        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override{

            const Tensor & t = *inputs.at(0);

            for(int i = 0; i < t.numel(); i++){
                output[i] = std::max(0.f, t[i]);
            }

            return output;

        }

        TensorDesc forward_T(const std::vector<const Tensor*> &inputs) const override{

            TensorDesc t = TensorDesc(*inputs.at(0));
            if(t.backend_ != Backend::CPU) throw std::runtime_error("relu(t1) : error in GPU device");
            return t;
        }

};

#endif
