#ifndef RELU_HPP
#define RELU_HPP

#include "tensor.hpp"
#include "operation.hpp"
#include <algorithm>

class ReluOp : public Operation{

    public:

        Tensor forward(const std::vector<const Tensor*> &inputs) const override{

            if(inputs.at(0)->backend() != Backend::CPU) throw std::runtime_error("relu(t1) : error in GPU device");
            Tensor out = *inputs.at(0);

            for(int i = 0; i < out.numel(); i++){
                out[i] = std::max(0.f, out[i]);
            }

            return out;

        }
        static Tensor forward(const Tensor &t){

            if(t.backend() != Backend::CPU) throw std::runtime_error("relu(t1) : error in GPU device");
            Tensor out = t;
            for(int i = 0; i < out.numel(); i++){
                out[i] = std::max(0.f, out[i]);
            }

            return out;
        }

};

#endif
