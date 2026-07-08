#ifndef RELU_HPP
#define RELU_HPP

#include "tensor.hpp"
#include "operation.hpp"
#include <algorithm>

class ReluOp : public Operation{

    public:

        Tensor forward(const std::vector<const Tensor*> &inputs) const override{

            Tensor out = *inputs.at(0);
            for(int i = 0; i < out.data().size(); i++){
                out[i] = std::max(0.f, out[i]);
            }

            return out;

        }
        static Tensor forward(const Tensor &t){

            Tensor out = t;
            for(int i = 0; i < out.data().size(); i++){
                out[i] = std::max(0.f, out[i]);
            }

            return out;
        }

};

#endif