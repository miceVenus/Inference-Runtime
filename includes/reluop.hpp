#ifndef RELU_HPP
#define RELU_HPP

#include "tensor.hpp"
#include <algorithm>

class ReluOp{

    public:
        static Tensor forward(const Tensor &t){

            Tensor out = t;
            for(int i = 0; i < out.data().size(); i++){
                out[i] = std::max(0.f, out[i]);
            }

            return out;
        }

};

#endif