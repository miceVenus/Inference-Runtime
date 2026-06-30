#ifndef RELU_HPP
#define RELU_HPP

#include "tensor.hpp"
#include <algorithm>

class ReluOp{

    public:
        static Tensor forward(const Tensor &t){

            std::vector<Float32> t_data = t.data();
            for(int i = 0; i < t_data.size(); i++){
                t_data[i] = std::max(0.f, t_data[i]);
            }

            return Tensor(t.shape(), std::move(t_data), t.dtype());
        }

};

#endif