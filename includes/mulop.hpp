#ifndef MULOP_HPP
#define MULOP_HPP

#include "tensor.hpp"

class MulOp{

    public:
        static Tensor forward(const Tensor &t1, const Tensor &t2){
            if(t1.dtype() != t2.dtype()) throw std::runtime_error("mul(t1, t2) : error in dtype");
            if(t1.shape().dims().size() != t2.shape().dims().size()) throw std::runtime_error("mul(t1, t2) : error in shape");
            for(long i = 0; i < t1.shape().dims().size(); i++){
                if(t1.shape().dims()[i] != t2.shape().dims()[i])  throw std::runtime_error("mul(t1, t2) : error in shape");
            }
            std::vector<Float32> t_data = t1.data();
            for(int i = 0; i < t_data.size(); i++){
                t_data[i] *= t2.data()[i];
            }

            return Tensor(t1.shape(), std::move(t_data), t1.dtype());
        }

};

#endif