#ifndef ADDOP_HPP
#define ADDOP_HPP

#include "tensor.hpp"
#include <iostream>

class AddOp{

    public:
        static Tensor forward(const Tensor &t1, const Tensor &t2){
            if(t1.backend() != t2.backend()) throw std::runtime_error("add(t1, t2) : error in different device");
            if(t1.dtype() != t2.dtype()) throw std::runtime_error("add(t1, t2) : error in dtype");
            if(t1.shape() != t2.shape()) throw std::runtime_error("add(t1, t2) : error in shape");

            Tensor out = t1;

            for(int i = 0; i < t1.data().size(); i++){
                out[i] += t2[i];
            }

            return out;
        }

};

#endif