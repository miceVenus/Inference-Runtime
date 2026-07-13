#ifndef ADDOP_HPP
#define ADDOP_HPP

#include "tensor.hpp"
#include "operation.hpp"
#include <iostream>

class AddOp : public Operation{

    public:

        
        Tensor forward(const std::vector<const Tensor*> &inputs) const override{

            const Tensor & t1 = *inputs.at(0);
            const Tensor & t2 = *inputs.at(1);

            if(t1.backend() != t2.backend()) throw std::runtime_error("add(t1, t2) : error in different device");
            if(t1.backend() != Backend::CPU) throw std::runtime_error("add(t1, t2) : error in GPU device");
            if(t1.dtype() != t2.dtype()) throw std::runtime_error("add(t1, t2) : error in dtype");
            if(t1.shape() != t2.shape()) throw std::runtime_error("add(t1, t2) : error in shape");

            Tensor out = t1;

            for(int i = 0; i < t1.numel(); i++){
                out[i] += t2[i];
            }

            return out;
        }


        static Tensor forward(const Tensor &t1, const Tensor &t2){
            if(t1.backend() != t2.backend()) throw std::runtime_error("add(t1, t2) : error in different device");
            if(t1.backend() != Backend::CPU) throw std::runtime_error("add(t1, t2) : error in GPU device");
            if(t1.dtype() != t2.dtype()) throw std::runtime_error("add(t1, t2) : error in dtype");
            if(t1.shape() != t2.shape()) throw std::runtime_error("add(t1, t2) : error in shape");

            Tensor out = t1;

            for(int i = 0; i < t1.numel(); i++){
                out[i] += t2[i];
            }

            return out;
        }

};

#endif