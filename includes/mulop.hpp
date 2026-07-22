#ifndef MULOP_HPP
#define MULOP_HPP

#include "operation.hpp"
#include "tensor.hpp"

class MulOp : public Operation{

    
    public:

        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override{

            const Tensor & t1 = *inputs.at(0);
            const Tensor & t2 = *inputs.at(1);

            for(int i = 0; i < t1.numel(); i++){
                output[i] = t1[i] * t2[i];
            }

            return output;
        }

        TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const override{

            const TensorDesc& t1 = inputs.at(0);
            const TensorDesc& t2 = inputs.at(1);

            if(t1.backend_ != t2.backend_) throw std::runtime_error("mul(t1, t2) : error in different device");
            if(t1.backend_ != Backend::CPU) throw std::runtime_error("mul(t1, t2) : error in GPU device");
            if(t1.dtype_ != t2.dtype_) throw std::runtime_error("mul(t1, t2) : error in dtype");
            if(t1.shape_ != t2.shape_) throw std::runtime_error("mul(t1, t2) : error in shape");
            
            return TensorDesc(t1.shape_, t1.backend_, t1.dtype_);
        }

};

#endif