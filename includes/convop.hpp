#ifndef CONVOP_HPP
#define CONVOP_HPP

#include "operation.hpp"

class ConvParam{
    
    public:
        std::vector<std::int64_t> strides{1, 1};
        std::vector<std::int64_t> pads{0, 0, 0, 0};
        std::vector<std::int64_t> dilations{1, 1};
        std::int64_t group = 1;
        std::string auto_pad = "NOTSET";
};

class ConvOp : public Operation{

    public:

        explicit ConvOp(ConvParam param)
        :param_(std::move(param)){}

        TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const override{
            const TensorDesc & t1 = inputs[0];
            const TensorDesc & t2 = inputs[1];

            if(t1.backend_ != t2.backend_) throw std::runtime_error("add(t1, t2) : error in different device");
            if(t1.backend_ != Backend::CPU) throw std::runtime_error("add(t1, t2) : error in GPU device");
            if(t1.dtype_ != t2.dtype_) throw std::runtime_error("add(t1, t2) : error in dtype");
            if(t1.shape_ != t2.shape_) throw std::runtime_error("add(t1, t2) : error in shape");

            return TensorDesc(t1.shape_, t1.backend_, t1.dtype_);
        }


        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override{

            const Tensor & t1 = *inputs.at(0);
            const Tensor & t2 = *inputs.at(1);

            for(int i = 0; i < t1.numel(); i++){
                output[i] = t1[i] + t2[i];
            }

            return output;
        }

    private:
        ConvParam param_;

        
};

#endif