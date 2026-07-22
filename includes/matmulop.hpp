#ifndef MATMULOP_HPP
#define MATMULOP_HPP

#include "tensor.hpp"
#include "operation.hpp"
#include <iostream>

class MatMulOp : public Operation{
    public:


        TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const override{
            const TensorDesc& t1 = inputs.at(0);
            const TensorDesc& t2 = inputs.at(1);

            if(t1.backend_ != t2.backend_) throw std::runtime_error("matmul(t1, t2) : error in different device");
            if(t1.backend_ != Backend::CPU) throw std::runtime_error("matmul(t1, t2) : error in GPU device");
            if(t1.dtype_ != t2.dtype_) throw std::runtime_error("matmul(t1, t2) : error in dtype");

            if(t1.shape_.dims().size() != 2 || t2.shape_.dims().size() != 2) throw std::runtime_error("matmul(t1, t2) : only 2 dims mat is supported");

            if(t1.shape_.dims().back() != t2.shape_.dims().front()) throw std::runtime_error("matmul(t1, t2) : error in shape");

            long row_1 = t1.shape_.dims().front();
            long col_1 = t1.shape_.dims().back();

            long row_2 = t2.shape_.dims().front();
            long col_2 = t2.shape_.dims().back();

            return TensorDesc(Shape({row_1, col_2}), t1.backend_, t1.dtype_);
        }

        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override{

            const Tensor & t1 = *inputs.at(0);
            const Tensor & t2 = *inputs.at(1);

            long row_1 = t1.shape().dims().front();
            long col_1 = t1.shape().dims().back();

            long row_2 = t2.shape().dims().front();
            long col_2 = t2.shape().dims().back();

            for(long k = 0; k < row_1; k++){
                for(long l = 0; l < col_2; l++){

                    for(long i = 0; i < col_1; i++){
                        output[k * col_2 + l] += t1[k * col_1 + i] * t2[i * col_2 + l];
                    }
                }
            }

            return output;
        }


        static Tensor forward(const Tensor & t1, const Tensor & t2){
            if(t1.backend() != t2.backend()) throw std::runtime_error("matmul(t1, t2) : error in different device");
            if(t1.backend() != Backend::CPU) throw std::runtime_error("matmul(t1, t2) : error in GPU device");
            if(t1.dtype() != t2.dtype()) throw std::runtime_error("matmul(t1, t2) : error in dtype");
            const Shape & shape1 = t1.shape();
            const Shape & shape2 = t2.shape();

            if(shape1.dims().size() != 2 || shape2.dims().size() != 2) throw std::runtime_error("matmul(t1, t2) : only 2 dims mat is supported");

            if(shape1.dims().back() != shape2.dims().front()) throw std::runtime_error("matmul(t1, t2) : error in shape");

            long row_1 = shape1.dims().front();
            long col_1 = shape1.dims().back();

            long row_2 = shape2.dims().front();
            long col_2 = shape2.dims().back();

            Tensor out = Tensor(Shape({row_1, col_2}), std::vector<Float32>(row_1 * col_2, 0), t1.dtype());
            for(long k = 0; k < row_1; k++){
                for(long l = 0; l < col_2; l++){

                    for(long i = 0; i < col_1; i++){
                        out[k * col_2 + l] += t1[k * col_1 + i] * t2[i * col_2 + l];
                    }
                }
            }

            return out;

        }
};


#endif
