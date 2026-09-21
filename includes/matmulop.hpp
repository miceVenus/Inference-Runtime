#ifndef MATMULOP_HPP
#define MATMULOP_HPP

#include "tensor.hpp"
#include "operation.hpp"
#include "broadcast.hpp"

using MatMulParam = size_t;

class MatMulOp : public Operation{
    public:


        TensorDesc forward_T(const std::vector<TensorDesc> &inputs) const override{
            const TensorDesc& t1 = inputs.at(0);
            const TensorDesc& t2 = inputs.at(1);

            auto lhs_dims = t1.shape_.dims();
            auto rhs_dims = t2.shape_.dims();

            bool lhs_is_vector = false;
            bool rhs_is_vector = false;
            

            if(t1.backend_ != t2.backend_) throw std::runtime_error("matmul(t1, t2) : error in different device");
            if(t1.backend_ != Backend::CPU) throw std::runtime_error("matmul(t1, t2) : error in GPU device");
            if(t1.dtype_ != t2.dtype_) throw std::runtime_error("matmul(t1, t2) : error in dtype");


            if (lhs_dims.empty() || rhs_dims.empty()) {
                throw std::runtime_error(
                    "MatMul inputs must have rank >= 1"
                );
            }

            // rank == 1 t1 is a vector
            if(lhs_dims.size() == 1){
                lhs_dims = {1, lhs_dims[0]};
                lhs_is_vector = true;
            }

            // rank == 1 t2 is a vector
            if(rhs_dims.size() == 1){
                rhs_dims = {rhs_dims[0], 1};
                rhs_is_vector = true;
            }

            Shape lhs_batch_shape = Shape(std::vector<long>(lhs_dims.begin(), lhs_dims.begin() + lhs_dims.size() - 2));
            Shape rhs_batch_shape = Shape(std::vector<long>(rhs_dims.begin(), rhs_dims.begin() + rhs_dims.size() - 2));

            Shape lhs_matrix_shape = Shape({lhs_dims[lhs_dims.size() - 2], lhs_dims[lhs_dims.size() - 1]});
            Shape rhs_matrix_shape = Shape({rhs_dims[rhs_dims.size() - 2], rhs_dims[rhs_dims.size() - 1]});

            if(lhs_matrix_shape.dims().back() != rhs_matrix_shape.dims().front()) throw std::runtime_error("matmul(t1, t2) : error in shape");


            // Broadcast batch axes before computing each matrix product.
            Shape tmp = broadcast_shape(lhs_batch_shape, rhs_batch_shape);

            long M = lhs_matrix_shape.dims().front();
            long N = rhs_matrix_shape.dims().back();

            std::vector<long> res_shape = std::vector<long>(tmp.dims().begin(), tmp.dims().end());
            
            if(!lhs_is_vector) res_shape.push_back(M);

            if(!rhs_is_vector) res_shape.push_back(N);

            return TensorDesc(Shape(std::move(res_shape)), t1.backend_, t1.dtype_);
        }

        Tensor& forward(const std::vector<const Tensor*> &inputs, Tensor & output) const override;
};


#endif
