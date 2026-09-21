#include "addop.hpp"
#include "broadcast.hpp"

Tensor& AddOp::forward(const std::vector<const Tensor*> &inputs, Tensor & output) const{
    const Tensor & lhs = *inputs.at(0);
    const Tensor & rhs = *inputs.at(1);

    // Share the broadcast-aware SIMD path with multiplication.
    return binary_broadcast_cpu(
        lhs, rhs, output, cpu_runtime::BinaryOpKind::Add);
}

TensorDesc AddOp::forward_T(const std::vector<TensorDesc> &inputs) const{
    const TensorDesc & t1 = inputs[0];
    const TensorDesc & t2 = inputs[1];

    if(t1.backend_ != t2.backend_) throw std::runtime_error("add(t1, t2) : error in different device");
    if(t1.backend_ != Backend::CPU) throw std::runtime_error("add(t1, t2) : error in GPU device");
    if(t1.dtype_ != t2.dtype_) throw std::runtime_error("add(t1, t2) : error in dtype");
    
    Shape output_shape = broadcast_shape(t1.shape_, t2.shape_);

    return TensorDesc(std::move(output_shape), t1.backend_, t1.dtype_);
}
