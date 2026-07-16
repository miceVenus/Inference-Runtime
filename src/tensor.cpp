#include "tensor.hpp"


TensorDesc::TensorDesc(const Shape shape, Backend backend, Dtype dtype)
:shape_(shape), backend_(backend), dtype_(dtype){}

TensorDesc::TensorDesc(const Tensor & t)
:shape_(t.shape()), backend_(t.backend()), dtype_(t.dtype()){}



Tensor::Tensor(const TensorDesc &T)
:shape_(T.shape_), storage_(StorageFac::create(T.shape_.numel(), T.backend_)), dtype_(T.dtype_){}


Tensor::Tensor(const TensorDesc &T, std::unique_ptr<Storage> storage)
:shape_(T.shape_), storage_(std::move(storage)), dtype_(T.dtype_){}

Tensor::Tensor(const Shape & shape, Dtype dtype, Backend backend) 
: shape_(shape), storage_(StorageFac::create(shape.numel(), backend)), dtype_(dtype){}


Tensor::Tensor(const Shape & shape, std::vector<Float32> data, Dtype dtype, Backend backend)
: shape_(shape), storage_(StorageFac::create(std::move(data), backend)), dtype_(dtype){
    if(shape.numel() != storage_->size()){
        throw std::runtime_error("error in data shape");
    }
}

Tensor::Tensor(const Tensor & tensor) 
: shape_(tensor.shape_), storage_(tensor.storage_->clone()), dtype_(tensor.dtype_){}


Tensor& Tensor::operator=(const Tensor & tensor) {
    this->storage_ = tensor.storage_->clone();
    this->shape_ = tensor.shape_;
    this->dtype_ = tensor.dtype_;
    return *this;
}


std::unique_ptr<Storage> Tensor::move_storage(){
    return std::move(storage_);
}

Float32 * Tensor::raw_data(){
    return storage_->raw_data();
}

const Float32 * Tensor::raw_data() const{
    return storage_->raw_data();
}

Float32& Tensor::operator[](std::size_t index){
    if(index < storage_->size()){
        return storage_->raw_data()[index];
    }

    throw std::out_of_range("tensor index out of range");
}

const Float32& Tensor::operator[](std::size_t index) const{
    if(index < storage_->size()){
        return storage_->raw_data()[index];
    }

    throw std::out_of_range("tensor index out of range");
}



const Shape& Tensor::shape() const {
    return shape_;
}

long Tensor::numel() const{
    return this->shape_.numel();
}

const Dtype& Tensor::dtype() const{
    return this->dtype_;
}

Backend Tensor::backend() const{
    return storage_->backend();
}

bool Tensor::is_cpu() const{
    return backend() == Backend::CPU;
}

bool Tensor::is_cuda() const{
    return backend() == Backend::CUDA;
}

Tensor& Tensor::fill(Float32 value){
    storage_->fill(value);
    return *this;
}

const Float32 * print_recur(const Shape &shape, const Float32 *data, long times){

    if(shape.dims().size() == 1){
        std::cout << "[";
        for(int j = 0; j < shape.dims()[0]; j++){
            if(j < shape.dims()[0] - 1)
                std::cout << *(data++) << ",";
            else
                std::cout << *(data++);
        }
        std::cout << "]" << "," <<std::endl;
    }else{

        std::vector<long> t_v = std::vector<long>(shape.dims().cbegin() + 1, shape.dims().cend());
        Shape t_s = Shape(std::move(t_v));
        std::cout << "[" << std::endl;
        while(times--) data = print_recur(t_s, data, t_s.dims()[0]);
        std::cout << "]" << "," <<std::endl;
    }

    return data;
}



void print(const Tensor &tensor){
    const Float32 * data = tensor.raw_data();
    std::cout << "(" << std::endl;
    print_recur(tensor.shape(), data , tensor.shape().dims()[0]);
    std::cout << tensor.shape() << ")" << std::endl;
}