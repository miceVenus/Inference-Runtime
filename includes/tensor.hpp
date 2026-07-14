#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <memory>

#include "shape.hpp"
#include "storage.hpp"
#include "storage_fac.hpp"
#include "backend.hpp"

class Tensor{
    public:

        Tensor(const Shape & shape, Dtype dtype, Backend backend = Backend::CPU) 
        : shape_(shape), storage_(StorageFac::create(shape.numel(), backend)), dtype_(dtype){}
        

        Tensor(const Shape & shape, std::vector<Float32> data, Dtype dtype, Backend backend = Backend::CPU)
        : shape_(shape), storage_(StorageFac::create(std::move(data), backend)), dtype_(dtype){
            if(shape.numel() != storage_->size()){
                throw std::runtime_error("error in data shape");
            }
        }

        Tensor(const Tensor & tensor) 
        : shape_(tensor.shape_), storage_(tensor.storage_->clone()), dtype_(tensor.dtype_){}

        Tensor(Tensor && other) noexcept = default;

        ~Tensor() = default;
        Tensor() = default;

        Tensor& operator=(Tensor && other) noexcept = default;

        Tensor& operator=(const Tensor & tensor) {
            this->storage_ = tensor.storage_->clone();
            this->shape_ = tensor.shape_;
            this->dtype_ = tensor.dtype_;
            return *this;
        }

        Float32 * raw_data(){
            return storage_->raw_data();
        }

        const Float32 * raw_data() const{
            return storage_->raw_data();
        }

        Float32& operator[](std::size_t index){
            if(index < storage_->size()){
                return storage_->raw_data()[index];
            }

            throw std::out_of_range("tensor index out of range");
        }

        const Float32& operator[](std::size_t index) const{
            if(index < storage_->size()){
                return storage_->raw_data()[index];
            }

            throw std::out_of_range("tensor index out of range");
        }



        const Shape& shape() const {
            return shape_;
        }

        long numel() const{
            return this->shape_.numel();
        }

        const Dtype& dtype() const{
            return this->dtype_;
        }

        Backend backend() const{
            return storage_->backend();
        }

        bool is_cpu() const{
            return backend() == Backend::CPU;
        }

        bool is_cuda() const{
            return backend() == Backend::CUDA;
        }

    private:
        Shape shape_;
        std::unique_ptr<Storage> storage_;
        Dtype dtype_ = Dtype::Float32;
};


inline const char * dtype2str(const Dtype &dtype){
    switch (dtype){

        case Dtype::Float32:
            return "Float 32";
        
        default:
            return "Unknown Type";
    }
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

#endif