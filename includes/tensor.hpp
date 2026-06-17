#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>

#include "shape.hpp"

class Tensor{
    public:
        Tensor(const Shape & shape, const Dtype &dtype) 
        : shape_(shape), dtype_(dtype){
            data_.assign(shape.numel(), 0);
        }
        
        Tensor(const Shape & shape, const std::initializer_list<Float32> data, const Dtype &dtype) 
        : shape_(shape), data_(data), dtype_(dtype){
            if(shape.numel() != data.size()){
                throw std::runtime_error("error in data shape");
            }
        }

        Tensor(const Shape & shape, std::vector<Float32> data, const Dtype &dtype) 
        : shape_(shape), data_(std::move(data)), dtype_(dtype){
            if(shape.numel() != data_.size()){
                throw std::runtime_error("error in data shape");
            }
        }

        Tensor(const Tensor & tensor) : shape_(tensor.shape_), data_(tensor.data_), dtype_(tensor.dtype_){}

        Tensor& operator=(const Tensor & tensor) {
            this->data_.assign(tensor.data_.cbegin(), tensor.data_.cend());
            this->shape_ = tensor.shape_;
            this->dtype_ = tensor.dtype_;

            return *this;
        }

        const Shape& shape() const {
            return shape_;
        }

        const std::vector<Float32>& data() const{
            return data_;
        }

        long numel() const{
            return this->shape_.numel();
        }

        const Dtype& dtype() const{
            return this->dtype_;
        }

    private:
        Shape shape_;
        std::vector<Float32> data_;
        Dtype dtype_;
};


inline const char * dtype2str(const Dtype &dtype){
    switch (dtype){

        case Dtype::Float32:
            return "Float 32";
        
        default:
            return "Unknown Type";
    }
}


void print(const Tensor &tensor){
    std::cout << "(";
    auto data = tensor.data().cbegin();
    long column_size = tensor.shape().dims().back();
    std::cout << "[" << std::endl;
    for(auto i = tensor.shape().dims().cend() - 1; i != tensor.shape().dims().cbegin(); i--){
        std::cout << "[";
        for(int j = 0; j < column_size; j++){
            std::cout << *(data++) << ", ";
        }
        std::cout << "]" << "," <<std::endl;
    }
    std::cout << "], " << tensor.shape() << ")" << std::endl;
}


#endif