#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>

#include "shape.hpp"

enum class Backend{
    CPU,
    CUDA
};


class Tensor{
    public:
        Tensor(const Shape & shape, const Dtype &dtype) 
        : shape_(shape), dtype_(dtype){
            data_.assign(shape.numel(), 0);
        }
        
        Tensor(const Shape & shape, std::initializer_list<Float32> data, const Dtype &dtype) 
        : shape_(shape), data_(data), dtype_(dtype){
            if(shape.numel() != data_.size()){
                throw std::runtime_error("error in data shape");
            }
        }

        Tensor(const Shape & shape, std::vector<Float32> && data, const Dtype dtype)
        : shape_(shape), data_(std::move(data)), dtype_(dtype){
            if(shape.numel() != data_.size()){
                throw std::runtime_error("error in data shape");
            }
        }

        Tensor(const Shape & shape, const Dtype dtype, const Backend backend) 
        : shape_(shape), dtype_(dtype), backend_(backend){
            data_.assign(shape.numel(), 0);
        }
        
        Tensor(const Shape & shape, std::initializer_list<Float32> data, const Dtype &dtype, const Backend backend) 
        : shape_(shape), data_(data), dtype_(dtype), backend_(backend){
            if(shape.numel() != data_.size()){
                throw std::runtime_error("error in data shape");
            }
        }

        Tensor(const Shape & shape, std::vector<Float32> && data, const Dtype &dtype, const Backend backend)
        : shape_(shape), data_(std::move(data)), dtype_(dtype), backend_(backend){
            if(shape.numel() != data_.size()){
                throw std::runtime_error("error in data shape");
            }
        }


        Tensor(const Tensor & tensor) 
        : shape_(tensor.shape_), data_(tensor.data_), dtype_(tensor.dtype_), backend_(tensor.backend_){}

        Tensor(Tensor && other) noexcept
        : shape_(std::move(other.shape_)), data_(std::move(other.data_)), dtype_(other.dtype_), backend_(other.backend_){}

        ~Tensor() = default;

        Tensor& operator=(Tensor && other) noexcept {

            if(this != &other){
                data_  = std::move(other.data_);
                shape_ = std::move(other.shape_);
                dtype_ = other.dtype_;
                backend_ = other.backend_;
            }

            return *this;
        }
        Tensor& operator=(const Tensor & tensor) {
            this->data_.assign(tensor.data_.cbegin(), tensor.data_.cend());
            this->shape_ = tensor.shape_;
            this->dtype_ = tensor.dtype_;
            this->backend_ = tensor.backend_;

            return *this;
        }

        Float32 * raw_data(){
            return data_.data();
        }

        const Float32 * raw_data() const{
            return data_.data();
        }

        Float32& operator[](std::size_t index){
            return data_.at(index);
        }

        const Float32& operator[](std::size_t index) const{
            return data_.at(index);
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

        bool is_cpu() const{
            return backend_ == Backend::CPU;
        }

        bool is_cuda() const{
            return backend_ == Backend::CUDA;
        }

        Backend backend() const{
            return backend_;
        }

    private:
        Shape shape_;
        std::vector<Float32> data_;
        Dtype dtype_ = Dtype::Float32;
        Backend backend_ = Backend::CPU;
};


inline const char * dtype2str(const Dtype &dtype){
    switch (dtype){

        case Dtype::Float32:
            return "Float 32";
        
        default:
            return "Unknown Type";
    }
}

void print_recur(const Shape &shape, auto &it, long times){

    if(shape.dims().size() == 1){
        std::cout << "[";
        for(int j = 0; j < shape.dims()[0]; j++){
            if(j < shape.dims()[0] - 1)
                std::cout << *(it++) << ",";
            else
                std::cout << *(it++);
        }
        std::cout << "]" << "," <<std::endl;
    }else{

        std::vector<long> t_v = std::vector<long>(shape.dims().cbegin() + 1, shape.dims().cend());
        Shape t_s = Shape(std::move(t_v));
        std::cout << "[" << std::endl;
        while(times--) print_recur(t_s, it, t_s.dims()[0]);
        std::cout << "]" << "," <<std::endl;
    }
}



void print(const Tensor &tensor){
    auto it = tensor.data().cbegin();
    std::cout << "(" << std::endl;
    print_recur(tensor.shape(), it , tensor.shape().dims()[0]);
    std::cout << tensor.shape() << ")" << std::endl;
}

#endif