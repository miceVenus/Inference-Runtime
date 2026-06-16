#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>

#include "shape.hpp"

class Tensor{
    public:
        Tensor(const Shape & shape) : shape_(shape){
            data_.assign(shape.numel(), 0);
        }
        
        Tensor(const Shape & shape, const std::initializer_list<float32_t> data) : 
        shape_(shape), data_(data){
            if(shape.numel() != data.size()){
                throw std::runtime_error("error in data shape");
            }
        }

        Tensor(const Tensor & tensor) : shape_(tensor.shape_), data_(tensor.data_){}

        Tensor operator=(const Tensor & tensor) const{
            Tensor n_tensor = Tensor(tensor);
            return n_tensor;
        }

        const Shape& shape() const {
            return shape_;
        }

        const std::vector<float32_t>& data() const{
            return data_;
        }

        const int64_t numel() const{
            return this->shape_.numel();
        }

    private:
        Shape shape_;
        std::vector<float32_t> data_;
};

void print(const Tensor &tensor){
    std::cout << "(";
    auto data = tensor.data().cbegin();
    int64_t column_size = tensor.shape().dims().back();
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