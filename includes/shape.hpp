#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <vector>
#include <initializer_list>
#include <iostream>

#include "dtype.hpp"

class Shape{

    public:
        Shape(std::initializer_list<long> const &dims) : dims_(dims){}
        Shape(std::vector<long> const &dims) : dims_(dims){}

        Shape(const Shape & shape)
        : dims_(shape.dims_){}

        Shape(Shape && shape) noexcept
        : dims_(std::move(shape.dims_)){}

        ~Shape() = default;
        Shape() = default;


        long numel() const{
            long temp = 1;
            for(auto &dim : dims_) temp *= dim;
            return temp;
        }

        Shape& operator=(const Shape &shape) {
            this->dims_ = shape.dims_;
            return *this;
        }

        Shape& operator=(Shape && shape) {
            if(this != &shape){
                this->dims_ = std::move(shape.dims_);
            }
            return *this;
        }

        bool operator==(const Shape &shape) const{
            if(dims_.size() != shape.dims_.size()) return false;

            for(int i = 0; i < dims_.size(); i++){
                if(dims_[i] != shape.dims_[i]) 
                    return false;
            }
            
            return true;
        }

        bool operator!=(const Shape &shape) const{
            return !(*this == shape);
        }

        const std::vector<long> &dims() const{
            return dims_;
        }

        friend std::ostream& operator<<(std::ostream& os, const Shape& shape);

    private:
        std::vector<long> dims_;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape){

    os << "[";
    for(auto &dim : shape.dims_) os << dim << ", ";
    os << "]";
    return os;
}


#endif