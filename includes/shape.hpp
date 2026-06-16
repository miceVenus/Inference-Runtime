#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <vector>
#include <initializer_list>
#include <iostream>

#include "dtype.hpp"

class Shape{

    public:
        Shape(std::initializer_list<int64_t> const &dims) : dims_(dims){}
        Shape(std::vector<int64_t> const &dims) : dims_(dims){}

        int64_t numel() const{
            int64_t temp = 1;
            for(auto &dim : dims_) temp *= dim;
            return temp;
        }

        Shape operator=(const Shape &shape) const{
            return Shape(shape.dims_);
        }

        const std::vector<int64_t> &dims() const{
            return dims_;
        }

        friend std::ostream& operator<<(std::ostream& os, const Shape& shape);

    private:
        std::vector<int64_t> dims_;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape){

    os << "[";
    for(auto &dim : shape.dims_) os << dim << ", ";
    os << "]";
    return os;
}


#endif