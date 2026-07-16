#include "dtype.hpp"
#include <stdexcept> 

std::size_t dtype_size(Dtype dtype){
    switch (dtype){
        case Dtype::Float32:
            return sizeof(Float32);

        default:
            throw std::invalid_argument("Unsupported dtype");
    }
}