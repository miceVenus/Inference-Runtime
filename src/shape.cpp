#include "shape.hpp"


std::ostream& operator<<(std::ostream& os, const Shape& shape){
    os << "[";
    for(auto &dim : shape.dims_) os << dim << ", ";
    os << "]";
    return os;
}