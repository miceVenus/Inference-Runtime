#ifndef DTYPE_HPP
#define DTYPE_HPP

#include <utility>
using Float32 = float;

enum class Dtype{
    Float32,
};

std::size_t dtype_size(Dtype dtype);


#endif
