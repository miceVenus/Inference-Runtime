#ifndef OP_ATTRIBUTE_HPP
#define OP_ATTRIBUTE_HPP

#include "addop.hpp"
#include "convop.hpp"
#include "matmulop.hpp"
#include "mulop.hpp"
#include "reluop.hpp"


#include <variant>

class NoParam{

};

using OpParam = std::variant<
    NoParam,
    ConvParam
>;

#endif