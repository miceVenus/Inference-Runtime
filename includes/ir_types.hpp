#ifndef IR_TYPES_HPP
#define IR_TYPES_HPP

#include <cstddef>


// struct NodeId{
//     public:
//         std::size_t index;
//         bool operator==(const NodeId&) const = default;
// };

// struct ValueId{
//     public:
//         std::size_t index;
//         bool operator==(const ValueId&) const = default;
// };

using NodeId = std::size_t;
using ValueId = std::size_t;

#endif