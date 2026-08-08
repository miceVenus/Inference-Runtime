#ifndef OPERATION_FAC_HPP
#define OPERATION_FAC_HPP

#include "addop.hpp"
#include "mulop.hpp"
#include "matmulop.hpp"
#include "reluop.hpp"
#include "convop.hpp"

#include "operation.hpp"
#include <memory>
#include <format>
#include <utility>

class OperationFactory{
    public:
        static std::unique_ptr<Operation> create(const IRNode & node){
            switch (node.op_type()){
                case OP_TYPE::AddOp:
                    return std::make_unique<AddOp>();

                case OP_TYPE::MulOp:
                    return std::make_unique<MulOp>();

                case OP_TYPE::MatMulOp:
                    return std::make_unique<MatMulOp>();

                case OP_TYPE::ReluOp:
                    return std::make_unique<ReluOp>();

                case OP_TYPE::ConvOp:
                    return std::make_unique<ConvOp>(std::get<ConvParam>(node.attribute()));

                default:
                    throw std::runtime_error(std::format("unknown OP_TYPE in operation factory"));
            }
        }

        static std::pair<std::size_t, std::size_t> schema(OP_TYPE type){
            
            switch (type){
                case OP_TYPE::AddOp:
                    return std::pair<std::size_t, std::size_t>(2, 1);

                case OP_TYPE::MulOp:
                    return std::pair<std::size_t, std::size_t>(2, 1);

                case OP_TYPE::ReluOp:
                    return std::pair<std::size_t, std::size_t>(1, 1);

                case OP_TYPE::MatMulOp:
                    return std::pair<std::size_t, std::size_t>(2, 1);
                
                case OP_TYPE::ConvOp:
                    return std::pair<std::size_t, std::size_t>(3, 1);
                
            
            default:
                throw std::runtime_error(std::format("unknown OP_TYPE in operation factory when schema"));
            }
        }
};

#endif