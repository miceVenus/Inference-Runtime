#ifndef IR_NODES_HPP
#define IR_NODES_HPP

#include "ir_types.hpp"
#include <string>
#include <vector>
#include <operation.hpp>


class IRNode{
    public:
        IRNode( 
                NodeId id,
                std::string name,
                OP_TYPE op_type,
                std::vector<ValueId> in, 
                std::vector<ValueId> out)
        :id_(id), name_(std::move(name)), op_type_(op_type), input_(std::move(in)), output_(std::move(out)){}


        NodeId id() const{
            return id_;
        }

        const std::string& name() const{
            return name_;
        }

        OP_TYPE op_type() const{
            return op_type_;
        }

        const std::vector<ValueId>& outputs() const{
            return output_;
        }
        
        const std::vector<ValueId>& inputs() const{
            return input_;
        }

        const char* op_type_str() const{
            switch (op_type_){
                case OP_TYPE::AddOp:
                    return "Add";
                case OP_TYPE::MulOp:
                    return "Mul";
                case OP_TYPE::ReluOp:
                    return "Relu";
                case OP_TYPE::MatMulOp:
                    return "MatMul";
                default:
                    return "Unknown Type";
            }
            
        }

        // void show() const{
        //     std::cout << *this;
        // }

    private:
        NodeId id_;
        std::string name_;
        OP_TYPE op_type_;

        std::vector<ValueId> input_;
        std::vector<ValueId> output_;

};


#endif