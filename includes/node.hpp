#ifndef NODE_HPP
#define NODE_HPP

#include <string>
#include <iostream>
#include <initializer_list>
#include <vector>

enum class OP_TYPE{
    AddOp,
};

class Node{

    public:
        Node(   const std::initializer_list<std::string> &in, 
                const std::initializer_list<std::string> &out,
                const std::string &name,
                const OP_TYPE &op_type)
        :output_(out), input_(in), name_(name), op_type_(op_type){

        }

        const std::vector<std::string>& outputs() const{
            return output_;
        }

        const std::vector<std::string>& inputs() const{
            return input_;
        }

        const std::string& name() const{
            return name_;
        }

        const char* op_type_str() const{
            switch (op_type_){
                case OP_TYPE::AddOp:
                    return "Add";
                default:
                    return "Unknown Type";
            }
            
        }

        OP_TYPE op_type() const{
            return op_type_;
        }

        void show() const{
            std::cout << *this;
        }


    private:
        std::string name_;
        std::vector<std::string> output_;
        std::vector<std::string> input_;
        OP_TYPE op_type_;
        friend std::ostream& operator<<(std::ostream &os, const Node &node);


};

inline std::ostream& operator<<(std::ostream &os, const Node &node){
    os << "Node name: " << node.name_ << std::endl;
    os << "Op type: " << node.op_type_str() << std::endl;
    os << "Inputs: ";

    for(auto &i : node.input_){
        os << i << ", ";
    }

    os << std::endl;

    os << "Outputs: ";
    for(auto &o : node.output_){
        os << o << ", ";
    }

    os << std::endl;  
    return os;    
}

#endif