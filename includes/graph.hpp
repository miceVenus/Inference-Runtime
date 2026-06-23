#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "node.hpp"
#include <unordered_set>
#include <stdexcept>
#include <format>

class Graph{

    public:
        Graph(  
            const std::initializer_list<Node> &nodes,
            const std::initializer_list<std::string> &inputs,
            const std::initializer_list<std::string> &outputs,
            const std::string &name) :
            nodes_(nodes), inputs_(inputs), outputs_(outputs), name_(name){

            validate();

        }

        const std::vector<Node> &nodes() const{
            return nodes_;
        }

        const std::vector<std::string> &inputs() const{
            return inputs_;
        }

        const std::vector<std::string> &outputs() const{
            return outputs_;
        }

        void show_graph_info() const{

            std::ostream &os = std::cout;

            os << "name : " << name_ << std::endl;

            os << "nodes : " << nodes_.size() << std::endl;

            for(auto &i : nodes_){
                os << i << ", " << std::endl;
            }

            os << "Inputs: ";

            for(auto &i : inputs_){
                os << i << ", ";
            }

            os << std::endl;

            os << "Outputs: ";
            for(auto &o : outputs_){
                os << o << ", ";
            }

        }


    private:
        std::vector<Node> nodes_;
        std::vector<std::string> inputs_;
        std::vector<std::string> outputs_;
        std::string name_;


        void validate() const{

            std::vector<std::string> c_in;
            std::vector<std::string> p_out;

            std::unordered_set<std::string> available_set;

            available_set.insert(inputs().cbegin(), inputs().cend());
            // for(auto &i : nodes_)
            //     available_set.insert(i.outputs().cbegin(), i.outputs().cend());

            for(auto &i : nodes_){
                c_in = i.inputs();

                for(auto &j : c_in){
                    if(available_set.find(j) == available_set.cend()) 
                        throw std::runtime_error("input can not be deduced by prev out");
                }
                    
                p_out = i.outputs();

                for(auto &j : p_out){
                    auto [it, inserted] = available_set.insert(j);
                    if(!inserted) throw std::runtime_error("duplicated output");
                }

                validate_node_schema(i);

                
            }

            for(auto &i : outputs()){
                if(available_set.find(i) == available_set.cend()) 
                    throw std::runtime_error("graph outputs can not be deduced");
            }

        }

        void validate_node_schema(const Node &node) const{

            switch (node.op_type()){
                case OP_TYPE::AddOp:
                    if(node.inputs().size() != 2) 
                        throw std::runtime_error(std::format("AddOp expect 2 inputs but received {}", node.inputs().size()));
                    if(node.outputs().size() != 1) 
                        throw std::runtime_error(std::format("AddOp expect 1 outputs but generated {}", node.outputs().size()));
                    break;
            
            default:
                throw std::runtime_error(std::format("Unknown operation {}", node.op_type_str()));
            }

        }


        // static bool check_param(const std::vector<std::string> & a, const std::vector<std::string> & b){
        //     if(a.size() != b.size()) return false;

        //     for(int i = 0; i < a.size(); i++){
        //         if(a[i] != b[i]) return false;
        //     }

        //     return true;
        // }
};




#endif