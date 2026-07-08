#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "tensor.hpp"
#include "graph.hpp"
#include "addop.hpp"
#include "mulop.hpp"
#include "reluop.hpp"
#include "matmulop.hpp"

#include <algorithm>
#include <unordered_map>

class Executor{

    public:
        Executor(const Graph & graph) : e_graph(graph){
        }

        void set_input(const std::string &name, const Tensor &t){
            e_map.insert_or_assign(name, t);
        }

        void set_input(const std::string &name, Tensor &&t){
            e_map.insert_or_assign(name, std::move(t));
        }

        Tensor &get_tensor(const std::string &name) {
            return e_map.at(name);
        }

        Tensor &get_output(const std::string &name){
            auto & o = e_graph.outputs();
            if(std::find(o.cbegin(), o.cend(), name) == o.cend()){
                throw std::runtime_error(std::format("no varible named {} in executor", name));
            }

            return get_tensor(name);
        }

        void run(){
            for(auto &i : e_graph.inputs()){
                if(e_map.find(i) == e_map.cend())
                    throw std::runtime_error("error in graph input check");
            }

            for(auto &i : e_graph.nodes()){
                switch (i.op_type()){
                    case OP_TYPE::AddOp:
                        set_input(i.outputs()[0], AddOp::forward(e_map.at(i.inputs()[0]), e_map.at(i.inputs()[1])));
                        break;

                    case OP_TYPE::MulOp:
                        set_input(i.outputs()[0], MulOp::forward(e_map.at(i.inputs()[0]), e_map.at(i.inputs()[1])));
                        break;

                    case OP_TYPE::ReluOp:
                        set_input(i.outputs()[0], ReluOp::forward(e_map.at(i.inputs()[0])));
                        break;

                    case OP_TYPE::MatMulOp:
                        set_input(i.outputs()[0], MatMulOp::forward(e_map.at(i.inputs()[0]), e_map.at(i.inputs()[1])));
                        break;

                    default:
                        throw std::runtime_error(std::format("unknown op called {}", i.op_type_str()));
                }
            }
        }

    private:
        std::unordered_map<std::string, Tensor> e_map;
        Graph e_graph;
};



#endif