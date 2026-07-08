#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "tensor.hpp"
#include "graph.hpp"

#include "operation_fac.hpp"

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

                auto op = OperationFactory::create(i.op_type());

                std::vector<const Tensor*> t;
                for(auto & j : i.inputs()) t.emplace_back(&e_map.at(j));
                set_input(i.outputs()[0], std::move(op->forward(t)));

            }
        }

    private:
        std::unordered_map<std::string, Tensor> e_map;
        Graph e_graph;
};



#endif