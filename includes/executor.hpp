#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "tensor.hpp"
#include "graph.hpp"

#include "operation_fac.hpp"

#include <algorithm>
#include <unordered_map>

class Executor{

    public:
        Executor(const Graph & graph) 
        : e_graph(graph){}

        void set_input(const ValueId v_id, Tensor &&t){
            e_map[v_id] = std::move(t);
        }

        const Tensor &get_tensor(ValueId v_id){
            const auto & it = e_map.find(v_id);

            if(it == e_map.cend()){
                if(!e_graph.value(v_id).initializer().has_value())
                    throw std::runtime_error("value is unavailable");

                return e_graph.value(v_id).initializer().value();
            }
            return it->second;
        }

        const Tensor &get_output(const std::string &name){
            ValueId v_id = e_graph.value_id(name);
            const auto & o = e_graph.outputs();
            if(std::find(o.cbegin(), o.cend(), v_id) == o.cend()){
                throw std::runtime_error(std::format("no varible named {} in executor", name));
            }

            return get_tensor(v_id);
        }

        void run(){

            for(auto i : e_graph.inputs()){
                if(!e_map.contains(i) && !e_graph.value(i).is_initializer()) throw std::runtime_error("error in graph input check");
            }

            for(auto i : e_graph.executor_order()){

                auto op = OperationFactory::create(e_graph.node(i).op_type());

                std::vector<const Tensor*> t;
                for(auto j : e_graph.node(i).inputs()) 
                    t.emplace_back(&get_tensor(j));
                
                // no multi outputs for now
                set_input(e_graph.node(i).outputs()[0], std::move(op->forward(t)));
            }
        }

    private:
        std::unordered_map<ValueId, Tensor> e_map;
        const Graph &e_graph;
};



#endif