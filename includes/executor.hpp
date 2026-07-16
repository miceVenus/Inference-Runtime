#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "tensor.hpp"
#include "graph.hpp"
#include "storage_pool.hpp"

#include "operation_fac.hpp"

#include <algorithm>
#include <unordered_map>

class Executor{

    public:
        Executor(const Graph & graph) 
        : e_graph(graph){}

        void set_input(const ValueId v_id, Tensor &&t){

            const auto & i = e_graph.inputs();
            // const auto & w = e_graph.initializers();

            const auto & it_i = std::find(i.cbegin(), i.cend(), v_id);
            // const auto & it_w = std::find(w.cbegin(), w.cend(), v_id);

            if(it_i != i.cend()){
                e_map[v_id] = std::move(t);
            }else{
                throw std::runtime_error("set input received v_id which is not in graph input");
            }
            
        }

        // static graph tensor is the first so set_input wouldn`t override static data
        const Tensor &get_tensor(ValueId v_id){
            if(e_graph.value(v_id).is_initializer()){
                return e_graph.value(v_id).initializer().value();
            }else{
                const auto & it = e_map.find(v_id);
                if(it != e_map.cend()){
                    return it->second;
                }else{
                    throw std::runtime_error("value is unavailable");
                }
            }
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

            std::unordered_map<ValueId, std::size_t> v_id_consumer;

            for(auto i : e_graph.inputs()){
                if(!e_map.contains(i) && !e_graph.value(i).is_initializer()) throw std::runtime_error("error in graph input check");
            }

            for(auto i : e_graph.executor_order()){

                for(auto j : e_graph.node(i).inputs()){
                    if(!v_id_consumer.contains(j)){
                        v_id_consumer[j] = e_graph.value(j).consumer().size();
                    }
                }

                for(auto j : e_graph.node(i).outputs()){
                    if(!v_id_consumer.contains(j)){
                        v_id_consumer[j] = e_graph.value(j).consumer().size();
                    }
                }


                auto op = OperationFactory::create(e_graph.node(i).op_type());

                std::vector<const Tensor*> t;
                for(auto j : e_graph.node(i).inputs()) 
                    t.emplace_back(&get_tensor(j));

                TensorDesc t_T = op->forward_T(t);

                Tensor out(t_T, mem_pool.get(t_T));


                // no multi outputs for now
                set_tensor(e_graph.node(i).outputs()[0], std::move(op->forward(t, out.fill(0))));


                for(auto j : e_graph.node(i).inputs()){
                    if(--v_id_consumer[j]){
                    }else{

                        const auto & wt = e_graph.initializers();
                        if(std::find(wt.cbegin(), wt.cend(), j) != wt.cend())
                            continue;

                        const auto & ot = e_graph.outputs();
                        if(std::find(ot.cbegin(), ot.cend(), j) != ot.cend())
                            continue;

                        const auto & in = e_graph.inputs();
                        if(std::find(in.cbegin(), in.cend(), j) != in.cend())
                            continue;

                        mem_pool.put(e_map[j]);
                        e_map.erase(j);
                    }
                }

                for(auto j : e_graph.node(i).outputs()){
                    if(!v_id_consumer[j]){
                        const auto & ot = e_graph.outputs();
                        if(std::find(ot.cbegin(), ot.cend(), j) != ot.cend())
                            continue;

                        mem_pool.put(e_map[j]);
                        e_map.erase(j);
                    }
                }

            }
        }

    private:
        StoragePool mem_pool;
        std::unordered_map<ValueId, Tensor> e_map;
        const Graph &e_graph;

        void set_tensor(const ValueId v_id, Tensor &&t){
                e_map[v_id] = std::move(t); 
        }
};



#endif