#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "tensor.hpp"
#include "graph.hpp"
#include "storage_pool.hpp"

#include "operation_fac.hpp"

#include <algorithm>
#include <unordered_map>
#include <cassert>

class Executor{

    public:
        Executor(const Graph & graph) 
        : e_graph_(graph){
            storage_planning();
            storage_creating();
        }

        void set_input(const ValueId v_id, Tensor &&t){

            assert(e_graph_.desc(v_id) == TensorDesc(t));

            const auto & i = e_graph_.inputs();
            // const auto & w = e_graph.initializers();

            const auto & it_i = std::find(i.cbegin(), i.cend(), v_id);
            // const auto & it_w = std::find(w.cbegin(), w.cend(), v_id);

            if(it_i != i.cend()){
                e_map_[v_id] = std::move(t);
            }else{
                throw std::runtime_error("set input received v_id which is not in graph input");
            }
            
        }

        // static graph tensor is the first so set_input wouldn`t override static data
        const Tensor &get_tensor(ValueId v_id){
            if(e_graph_.value(v_id).is_initializer()){
                return e_graph_.value(v_id).initializer().value();
            }else{
                const auto & it = e_map_.find(v_id);
                if(it != e_map_.cend()){
                    return it->second;
                }else{
                    throw std::runtime_error("value is unavailable");
                }
            }
        }

        const Tensor &get_output(const std::string &name){
            ValueId v_id = e_graph_.value_id(name);
            const auto & o = e_graph_.outputs();
            if(std::find(o.cbegin(), o.cend(), v_id) == o.cend()){
                throw std::runtime_error(std::format("no varible named {} in executor", name));
            }

            return get_tensor(v_id);
        }

        const StorageId get_b_id(ValueId v_id) const{
            return v_b_map_.at(v_id);
        }

        const std::vector<StorageDesc>& get_buffers() const{
            return buffers_;
        }

        void run(){

            reset_storage();

            for(auto i : e_graph_.inputs()){
                if(!e_map_.contains(i) && !e_graph_.value(i).is_initializer()) throw std::runtime_error("error in graph input check");
            }

            const std::vector<NodeId>& exe_queue = e_graph_.executor_order();
            for(size_t index = 0; index < exe_queue.size(); index++){

                NodeId i = exe_queue[index];


                auto op = OperationFactory::create(e_graph_.node(i));

                std::vector<const Tensor*> t;
                for(auto j : e_graph_.node(i).inputs()) 
                    t.emplace_back(&get_tensor(j));

                
                ValueId v_id = e_graph_.node(i).outputs()[0];
                
                TensorDesc t_T = e_graph_.desc(v_id);
                Tensor out(t_T, mem_pool_.get(buffers_[get_b_id(v_id)]));


                // no multi outputs for now
                set_tensor(v_id, std::move(op->forward(t, out.fill(0))));

                for(auto j : e_graph_.node(i).inputs()){
                    if(v_b_map_.contains(j)){
                        Tensor & t = e_map_[j];
                        auto tmp = e_graph_.life_span(j).second;
                        if(e_graph_.life_span(j).second == index){
                            t.fill(0);
                            mem_pool_.put(t.move_storage(), v_b_map_[j]);
                            e_map_.erase(j);
                        }
                    }
                }
            }
        }

    private:
        StoragePool mem_pool_;
        std::vector<StorageDesc> buffers_;
        // v_b_map_ only care tensors created by nodes 
        // except initilizers and inputs 
        std::unordered_map<ValueId, StorageId> v_b_map_;
        std::unordered_map<ValueId, Tensor> e_map_;
        const Graph &e_graph_;

        void set_tensor(const ValueId v_id, Tensor &&t){
                e_map_[v_id] = std::move(t); 
        }

        void storage_planning(){

            // assume here queue is pure without dead node
            const std::vector<NodeId> & exe_queue = e_graph_.executor_order();

            for(size_t index_1 = 0; index_1 < exe_queue.size(); index_1++){
                NodeId n_id_1 = exe_queue[index_1];
                for(auto v_id_1 : e_graph_.node(n_id_1).outputs()){

                    // there need a better planning

                    bool founded = false;

                    for(auto & it : buffers_){
                        if(it.ava_after < e_graph_.life_span(v_id_1).first){
                            const TensorDesc & v_desc = e_graph_.desc(v_id_1);
                            if(it.size_bytes == (v_desc.shape_.numel() * dtype_size(v_desc.dtype_)) && it.backend == v_desc.backend_){
                                v_b_map_[v_id_1] = it.id;
                                it.ava_after = e_graph_.life_span(v_id_1).second;
                                founded = true;
                                break;
                            }
                        }
                    }

                    if(!founded){
                        v_b_map_[v_id_1] = buffers_.size();
                        StorageDesc b_desc(e_graph_.desc(v_id_1));
                        b_desc.id = buffers_.size();
                        b_desc.ava_after = e_graph_.life_span(v_id_1).second;
                        buffers_.push_back(std::move(b_desc));
                    }
                    
                }
            }
        }


        void storage_creating(){
            for(auto & i : buffers_){
                mem_pool_.add(i);
            }
        }

        void reset_storage(){
            for(auto & it : v_b_map_){
                if(e_map_.contains(it.first)){
                    mem_pool_.put(e_map_[it.first].move_storage(), it.second);
                    e_map_.erase(it.first);
                }
            }
        }
};



#endif
