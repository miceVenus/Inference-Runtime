#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "node.hpp"
#include "value.hpp"
#include "ir_nodes.hpp"
#include "operation_fac.hpp"

#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <stdexcept>
#include <format>

class Graph{

    public:
        Graph(  
            std::string name,
            std::vector<IRNode>     nodes,
            std::vector<Value>      values,
            std::vector<ValueId>    inputs,
            std::vector<ValueId>    outputs,
            std::vector<ValueId>    initializers,
            std::unordered_map<std::string, NodeId>     node_ids = {},
            std::unordered_map<std::string, ValueId>    value_ids = {})
            : name_(std::move(name)), nodes_(std::move(nodes)), values_(std::move(values)),
            inputs_(std::move(inputs)), outputs_(std::move(outputs)),initializers_(std::move(initializers)),
            node_ids_(std::move(node_ids)), value_ids_(std::move(value_ids)){

            topological_sort();
            validate();
            desc_deduce();
            life_span_deduce();
        }

        ValueId value_id(const std::string& name) const {
            return value_ids_.at(name);
        }

        NodeId node_id(const std::string& name) const {
            return node_ids_.at(name);
        }

        Value& value(ValueId id) {
            return values_.at(id);
        }

        const IRNode& node(NodeId id) const{
            return nodes_.at(id);
        }

        const TensorDesc& desc(ValueId id) const{

            if(!value(id).desc().has_value())
                throw std::runtime_error(std::format("value {} desc is not prepared", value(id).name()));
            return value(id).desc().value();
        }

        const Value& value(ValueId id) const {
            return values_.at(id);
        }

        const std::vector<ValueId>& inputs() const{
            return inputs_;
        }
        const std::vector<ValueId>& outputs() const{
            return outputs_;
        }

        const std::vector<NodeId>& executor_order() const{
            return executor_order_;
        }

        const std::vector<NodeId>& initializers() const{
            return initializers_;
        }

        const std::pair<size_t, size_t> life_span(ValueId v_id) const{
            return value(v_id).life_span();
        }

    private:

        std::vector<IRNode>     nodes_;
        std::vector<Value>      values_;
        std::vector<ValueId>    inputs_;
        std::vector<ValueId>    outputs_;
        std::vector<ValueId>    initializers_;

        std::unordered_map<std::string, NodeId>     node_ids_;
        std::unordered_map<std::string, ValueId>    value_ids_;

        std::vector<NodeId>    executor_order_;
        
        std::string name_;

        void topological_sort(){

            std::vector<std::vector<NodeId>> adjacency(nodes_.size());
            std::vector<std::size_t> indegree(nodes_.size(), 0);
            std::queue<NodeId> ready;

            for(auto & node : nodes_){
                for(auto & v_id : node.outputs()){
                    for(auto & consumer : value(v_id).consumer())
                        adjacency[node.id()].emplace_back(consumer);
                }

                for(auto & v_id : node.inputs()){
                    if(value(v_id).producer().has_value())
                        indegree[node.id()]++;
                }
            }

            for(std::size_t j = 0; j < indegree.size(); j++)
                if(indegree[j] == 0) ready.push({j});

            
            while(!ready.empty()){

                NodeId i = ready.front();
                executor_order_.emplace_back(i);

                for(auto k : adjacency[i]){
                    if(--indegree[k] == 0) ready.push(k);
                }

                ready.pop();
            }

            if(executor_order_.size() != nodes_.size()) throw std::runtime_error("error in topological sort");
        }



        void validate() {

            std::unordered_set<ValueId> available_set;

            available_set.insert(inputs_.cbegin(), inputs_.cend());
            available_set.insert(initializers_.cbegin(), initializers_.cend());
            // for(auto &i : nodes_)
            //     available_set.insert(i.outputs().cbegin(), i.outputs().cend());

            for(auto &i : executor_order_){                
                for(auto &j : node(i).inputs()){
                    if(!available_set.contains(j))
                        throw std::runtime_error("input can not be deduced by prev out");
                }

                for(auto &j : node(i).outputs()){
                    auto [it, inserted] = available_set.insert(j);
                    if(!inserted) throw std::runtime_error("duplicated output");
                }
                
            }

            for(auto &i : outputs_){
                if(!available_set.contains(i)) 
                    throw std::runtime_error("graph outputs can not be deduced");
            }

        }

        void desc_deduce(){

            for(auto i : inputs_){
                if(!value(i).desc().has_value()){
                    throw std::runtime_error("graph input need tensor describation for static deduce");
                }
            }

            for (auto i : initializers_) {
                TensorDesc initializer_desc(value(i).initializer().value());

                if (!value(i).desc().has_value()) 
                    value(i).set_desc(initializer_desc);

                else if (value(i).desc().value() != initializer_desc) {
                    throw std::runtime_error("initializer and input descriptions conflict");
                }
            }

            // for(auto i : initializers_){
            //     value(i).set_desc(TensorDesc(value(i).initializer().value()));
            // }


            for(auto i : executor_order_){

                auto op = OperationFactory::create(node(i).op_type());

                std::vector<TensorDesc> t;
                for(auto j : node(i).inputs()){
                    if(!value(j).desc().has_value()){
                      throw std::runtime_error("static shape deduce fail because lack of input desc");  
                    }
                    t.push_back(value(j).desc().value());
                }


                // To Be Care I assumed that Op`s result must be size(1);
                for(auto j : node(i).outputs()){
                    value(j).set_desc(std::move(op->forward_T(t)));
                }
            }

            for(const auto & i : values_){
                if(!i.desc().has_value ()){
                    throw std::runtime_error(std::format("value : {} has no desc after deduce", i.name()));  
                }

                std::cout << i.desc().value().shape_ << std::endl;
            }
        }

        void life_span_deduce(){

            std::unordered_map<ValueId, std::size_t> v_id_consumer;

            for(size_t index = 0; index < executor_order_.size(); index++){

                ValueId i = executor_order_[index];
                
                for(auto j : node(i).inputs()){
                    if(!v_id_consumer.contains(j)){
                        v_id_consumer[j] = value(j).consumer().size();
                    }
                }

                for(auto j : node(i).outputs()){
                    if(!v_id_consumer.contains(j)){
                        v_id_consumer[j] = value(j).consumer().size();
                    }

                    value(j).set_birth(index);
                    value(j).set_death(index);
                }

                for(auto j : outputs()){
                    value(j).set_death(executor_order_.size());
                }

                for(auto j : node(i).inputs()){
                    if(--v_id_consumer[j]){
                    }else{

                        const auto & wt = initializers();
                        if(std::find(wt.cbegin(), wt.cend(), j) != wt.cend()){
                            continue;
                        }

                        const auto & ot = outputs();
                        if(std::find(ot.cbegin(), ot.cend(), j) != ot.cend()){
                            continue;
                        }

                        const auto & in = inputs();
                        if(std::find(in.cbegin(), in.cend(), j) != in.cend()){
                            continue;
                        }

                        value(j).set_death(index);
                    }
                }

            }

        }
};




#endif
