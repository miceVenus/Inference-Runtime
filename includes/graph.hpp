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
};




#endif
