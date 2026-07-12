#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "node.hpp"
#include "operation_fac.hpp"

#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <stdexcept>
#include <format>

class Graph{

    public:
        Graph(  
            const std::initializer_list<Node> &nodes,
            const std::initializer_list<std::string> &inputs,
            const std::initializer_list<std::string> &outputs,
            const std::string &name) :
            inputs_(inputs), outputs_(outputs), name_(name){

            for(auto &i : nodes){
                if(node_name_map_id_.contains(i.name())) throw std::runtime_error(std::format("same node name {} when graph building", i.name()));
                node_name_map_id_[i.name()] = node_name_map_id_.size();
            }

            for(auto &i : nodes){
                for(auto &j : i.outputs()){
                    if(value_name_map_producer_.contains(j)) throw std::runtime_error(std::format("same output name {} when graph building", j));
                    value_name_map_producer_[j] = node_name_map_id_.at(i.name());
                }

                for(auto &j : i.inputs()){
                    value_name_map_consumer_[j].emplace_back(node_name_map_id_.at(i.name()));
                }
            }

            validate_set(topological_sort(nodes));

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

        std::unordered_map<std::string, node_id> node_name_map_id_;
        std::unordered_map<std::string, node_id> value_name_map_producer_;
        std::unordered_map<std::string, std::vector<node_id>> value_name_map_consumer_;
        std::string name_;

        std::vector<Node> topological_sort(const std::vector<Node> &nodes){

            std::vector<std::vector<node_id>> adjacency(nodes.size());
            std::vector<std::size_t> indegree(nodes.size(), 0);
            std::vector<node_id> executor_order;
            std::queue<node_id> ready;

            for(std::size_t i = 0; i < nodes.size(); i++){
                for(auto & j : nodes[i].outputs()){
                    for(const auto & k : value_name_map_consumer_[j])
                        adjacency[i].emplace_back(k);
                }
                for(auto & j : nodes[i].inputs()){
                    if(value_name_map_producer_.contains(j)) indegree[i]++;
                }
            }

            for(std::size_t j = 0; j < indegree.size(); j++)
                if(indegree[j] == 0) ready.push(j);

            
            while(!ready.empty()){

                node_id index = ready.front();
                executor_order.emplace_back(index);

                for(auto k : adjacency[index]){
                    if(--indegree[k] == 0) ready.push(k);
                }

                ready.pop();
            }

            if(executor_order.size() != nodes.size()) throw std::runtime_error("error in topological sort");

            std::vector<Node> t_vec;

            for(auto &i : executor_order){
                t_vec.emplace_back(nodes[i]);
            }

            return t_vec;

        }



        void validate_set(std::vector<Node> && nodes) {

            std::unordered_set<std::string> available_set;

            available_set.insert(inputs().cbegin(), inputs().cend());
            // for(auto &i : nodes_)
            //     available_set.insert(i.outputs().cbegin(), i.outputs().cend());

            for(auto &i : nodes){

                for(auto &j : i.inputs()){
                    if(available_set.find(j) == available_set.cend()) 
                        throw std::runtime_error("input can not be deduced by prev out");
                }

                for(auto &j : i.outputs()){
                    auto [it, inserted] = available_set.insert(j);
                    if(!inserted) throw std::runtime_error("duplicated output");
                }

                validate_node_schema(i);

                
            }

            for(auto &i : outputs()){
                if(available_set.find(i) == available_set.cend()) 
                    throw std::runtime_error("graph outputs can not be deduced");
            }

            nodes_ = std::move(nodes);
        }

        void validate_node_schema(const Node &node) const{

            std::pair<std::size_t, std::size_t> io_pair = OperationFactory::schema(node.op_type());
            if(node.inputs().size() != io_pair.first)
                throw std::runtime_error(std::format("{} expect {} inputs but received {}", node.op_type_str(), io_pair.first, node.inputs().size()));
            if(node.outputs().size() != io_pair.second)
                throw std::runtime_error(std::format("{} expect {} outputs but generated {}", node.op_type_str(), io_pair.second, node.outputs().size()));

        }
};




#endif
