#ifndef GRAPH_BUILDER_HPP
#define GRAPH_BUILDER_HPP

#include "ir_types.hpp"
#include "ir_nodes.hpp"
#include "value.hpp"
#include "node.hpp"
#include "operation_fac.hpp"
#include "tensor.hpp"
#include "graph.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class GraphBuilder{
    public:

        Graph build(
            std::string name,
            const std::vector<Node> & nodes,
            const std::vector<std::string> &inputs,
            const std::vector<std::string> &outputs,
            std::unordered_map<std::string, Tensor> &&initializer){
            set_inputs(inputs);
            set_initializer(std::move(initializer));
            set_outputs(outputs);

            for(const auto & node : nodes){
                add_node(node);
            }

            return Graph(
                std::move(name),
                std::move(nodes_),
                std::move(values_),
                std::move(inputs_),
                std::move(outputs_),
                std::move(initializers_),
                std::move(node_ids_),
                std::move(value_ids_)
            );
        }

    private:
        std::vector<IRNode>     nodes_;
        std::vector<Value>      values_;
        std::vector<ValueId>    inputs_;
        std::vector<ValueId>    outputs_;
        std::vector<ValueId>    initializers_;
        std::unordered_map<std::string, NodeId>     node_ids_;
        std::unordered_map<std::string, ValueId>    value_ids_;


        NodeId add_node(const Node& node){

            validate_node_schema(node);

            if(node_ids_.contains(node.name()))
                throw std::runtime_error("duplicated node name");

            
            NodeId id{nodes_.size()};
            std::vector<ValueId> inputs;
            std::vector<ValueId> outputs;

            for(const auto & name : node.inputs()){
                ValueId v_id = intern_value(name);
                value(v_id).add_consumer(id);

                inputs.push_back(v_id);
            }

            for(const auto & name : node.outputs()){
                ValueId v_id = intern_value(name);

                // set_producer will check multi producer ,prevent a tensor gen by multi nodes 
                value(v_id).set_producer(id);

                outputs.push_back(v_id);
            }

            nodes_.emplace_back(
                id,
                node.name(),
                node.op_type(),
                std::move(inputs),
                std::move(outputs)
            );

            node_ids_.emplace(node.name(), id);
            return id;
        }

        ValueId intern_value(const std::string & name){
            auto it = value_ids_.find(name);
            if(it != value_ids_.end()){
                return it->second;
            }

            ValueId id{values_.size()};
            values_.emplace_back(id, name);
            value_ids_.emplace(name, id);
            return id;
        }

        void set_inputs(const std::vector<std::string> & inputs){
            std::unordered_set<std::string> available;
            for(const auto & it : inputs){
                auto [name, inserted] = available.insert(it);
                if(!inserted) throw std::runtime_error("bad model : duplicated input name");

                // allow input name equs output name or initializer
                ValueId v_id = intern_value(it);
                inputs_.push_back(v_id);
            }
        }

        void set_outputs(const std::vector<std::string> & outputs){
            std::unordered_set<std::string> available;
            for(const auto & it : outputs){
                auto [name, inserted] = available.insert(it);
                if(!inserted) throw std::runtime_error("bad model : duplicated output name");

                // allow output name equs input name or initializer
                ValueId v_id = intern_value(it);
                outputs_.push_back(v_id);
            }
        }

        void set_initializer(std::unordered_map<std::string, Tensor> && initializer){
            for(auto & it : initializer){

                // allow initializer name equs input name or output name
                ValueId v_id = intern_value(it.first);

                // set_initializer will deny the duplicated initializer
                value(v_id).set_initializer(std::move(it.second));

                initializers_.emplace_back(v_id);
            }
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


        void validate_node_schema(const Node &node) const{

            std::pair<std::size_t, std::size_t> io_pair = OperationFactory::schema(node.op_type());
            if(node.inputs().size() != io_pair.first)
                throw std::runtime_error(std::format("{} expect {} inputs but received {}", node.op_type_str(), io_pair.first, node.inputs().size()));
            if(node.outputs().size() != io_pair.second)
                throw std::runtime_error(std::format("{} expect {} outputs but generated {}", node.op_type_str(), io_pair.second, node.outputs().size()));
        }

};

#endif