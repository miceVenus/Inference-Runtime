#ifndef VALUE_HPP
#define VALUE_HPP

#include "operation.hpp"
#include "tensor.hpp"

#include "ir_types.hpp"

#include <string>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <optional>
#include <cassert>

class Value{

    public:
        Value(ValueId id, std::string name, std::optional<TensorDesc> desc = std::nullopt)
        :id_(id), name_(std::move(name)), desc_(desc){}

        const std::string& name() const{
            return name_;
        }

        ValueId id() const{
            return id_;
        }

        const std::optional<NodeId> producer() const{
            return producer_;
        }

        const std::vector<NodeId> &consumer() const{
            return consumer_;
        }

        void add_consumer(NodeId consumer){
            consumer_.emplace_back(consumer);
        }

        void set_producer(NodeId producer){
            if(producer_.has_value())
                throw std::runtime_error("value already has producer");

            producer_ = producer;
        }

        void set_initializer(Tensor initializer){
            if(initializer_.has_value())
                throw std::runtime_error("bad model : duplicated initializer");

            initializer_ = std::move(initializer);
        }

        bool is_initializer() const{
            return initializer_.has_value();
        }


        const std::optional<Tensor>& initializer() const {
            return initializer_;
        }

        const std::optional<TensorDesc>& desc() const {
            return desc_;
        }

        void set_desc(TensorDesc t_desc) {
            if(desc_.has_value())
                throw std::runtime_error("try to re set desc in a value");

            desc_ = std::move(t_desc);
        }

        void set_birth(size_t birth){
            assert(!birth_.has_value());

            birth_ = birth;
        }

        void set_death(size_t death){
            death_ = death;
        }

        const std::pair<size_t, size_t> life_span() const{
            assert(birth_.has_value());
            return {birth_.value(), death_};
        }

        
    private:

        ValueId id_;
        std::string name_;

        std::optional<NodeId> producer_;
        std::vector<NodeId> consumer_;

        std::optional<size_t> birth_;
        size_t death_;

        std::optional<TensorDesc> desc_;

        std::optional<Tensor> initializer_;
};

// inline std::ostream& operator<<(std::ostream &os, const Node &node){
//     os << "Node name: " << node.name_ << std::endl;
//     os << "Op type: " << node.op_type_str() << std::endl;
//     os << "Inputs: ";

//     for(auto &i : node.input_){
//         os << i << ", ";
//     }

//     os << std::endl;

//     os << "Outputs: ";
//     for(auto &o : node.output_){
//         os << o << ", ";
//     }

//     os << std::endl;  
//     return os;    
// }

#endif