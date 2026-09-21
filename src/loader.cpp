#include "onnx.pb.h"
#include "loader.hpp"
#include "graph.hpp"
#include "graph_builder.hpp"
#include "op_param.hpp"

#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <png.h>
#include <map>
#include <string>
#include <unordered_set>

namespace{


std::vector<long> shape_dims_trans(const google::protobuf::RepeatedPtrField<onnx::TensorShapeProto_Dimension> & onnx_dims){
    std::vector<long> res;
    for(auto & d : onnx_dims){
        if(d.has_dim_param()){
            // Symbolic dimensions use 1 until runtime shape binding is supported.
            res.push_back(1);
        }else if(d.has_dim_value()){
            res.push_back(d.dim_value());
        }else{
            throw std::runtime_error("unknown dims");
        }
    }
    return res;
}

std::vector<long> shape_dims_trans(const google::protobuf::RepeatedField<google::protobuf::int64> & onnx_dims){
    std::vector<long> res;
    res.assign(onnx_dims.begin(), onnx_dims.end());
    return res;
}

OpParam load_attribute(const onnx::NodeProto &node){

    OpParam op_param;

    auto load_ints = [](std::vector<long> & vec, const onnx::AttributeProto &attr){
        if (attr.type() != onnx::AttributeProto::INTS) {
            throw std::runtime_error("Conv.strides must be INTS");
        }
        vec.assign(
            attr.ints().begin(),
            attr.ints().end()
        );
    };

    auto load_int = [](long &param, const onnx::AttributeProto &attr){
        if (attr.type() != onnx::AttributeProto::INT) {
            throw std::runtime_error(
                "Conv.group must be INT"
            );
        }

        param = attr.i();
    };

    auto load_string = [](std::string &str, const onnx::AttributeProto &attr){
        if (attr.type() != onnx::AttributeProto::STRING) {
            throw std::runtime_error(
                "Conv.group must be INT"
            );
        }

        str = attr.s();
    };

    if(node.op_type() == "Conv"){
        ConvParam param;
        for(const auto & attr : node.attribute()){
            if (attr.name() == "group") {
                load_int(param.group, attr);
            } else if(attr.name() == "strides") {
                load_ints(param.strides, attr);
            } else if(attr.name() == "dilations"){
                load_ints(param.dilations, attr);
            } else if(attr.name() == "pads"){
                load_ints(param.pads, attr);
            } else if(attr.name() == "auto_pad"){
                load_string(param.auto_pad, attr);
            }
        }

        op_param = param;
    }else{
        op_param = NoParam();
    }

    return op_param;
}

Dtype load_dtype(const int onnx_dtype){
    // The runtime currently stores tensors as Float32 only.
    switch (onnx_dtype) {
        case onnx::TensorProto::FLOAT:
            return Dtype::Float32;
        case onnx::TensorProto::FLOAT16:
            break;
        case onnx::TensorProto::DOUBLE:
            break;
        case onnx::TensorProto::INT32:
            // return Dtype::Float32;
            break;
        case onnx::TensorProto::INT64:
            break;
        case onnx::TensorProto::BOOL:
            break;
        default:
            throw std::runtime_error("unknown onnx dtype");
    }

    throw std::runtime_error(std::format("unsupport onnx type {}", onnx::TensorProto_DataType_Name(onnx_dtype)));
}

std::vector<Float32> load_data(const onnx::TensorProto &tensor){
    switch (tensor.data_type()) {
        case onnx::TensorProto::FLOAT:
            if (tensor.has_raw_data()) {
                std::vector<Float32> res(tensor.raw_data().size() / sizeof(Float32));
                std::memcpy(
                    res.data(),
                    tensor.raw_data().data(),
                    tensor.raw_data().size());
                return res;
            } else {
                std::vector<Float32> res(tensor.float_data().begin(), tensor.float_data().end());
                return res;
            }

        case onnx::TensorProto::DOUBLE:
            // if (tensor.has_raw_data()) {
            //     std::vector<Float32> res(tensor.raw_data().size() / sizeof(Float32));
            //     std::memcpy(
            //         res.data(),
            //         tensor.raw_data().data(),
            //         tensor.raw_data().size());
            //     return res;
            // } else {
            //     tensor.float_data();
            //     std::vector<Float32> res(tensor.float_data().begin(), tensor.float_data().end());
            //     return res;
            // }
            break;

        case onnx::TensorProto::INT64:
            // if (tensor.has_raw_data()) {
            //     std::vector<Float32> res(tensor.raw_data().size() / sizeof(Float32));
            //     std::memcpy(
            //         res.data(),
            //         tensor.raw_data().data(),
            //         tensor.raw_data().size());
            //     return res;
            // } else {
            //     tensor.float_data();
            //     std::vector<Float32> res(tensor.float_data().begin(), tensor.float_data().end());
            //     return res;
            // }
            break;

        case onnx::TensorProto::INT32:
            // if (tensor.has_raw_data()) {
            //     std::vector<Float32> res(tensor.raw_data().size() / sizeof(Float32));
            //     std::memcpy(
            //         res.data(),
            //         tensor.raw_data().data(),
            //         tensor.raw_data().size());
            //     return res;
            // } else {
            //     std::vector<Float32> res(tensor.int32_data().begin(), tensor.int32_data().end());
            //     return res;
            // }
            break;

        case onnx::TensorProto::STRING:
            // if (tensor.has_raw_data()) {
            //     std::vector<Float32> res(tensor.raw_data().size() / sizeof(Float32));
            //     std::memcpy(
            //         res.data(),
            //         tensor.raw_data().data(),
            //         tensor.raw_data().size());
            //     return res;
            // } else {
            //     tensor.float_data();
            //     std::vector<Float32> res(tensor.float_data().begin(), tensor.float_data().end());
            //     return res;
            // }
            break;

        default:
            throw std::runtime_error("unknown onnx dtype");
    }

    throw std::runtime_error(std::format("unsupport onnx type {}", onnx::TensorProto_DataType_Name(tensor.data_type())));
}


OP_TYPE load_op_type(const std::string& type) {
    // Match ONNX names to runtime operator types here.
    if (type == "Add")    return OP_TYPE::AddOp;
    if (type == "Mul")    return OP_TYPE::MulOp;
    if (type == "Relu")   return OP_TYPE::ReluOp;
    if (type == "MatMul") return OP_TYPE::MatMulOp;

    // new op
    if (type == "Conv")   return OP_TYPE::ConvOp;

    throw std::runtime_error("unsupported ONNX operator: " + type);
}

onnx::ModelProto onnx_load(const std::string path){

    onnx::ModelProto model;

    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open model: " + path
        );
    }

    if (!model.ParseFromIstream(&file)) {
        throw std::runtime_error(
            "Failed to parse ONNX model: " + path
        );
    }

    if (!model.has_graph()) {
        throw std::runtime_error(
            "Invalid ONNX model: graph is missing"
        );
    }
    return model;
}

std::vector<Node> load_node(onnx::ModelProto & model){
    auto & graph = model.graph();

    std::vector<Node> res;

    for(int i = 0; i < graph.node_size(); i++){
        auto & node = graph.node(i);

        res.emplace_back(
            std::vector<std::string>(node.input().begin(), node.input().end()),
            std::vector<std::string>(node.output().begin(), node.output().end()),
            node.name(),
            load_op_type(node.op_type()),
            load_attribute(node)
        );
    }

    return res;
}

std::vector<std::string> load_input(onnx::ModelProto & model){
    std::vector<std::string> result;
    result.reserve(model.graph().input_size());

    for (const auto& input : model.graph().input()) {
        result.push_back(input.name());
    }

    return result;
}

std::vector<std::string> load_output(onnx::ModelProto & model){

    std::vector<std::string> result;
    result.reserve(model.graph().output_size());

    for (const auto& output : model.graph().output()) {
        result.push_back(output.name());
    }

    return result;
}

std::vector<TensorDesc> load_input_desc(
    onnx::ModelProto & model,
    std::optional<long> batch_size){
    auto & graph = model.graph();
    std::vector<TensorDesc> res;
    std::unordered_set<std::string> initializer_names;
    for (const auto& initializer : graph.initializer()) {
        initializer_names.insert(initializer.name());
    }

    bool batch_axis_bound = false;

    for(auto & i : graph.input()){
        auto & onnx_shape = i.type().tensor_type().shape();
        auto onnx_dtype = i.type().tensor_type().elem_type();
        auto dims = shape_dims_trans(onnx_shape.dim());

        // Bind the leading data axis before graph shape inference.
        if (batch_size && !initializer_names.contains(i.name())) {
            if (!dims.empty()) {
                dims.front() = *batch_size;
                batch_axis_bound = true;
            }
        }

        res.emplace_back(
            Shape(std::move(dims)),
            Backend::CPU,
            load_dtype(onnx_dtype)
        );
    }

    if (batch_size && !batch_axis_bound) {
        throw std::runtime_error(
            "batch size was provided, but the model has no data input");
    }

    return res;
}


std::unordered_map<std::string, Tensor> load_initializer(onnx::ModelProto & model){
    auto & graph = model.graph();
    std::unordered_map<std::string, Tensor> res;

    for(auto & i : graph.initializer()){
        res[i.name()] = 
        Tensor(
            Shape(shape_dims_trans(i.dims())), 
            load_data(i),
            load_dtype(i.data_type()),
            Backend::CPU
        );
    }

    return res;
}
}


Tensor load_image_batch(const std::vector<std::filesystem::path>& paths) {
    if (paths.empty()) {
        throw std::invalid_argument("image batch must not be empty");
    }

    constexpr std::size_t mnist_image_size = 28 * 28;
    std::vector<Float32> data;
    data.reserve(paths.size() * mnist_image_size);

    for (const auto& path : paths) {
        png_image image{};
        image.version = PNG_IMAGE_VERSION;

        if (!png_image_begin_read_from_file(&image, path.c_str())) {
            throw std::runtime_error(
                "failed to read PNG header: " + std::string(image.message));
        }

        if (image.width != 28 || image.height != 28) {
            png_image_free(&image);
            throw std::runtime_error(
                "MNIST images must have dimensions 28x28: " + path.string());
        }

        image.format = PNG_FORMAT_GRAY;
        std::vector<unsigned char> pixels(PNG_IMAGE_SIZE(image));
        if (!png_image_finish_read(&image, nullptr, pixels.data(), 0, nullptr)) {
            const std::string message = image.message;
            png_image_free(&image);
            throw std::runtime_error("failed to decode PNG: " + message);
        }

        png_image_free(&image);

        // Apply the same normalization to every row in the batch.
        for (unsigned char pixel : pixels) {
            const Float32 normalized = static_cast<Float32>(pixel) / 255.0f;
            data.push_back((normalized - 0.1307f) / 0.3081f);
        }
    }

    return Tensor(
        Shape({static_cast<long>(paths.size()),
               static_cast<long>(mnist_image_size)}),
        std::move(data),
        Dtype::Float32);
}

Tensor load_image(const std::filesystem::path& path) {
    return load_image_batch({path});
}

Graph Loader::load(const std::string path, std::optional<long> batch_size){
    if (batch_size && *batch_size <= 0) {
        throw std::invalid_argument("batch size must be positive");
    }

    onnx::ModelProto model = onnx_load(path);
    GraphBuilder graph_builder;
    
    std::vector<Node> nodes;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<TensorDesc> inputs_desc;
    std::unordered_map<std::string, Tensor> initializer;

    return graph_builder.build(
        model.graph().name(),
        load_node(model),
        load_input(model),
        load_output(model),
        load_input_desc(model, batch_size),
        load_initializer(model)
    );
}



// void inspect(const onnx::ModelProto& model){

//     std::cout << "ONNX model loaded successfully\n";
//     std::cout << "IR version: " << model.ir_version() << '\n';
//     std::cout << "Producer: " << model.producer_name() << '\n';
//     std::cout << "Graph name: " << model.graph().name() << '\n';

//     std::cout << "graph inputs:" << model.graph().input_size() << '\n';

//     std::cout << "inputs:" << model.graph().input().at(0).name() << '\n';
//     std::cout << "graph outputs:" << model.graph().output_size() << '\n';
//     std::cout << "graph initializers:" << model.graph().initializer_size() << '\n';
//     std::cout << "graph nodes:" << model.graph().node_size() << '\n';
//     std::cout << "graph opset:" << model.opset_import_size() << '\n';

//     for(const auto & i : model.opset_import()){
//         const std::string domain = i.domain().empty() ? "ai.onnx" : i.domain();

//         std::cout << "Domain: " << domain
//                 << ", opset: " << i.version()
//                 << '\n';

//     }
//     using op_key = std::pair<std::string, std::string>;

//     std::map<op_key, size_t> operators;

//     for (const auto& node : model.graph().node()) {
//         const std::string domain =
//             node.domain().empty() ? "ai.onnx" : node.domain();

//         const op_key key{
//             domain,
//             node.op_type()
//         };

//         ++operators[key];
//     }

//     for(const auto & i : operators){
//         std::cout << "name: " << i.first.second << " nums: " << i.second << std::endl;
//     }

// }

// int main(){

//     onnx::ModelProto model = onnx_load("../experiments/mnist_onnx/artifacts/mnist_pressure_net.onnx");
//     inspect(model);
//     return 0;
// }
