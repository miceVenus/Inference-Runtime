#include "executor.hpp"
#include "graph.hpp"
#include "graph_builder.hpp"
#include "node.hpp"
#include "operation.hpp"
#include "tensor.hpp"

#include <exception>
#include <iostream>
#include <unordered_map>
#include <vector>

int main() {
    try {
        const Node matmul({"x", "weight"}, {"linear"}, "matmul_0", OP_TYPE::MatMulOp);
        const Node add_bias({"linear", "bias"}, {"biased"}, "add_bias", OP_TYPE::AddOp);
        const Node relu({"biased"}, {"activated"}, "relu_0", OP_TYPE::ReluOp);
        const Node gated({"activated", "gate"}, {"gated"}, "mul_gate", OP_TYPE::MulOp);
        const Node residual({"gated", "residual"}, {"residual_sum"}, "add_residual", OP_TYPE::AddOp);
        const Node mask({"residual_sum", "mask"}, {"output"}, "mul_mask", OP_TYPE::MulOp);

        Graph graph = GraphBuilder().build(
            "demo_graph",
            {mask, relu, residual, matmul, gated, add_bias},
            {"x", "gate", "residual"},
            {"output"},
            {
                {"weight", Tensor(Shape({3, 2}), {1, -1, 2, 0, -1, 3}, Dtype::Float32)},
                {"bias", Tensor(Shape({2, 2}), {1, -2, 0, 4}, Dtype::Float32)},
                {"mask", Tensor(Shape({2, 2}), {1, 0, 1, 1}, Dtype::Float32)},
            });

        Executor executor(graph);
        executor.set_input(graph.value_id("x"), Tensor(Shape({2, 3}), {1, 2, 3, -1, 0, 2}, Dtype::Float32));
        executor.set_input(graph.value_id("gate"), Tensor(Shape({2, 2}), {2, 0.5, 4, -1}, Dtype::Float32));
        executor.set_input(graph.value_id("residual"), Tensor(Shape({2, 2}), {-1, 1, 2, 20}, Dtype::Float32));

        executor.run();

        std::cout << "Demo graph output:" << std::endl;
        print(executor.get_output("output"));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
