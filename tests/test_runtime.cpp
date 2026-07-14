#include "addop.hpp"
#include "executor.hpp"
#include "graph.hpp"
#include "graph_builder.hpp"
#include "matmulop.hpp"
#include "mulop.hpp"
#include "node.hpp"
#include "reluop.hpp"
#include "tensor.hpp"
#include "operation_fac.hpp"
#include "storage.hpp"
#include "value.hpp"

#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace {

void expect_data_eq(const Tensor& tensor, const std::vector<Float32>& expected) {
    assert(tensor.numel() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert(std::fabs(tensor[i] - expected[i]) < 1e-6f);
    }
}

void test_storage_interface() {
    CpuStorage cpu(std::vector<Float32>{1, 2, 3});

    Storage& storage = cpu;

    assert(storage.backend() == Backend::CPU);
    assert(storage.size() == 3);
    assert(storage.size_bytes() == 3 * sizeof(Float32));
    assert(storage.raw_data()[1] == 2);
}


template <typename Func>
void expect_throws(Func func) {
    bool thrown = false;

    try {
        func();
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}


void test_cpu_storage() {
    CpuStorage zeros(4);

    assert(zeros.size() == 4);
    assert(zeros.size_bytes() == 4 * sizeof(Float32));

    zeros[0] = 3.0f;
    assert(zeros[0] == 3.0f);
    assert(zeros.raw_data()[0] == 3.0f);

    std::vector<Float32> values{1, 2, 3};
    const Float32* original_ptr = values.data();

    CpuStorage moved(std::move(values));

    assert(moved.size() == 3);
    assert(moved.raw_data() == original_ptr);
    assert(moved[0] == 1);
    assert(moved[2] == 3);
}

void test_shape_and_tensor_creation() {
    const Shape shape({2, 3});
    assert(shape.numel() == 6);
    assert(shape == Shape({2, 3}));
    assert(shape != Shape({3, 2}));

    const Tensor zeros(shape, Dtype::Float32);
    assert(zeros.numel() == 6);
    assert(zeros.backend() == Backend::CPU);
    assert(zeros.is_cpu());
    assert(!zeros.is_cuda());
    expect_data_eq(zeros, {0, 0, 0, 0, 0, 0});

    const Tensor tensor(shape, {1, 2, 3, 4, 5, 6}, Dtype::Float32);
    assert(tensor.shape() == shape);
    assert(tensor.dtype() == Dtype::Float32);
    assert(tensor.backend() == Backend::CPU);
    expect_data_eq(tensor, {1, 2, 3, 4, 5, 6});

    expect_throws([] {
        Tensor bad_shape(Shape({2, 2}), {1, 2, 3}, Dtype::Float32);
    });
}

void test_tensor_copy_move_and_backend() {
    Tensor tensor(Shape({2, 2}), {1, 2, 3, 4}, Dtype::Float32);

    assert(tensor.backend() == Backend::CPU);
    assert(tensor.is_cpu());
    assert(!tensor.is_cuda());

    Tensor copied(tensor);
    copied[0] = 100;
    assert(tensor[0] == 1);
    assert(copied[0] == 100);
    assert(copied.backend() == Backend::CPU);

    Tensor assigned(Shape({1}), Dtype::Float32);
    assigned = tensor;
    assigned[1] = 200;
    assert(tensor[1] == 2);
    expect_data_eq(assigned, {1, 200, 3, 4});
    assert(assigned.shape() == Shape({2, 2}));
    assert(assigned.dtype() == Dtype::Float32);

    Tensor move_assigned(Shape({1}), Dtype::Float32);
    move_assigned = Tensor(Shape({2, 1}), {9, 10}, Dtype::Float32);
    assert(move_assigned.shape() == Shape({2, 1}));
    expect_data_eq(move_assigned, {9, 10});

    Tensor source(Shape({2}), {1, 2}, Dtype::Float32);
    Float32* old_ptr = source.raw_data();
    Tensor moved = std::move(source);

    assert(moved.backend() == Backend::CPU);
    expect_data_eq(moved, {1,2});

    assert(moved.raw_data() == old_ptr);
}

void test_value_initializer_semantics() {
    Value intermediate(0, "hidden");
    assert(!intermediate.is_initializer());

    Value weight(1, "weight");
    weight.set_initializer(Tensor(Shape({2}), {1, 2}, Dtype::Float32));
    assert(weight.is_initializer());
    assert(weight.name() == "weight");
    assert(weight.initializer().has_value());
    expect_data_eq(weight.initializer().value(), {1, 2});
}

void test_basic_ops() {
    const Tensor lhs(Shape({2, 3}), {2, 3, 5, 6, 7, 8}, Dtype::Float32);
    const Tensor rhs(Shape({2, 3}), {1, -2, 3, -4, 5, -6}, Dtype::Float32);

    expect_data_eq(AddOp::forward(lhs, rhs), {3, 1, 8, 2, 12, 2});
    expect_data_eq(MulOp::forward(lhs, rhs), {2, -6, 15, -24, 35, -48});
    expect_data_eq(ReluOp::forward(rhs), {1, 0, 3, 0, 5, 0});

    auto op = OperationFactory::create(OP_TYPE::AddOp);
    expect_data_eq(op->forward({&lhs, &rhs}), {3, 1, 8, 2, 12, 2});
    op = OperationFactory::create(OP_TYPE::MulOp);
    expect_data_eq(op->forward({&lhs, &rhs}), {2, -6, 15, -24, 35, -48});
    op = OperationFactory::create(OP_TYPE::ReluOp);
    expect_data_eq(op->forward({&rhs}), {1, 0, 3, 0, 5, 0});

    expect_throws([&] {
        AddOp::forward(lhs, Tensor(Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Dtype::Float32));
    });

    expect_throws([&] {
        MulOp::forward(lhs, Tensor(Shape({1, 6}), {1, 2, 3, 4, 5, 6}, Dtype::Float32));
    });
}

void test_matmul_op() {
    const Tensor lhs(Shape({2, 3}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
    const Tensor rhs(Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);

    const Tensor result = MatMulOp::forward(lhs, rhs);
    assert(result.shape() == Shape({2, 2}));
    assert(result.dtype() == Dtype::Float32);
    assert(result.backend() == Backend::CPU);
    expect_data_eq(result, {22, 28, 49, 64});

    const Tensor vector_like(Shape({3}), {1, 2, 3}, Dtype::Float32);
    expect_throws([&] {
        MatMulOp::forward(vector_like, rhs);
    });

    const Tensor incompatible_rhs(Shape({2, 2}), {1, 2, 3, 4}, Dtype::Float32);
    expect_throws([&] {
        MatMulOp::forward(lhs, incompatible_rhs);
    });
}

void test_graph_validation() {
    const Node add({"a", "b"}, {"c"}, "add_0", OP_TYPE::AddOp);
    const Node relu({"c"}, {"out"}, "relu_0", OP_TYPE::ReluOp);
    const Node mul({"out", "c"}, {"result"}, "mul_0", OP_TYPE::MulOp);
    const Graph graph = GraphBuilder().build(
        "unordered_graph",
        {mul, relu, add},
        {"a", "b"},
        {"result"},
        {});
    const auto& executor_order = graph.executor_order();

    assert(executor_order.size() == 3);
    assert(graph.inputs().size() == 2);
    assert(graph.outputs().size() == 1);
    assert(graph.node(executor_order[0]).name() == "add_0");
    assert(graph.node(executor_order[1]).name() == "relu_0");
    assert(graph.node(executor_order[2]).name() == "mul_0");

    expect_throws([] {
        GraphBuilder().build(
            "missing_input",
            {Node({"a", "missing"}, {"c"}, "bad_add", OP_TYPE::AddOp)},
            {"a"},
            {"c"},
            {});
    });

    expect_throws([] {
        GraphBuilder().build(
            "duplicate_output",
            {
                Node({"a", "b"}, {"c"}, "add_0", OP_TYPE::AddOp),
                Node({"a", "b"}, {"c"}, "add_1", OP_TYPE::AddOp),
            },
            {"a", "b"},
            {"c"},
            {});
    });

    expect_throws([] {
        GraphBuilder().build(
            "wrong_schema",
            {Node({"a"}, {"c"}, "bad_add", OP_TYPE::AddOp)},
            {"a"},
            {"c"},
            {});
    });

    expect_throws([] {
        GraphBuilder().build(
            "wrong_matmul_schema",
            {Node({"a"}, {"c"}, "bad_matmul", OP_TYPE::MatMulOp)},
            {"a"},
            {"c"},
            {});
    });

    expect_throws([] {
        GraphBuilder().build(
            "duplicate_node_name",
            {
                Node({"a", "b"}, {"c"}, "same_name", OP_TYPE::AddOp),
                Node({"c"}, {"out"}, "same_name", OP_TYPE::ReluOp),
            },
            {"a", "b"},
            {"out"},
            {});
    });

    expect_throws([] {
        GraphBuilder().build(
            "cyclic_graph",
            {
                Node({"b", "input"}, {"a"}, "add_0", OP_TYPE::AddOp),
                Node({"a"}, {"b"}, "relu_0", OP_TYPE::ReluOp),
            },
            {"input"},
            {"b"},
            {});
    });
}

void test_complex_graph_execution() {
    const Node matmul({"x", "weight"}, {"linear"}, "matmul_0", OP_TYPE::MatMulOp);
    const Node add_bias({"linear", "bias"}, {"biased"}, "add_bias", OP_TYPE::AddOp);
    const Node relu({"biased"}, {"activated"}, "relu_0", OP_TYPE::ReluOp);
    const Node gated({"activated", "gate"}, {"gated"}, "mul_gate", OP_TYPE::MulOp);
    const Node residual({"gated", "residual"}, {"residual_sum"}, "add_residual", OP_TYPE::AddOp);
    const Node mask({"residual_sum", "mask"}, {"output"}, "mul_mask", OP_TYPE::MulOp);

    const Graph graph = GraphBuilder().build(
        "complex_initializer_graph",
        {mask, relu, residual, matmul, gated, add_bias},
        {"x", "gate", "residual"},
        {"output"},
        {
            {"weight", Tensor(Shape({3, 2}), {1, -1, 2, 0, -1, 3}, Dtype::Float32)},
            {"bias", Tensor(Shape({2, 2}), {1, -2, 0, 4}, Dtype::Float32)},
            {"mask", Tensor(Shape({2, 2}), {1, 0, 1, 1}, Dtype::Float32)},
        });

    const auto& order = graph.executor_order();
    assert(order.size() == 6);
    assert(graph.node(order[0]).name() == "matmul_0");
    assert(graph.node(order[1]).name() == "add_bias");
    assert(graph.node(order[2]).name() == "relu_0");
    assert(graph.node(order[3]).name() == "mul_gate");
    assert(graph.node(order[4]).name() == "add_residual");
    assert(graph.node(order[5]).name() == "mul_mask");

    assert(graph.value(graph.value_id("weight")).is_initializer());
    assert(graph.value(graph.value_id("bias")).is_initializer());
    assert(graph.value(graph.value_id("mask")).is_initializer());
    assert(!graph.value(graph.value_id("x")).is_initializer());

    Executor executor(graph);
    executor.set_input(graph.value_id("x"), Tensor(Shape({2, 3}), {1, 2, 3, -1, 0, 2}, Dtype::Float32));
    executor.set_input(graph.value_id("gate"), Tensor(Shape({2, 2}), {2, 0.5, 4, -1}, Dtype::Float32));
    executor.set_input(graph.value_id("residual"), Tensor(Shape({2, 2}), {-1, 1, 2, 20}, Dtype::Float32));
    executor.run();

    const Tensor& output = executor.get_output("output");
    assert(output.shape() == Shape({2, 2}));
    assert(output.backend() == Backend::CPU);
    expect_data_eq(output, {5, 0, 2, 9});

    expect_throws([&] {
        executor.get_output("biased");
    });

    Executor missing_input_executor(graph);
    missing_input_executor.set_input(graph.value_id("x"), Tensor(Shape({2, 3}), {1, 2, 3, -1, 0, 2}, Dtype::Float32));
    missing_input_executor.set_input(graph.value_id("gate"), Tensor(Shape({2, 2}), {2, 0.5, 4, -1}, Dtype::Float32));
    expect_throws([&] {
        missing_input_executor.run();
    });
}

} // namespace

int main() {
    test_shape_and_tensor_creation();
    test_tensor_copy_move_and_backend();
    test_value_initializer_semantics();
    test_cpu_storage();
    test_storage_interface();
    test_basic_ops();
    test_matmul_op();
    test_graph_validation();
    test_complex_graph_execution();

    std::cout << "All runtime tests passed." << std::endl;
    return 0;
}
