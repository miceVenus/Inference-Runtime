#include "addop.hpp"
#include "executor.hpp"
#include "graph.hpp"
#include "matmulop.hpp"
#include "mulop.hpp"
#include "node.hpp"
#include "reluop.hpp"
#include "tensor.hpp"
#include "operation_fac.hpp"
#include "storage.hpp"

#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
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
    const Graph graph({mul, relu, add}, {"a", "b"}, {"result"}, "unordered_graph");

    assert(graph.nodes().size() == 3);
    assert(graph.inputs().size() == 2);
    assert(graph.outputs().size() == 1);
    assert(graph.nodes()[0].name() == "add_0");
    assert(graph.nodes()[1].name() == "relu_0");
    assert(graph.nodes()[2].name() == "mul_0");

    expect_throws([] {
        Graph missing_input(
            {Node({"a", "missing"}, {"c"}, "bad_add", OP_TYPE::AddOp)},
            {"a"},
            {"c"},
            "missing_input");
    });

    expect_throws([] {
        Graph duplicate_output(
            {
                Node({"a", "b"}, {"c"}, "add_0", OP_TYPE::AddOp),
                Node({"a", "b"}, {"c"}, "add_1", OP_TYPE::AddOp),
            },
            {"a", "b"},
            {"c"},
            "duplicate_output");
    });

    expect_throws([] {
        Graph wrong_schema(
            {Node({"a"}, {"c"}, "bad_add", OP_TYPE::AddOp)},
            {"a"},
            {"c"},
            "wrong_schema");
    });

    expect_throws([] {
        Graph wrong_matmul_schema(
            {Node({"a"}, {"c"}, "bad_matmul", OP_TYPE::MatMulOp)},
            {"a"},
            {"c"},
            "wrong_matmul_schema");
    });

    expect_throws([] {
        Graph duplicate_node_name(
            {
                Node({"a", "b"}, {"c"}, "same_name", OP_TYPE::AddOp),
                Node({"c"}, {"out"}, "same_name", OP_TYPE::ReluOp),
            },
            {"a", "b"},
            {"out"},
            "duplicate_node_name");
    });

    expect_throws([] {
        Graph cyclic_graph(
            {
                Node({"b", "input"}, {"a"}, "add_0", OP_TYPE::AddOp),
                Node({"a"}, {"b"}, "relu_0", OP_TYPE::ReluOp),
            },
            {"input"},
            {"b"},
            "cyclic_graph");
    });
}

void test_executor_end_to_end() {

    std::clock_t start = std::clock();

    const Node add_0({"a", "b"}, {"c"}, "add_0", OP_TYPE::AddOp);
    const Node add_1({"c", "a"}, {"d"}, "add_1", OP_TYPE::AddOp);
    const Node mul_0({"d", "c"}, {"e"}, "mul_0", OP_TYPE::MulOp);
    const Node relu_0({"e"}, {"relu_e"}, "relu_0", OP_TYPE::ReluOp);
    const Node mask_0({"relu_e", "mask"}, {"masked_relu_e"}, "mask_0", OP_TYPE::MulOp);
    const Node matmul_0({"matmul_input", "matmul_weight"}, {"matmul_output"}, "matmul_0", OP_TYPE::MatMulOp);
    const Node relu_1({"matmul_output"}, {"relu_matmul_output"}, "relu_1", OP_TYPE::ReluOp);

    const Graph graph(
        {relu_1, mask_0, mul_0, add_1, matmul_0, relu_0, add_0},
        {"a", "b", "mask", "matmul_input", "matmul_weight"},
        {"masked_relu_e", "relu_matmul_output"},
        "unordered_toy_graph");

    Executor executor(graph);
    executor.set_input("a", Tensor(Shape({1, 2, 3}), {2, 3, 5, 6, 7, 8}, Dtype::Float32));
    executor.set_input("b", Tensor(Shape({1, 2, 3}), {1, -2, 3, -4, 5, -6}, Dtype::Float32));
    executor.set_input("mask", Tensor(Shape({1, 2, 3}), {0, 1, 1, 0, 1, 0}, Dtype::Float32));
    executor.set_input("matmul_input", Tensor(Shape({2, 3}), {1, -2, 3, 4, 0, -1}, Dtype::Float32));
    executor.set_input("matmul_weight", Tensor(Shape({3, 2}), {2, 1, -1, 3, 4, -2}, Dtype::Float32));

    executor.run();

    assert(executor.get_output("masked_relu_e").backend() == Backend::CPU);
    assert(executor.get_output("relu_matmul_output").backend() == Backend::CPU);
    expect_data_eq(executor.get_output("masked_relu_e"), {0, 4, 104, 0, 228, 0});
    expect_data_eq(executor.get_output("relu_matmul_output"), {16, 0, 4, 6});

    std::clock_t end = std::clock();

    std::cout << "times: " << end - start << std::endl; 


    expect_throws([&] {
        executor.get_output("not_an_output");
    });

    Executor missing_input_executor(graph);
    missing_input_executor.set_input("a", Tensor(Shape({1, 2, 3}), {2, 3, 5, 6, 7, 8}, Dtype::Float32));
    expect_throws([&] {
        missing_input_executor.run();
    });

}

} // namespace

int main() {
    test_shape_and_tensor_creation();
    test_tensor_copy_move_and_backend();
    test_cpu_storage();
    test_storage_interface();
    test_basic_ops();
    test_matmul_op();
    test_graph_validation();
    test_executor_end_to_end();

    std::cout << "All runtime tests passed." << std::endl;
    return 0;
}
