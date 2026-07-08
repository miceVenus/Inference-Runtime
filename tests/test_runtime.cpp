#include "addop.hpp"
#include "executor.hpp"
#include "graph.hpp"
#include "matmulop.hpp"
#include "mulop.hpp"
#include "node.hpp"
#include "reluop.hpp"
#include "tensor.hpp"
#include "operation_fac.hpp"

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
    assert(tensor.data().size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert(std::fabs(tensor.data()[i] - expected[i]) < 1e-6f);
    }
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

void test_shape_and_tensor() {
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

void test_backend_semantics() {
    const Tensor cpu_tensor(Shape({2, 2}), {1, 2, 3, 4}, Dtype::Float32);
    const Tensor cuda_tensor(Shape({2, 2}), {1, 2, 3, 4}, Dtype::Float32, Backend::CUDA);

    assert(cpu_tensor.backend() == Backend::CPU);
    assert(cpu_tensor.is_cpu());
    assert(!cpu_tensor.is_cuda());

    assert(cuda_tensor.backend() == Backend::CUDA);
    assert(!cuda_tensor.is_cpu());
    assert(cuda_tensor.is_cuda());

    const Tensor copied(cuda_tensor);
    assert(copied.backend() == Backend::CUDA);
    expect_data_eq(copied, {1, 2, 3, 4});

    Tensor assigned(Shape({2, 2}), Dtype::Float32);
    assigned = cuda_tensor;
    assert(assigned.backend() == Backend::CUDA);
    expect_data_eq(assigned, {1, 2, 3, 4});

    Tensor moved(Tensor(Shape({2, 2}), {5, 6, 7, 8}, Dtype::Float32, Backend::CUDA));
    assert(moved.backend() == Backend::CUDA);
    expect_data_eq(moved, {5, 6, 7, 8});

    const Tensor cuda_lhs(Shape({2, 2}), {1, -2, 3, -4}, Dtype::Float32, Backend::CUDA);
    const Tensor cuda_rhs(Shape({2, 2}), {5, 6, 7, 8}, Dtype::Float32, Backend::CUDA);

    const Tensor add_result = AddOp::forward(cuda_lhs, cuda_rhs);
    assert(add_result.backend() == Backend::CUDA);
    expect_data_eq(add_result, {6, 4, 10, 4});

    const Tensor mul_result = MulOp::forward(cuda_lhs, cuda_rhs);
    assert(mul_result.backend() == Backend::CUDA);
    expect_data_eq(mul_result, {5, -12, 21, -32});

    const Tensor relu_result = ReluOp::forward(cuda_lhs);
    assert(relu_result.backend() == Backend::CUDA);
    expect_data_eq(relu_result, {1, 0, 3, 0});

    const Tensor matmul_lhs(Shape({2, 3}), {1, 2, 3, 4, 5, 6}, Dtype::Float32, Backend::CUDA);
    const Tensor matmul_rhs(Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Dtype::Float32, Backend::CUDA);
    const Tensor matmul_result = MatMulOp::forward(matmul_lhs, matmul_rhs);
    assert(matmul_result.backend() == Backend::CUDA);
    expect_data_eq(matmul_result, {22, 28, 49, 64});

    expect_throws([&] {
        AddOp::forward(cpu_tensor, cuda_tensor);
    });

    expect_throws([&] {
        MulOp::forward(cpu_tensor, cuda_tensor);
    });

    expect_throws([&] {
        MatMulOp::forward(
            Tensor(Shape({2, 2}), {1, 2, 3, 4}, Dtype::Float32),
            Tensor(Shape({2, 2}), {1, 2, 3, 4}, Dtype::Float32, Backend::CUDA));
    });
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
    const Graph graph({add, relu}, {"a", "b"}, {"out"}, "valid_graph");

    assert(graph.nodes().size() == 2);
    assert(graph.inputs().size() == 2);
    assert(graph.outputs().size() == 1);

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
        {add_0, add_1, mul_0, relu_0, mask_0, matmul_0, relu_1},
        {"a", "b", "mask", "matmul_input", "matmul_weight"},
        {"masked_relu_e", "relu_matmul_output"},
        "toy_graph");

    Executor executor(graph);
    executor.set_input("a", Tensor(Shape({1, 2, 3}), {2, 3, 5, 6, 7, 8}, Dtype::Float32, Backend::CPU));
    executor.set_input("b", Tensor(Shape({1, 2, 3}), {1, -2, 3, -4, 5, -6}, Dtype::Float32, Backend::CPU));
    executor.set_input("mask", Tensor(Shape({1, 2, 3}), {0, 1, 1, 0, 1, 0}, Dtype::Float32, Backend::CPU));
    executor.set_input("matmul_input", Tensor(Shape({2, 3}), {1, -2, 3, 4, 0, -1}, Dtype::Float32, Backend::CPU));
    executor.set_input("matmul_weight", Tensor(Shape({3, 2}), {2, 1, -1, 3, 4, -2}, Dtype::Float32, Backend::CPU));

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
    test_shape_and_tensor();
    test_backend_semantics();
    test_basic_ops();
    test_matmul_op();
    test_graph_validation();
    test_executor_end_to_end();

    std::cout << "All runtime tests passed." << std::endl;
    return 0;
}
