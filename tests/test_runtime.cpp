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
#include "storage_pool.hpp"
#include "value.hpp"

#include <cassert>
#include <algorithm>
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

IRNode make_operation_node(OP_TYPE type) {
    return IRNode(
        0,
        "factory_test",
        type,
        std::vector<ValueId>{},
        std::vector<ValueId>{},
        NoParam{});
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

    auto op = OperationFactory::create(make_operation_node(OP_TYPE::AddOp));
    TensorDesc add_meta = op->forward_T({TensorDesc(lhs), TensorDesc(rhs)});
    assert(add_meta.shape_ == lhs.shape());
    assert(add_meta.dtype_ == Dtype::Float32);
    assert(add_meta.backend_ == Backend::CPU);
    Tensor add_out(add_meta);
    Tensor& add_ref = op->forward({&lhs, &rhs}, add_out);
    assert(&add_ref == &add_out);
    expect_data_eq(add_out, {3, 1, 8, 2, 12, 2});

    op = OperationFactory::create(make_operation_node(OP_TYPE::MulOp));
    TensorDesc mul_meta = op->forward_T({TensorDesc(lhs), TensorDesc(rhs)});
    assert(mul_meta.shape_ == lhs.shape());
    assert(mul_meta.dtype_ == Dtype::Float32);
    assert(mul_meta.backend_ == Backend::CPU);
    Tensor mul_out(mul_meta);
    Tensor& mul_ref = op->forward({&lhs, &rhs}, mul_out);
    assert(&mul_ref == &mul_out);
    expect_data_eq(mul_out, {2, -6, 15, -24, 35, -48});

    op = OperationFactory::create(make_operation_node(OP_TYPE::ReluOp));
    TensorDesc relu_meta = op->forward_T({TensorDesc(rhs)});
    assert(relu_meta.shape_ == rhs.shape());
    assert(relu_meta.dtype_ == Dtype::Float32);
    assert(relu_meta.backend_ == Backend::CPU);
    Tensor relu_out(relu_meta);
    Tensor& relu_ref = op->forward({&rhs}, relu_out);
    assert(&relu_ref == &relu_out);
    expect_data_eq(relu_out, {1, 0, 3, 0, 5, 0});

    expect_throws([&] {
        auto bad_rhs = Tensor(Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
        OperationFactory::create(make_operation_node(OP_TYPE::AddOp))->forward_T(
            {TensorDesc(lhs), TensorDesc(bad_rhs)});
    });

    expect_throws([&] {
        auto bad_rhs = Tensor(Shape({1, 6}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
        OperationFactory::create(make_operation_node(OP_TYPE::MulOp))->forward_T(
            {TensorDesc(lhs), TensorDesc(bad_rhs)});
    });
}

void expect_binary_broadcast(
    OP_TYPE type,
    const Tensor& lhs,
    const Tensor& rhs,
    const Shape& expected_shape,
    const std::vector<Float32>& expected_data) {
    auto op = OperationFactory::create(make_operation_node(type));
    const TensorDesc output_desc = op->forward_T({TensorDesc(lhs), TensorDesc(rhs)});

    assert(output_desc.shape_ == expected_shape);
    assert(output_desc.backend_ == Backend::CPU);
    assert(output_desc.dtype_ == Dtype::Float32);

    Tensor output(output_desc);
    Tensor& result = op->forward({&lhs, &rhs}, output);
    assert(&result == &output);
    expect_data_eq(output, expected_data);
}

void test_binary_broadcast_ops() {
    const Tensor matrix_lhs(
        Shape({2, 3}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
    const Tensor matrix_rhs(
        Shape({2, 3}), {10, 20, 30, 40, 50, 60}, Dtype::Float32);

    for (const OP_TYPE type : {OP_TYPE::AddOp, OP_TYPE::MulOp}) {
        expect_binary_broadcast(
            type,
            matrix_lhs,
            matrix_rhs,
            Shape({2, 3}),
            type == OP_TYPE::AddOp
                ? std::vector<Float32>{11, 22, 33, 44, 55, 66}
                : std::vector<Float32>{10, 40, 90, 160, 250, 360});
    }

    const Tensor row(
        Shape({3}), {10, 20, 30}, Dtype::Float32);
    for (const OP_TYPE type : {OP_TYPE::AddOp, OP_TYPE::MulOp}) {
        expect_binary_broadcast(
            type,
            matrix_lhs,
            row,
            Shape({2, 3}),
            type == OP_TYPE::AddOp
                ? std::vector<Float32>{11, 22, 33, 14, 25, 36}
                : std::vector<Float32>{10, 40, 90, 40, 100, 180});
    }

    const Tensor cube_lhs(
        Shape({2, 1, 4}), {1, 2, 3, 4, 5, 6, 7, 8}, Dtype::Float32);
    const Tensor matrix_rhs_3x4(
        Shape({3, 4}),
        {10, 20, 30, 40, 100, 200, 300, 400, 1000, 2000, 3000, 4000},
        Dtype::Float32);
    for (const OP_TYPE type : {OP_TYPE::AddOp, OP_TYPE::MulOp}) {
        expect_binary_broadcast(
            type,
            cube_lhs,
            matrix_rhs_3x4,
            Shape({2, 3, 4}),
            type == OP_TYPE::AddOp
                ? std::vector<Float32>{
                      11, 22, 33, 44, 101, 202, 303, 404,
                      1001, 2002, 3003, 4004, 15, 26, 37, 48,
                      105, 206, 307, 408, 1005, 2006, 3007, 4008}
                : std::vector<Float32>{
                      10, 40, 90, 160, 100, 400, 900, 1600,
                      1000, 4000, 9000, 16000, 50, 120, 210, 320,
                      500, 1200, 2100, 3200, 5000, 12000, 21000, 32000});
    }

    const Tensor scalar_like(
        Shape({1}), {5}, Dtype::Float32);
    const Tensor cube(
        Shape({2, 3, 4}),
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
        Dtype::Float32);
    expect_binary_broadcast(
        OP_TYPE::AddOp,
        cube,
        scalar_like,
        Shape({2, 3, 4}),
        {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
    expect_binary_broadcast(
        OP_TYPE::MulOp,
        cube,
        scalar_like,
        Shape({2, 3, 4}),
        {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
         65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120});

    const Tensor scalar(
        std::vector<long>{}, {2}, Dtype::Float32);
    expect_binary_broadcast(
        OP_TYPE::AddOp,
        cube,
        scalar,
        Shape({2, 3, 4}),
        {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
    expect_binary_broadcast(
        OP_TYPE::MulOp,
        cube,
        scalar,
        Shape({2, 3, 4}),
        {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
         26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48});

    expect_throws([&] {
        auto op = OperationFactory::create(make_operation_node(OP_TYPE::AddOp));
        const Tensor incompatible(
            Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
        op->forward_T({TensorDesc(matrix_lhs), TensorDesc(incompatible)});
    });

    expect_throws([&] {
        auto op = OperationFactory::create(make_operation_node(OP_TYPE::MulOp));
        const Tensor incompatible(
            Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
        op->forward_T({TensorDesc(matrix_lhs), TensorDesc(incompatible)});
    });
}

void test_matmul_op() {
    const auto expect_matmul = [](
        const Tensor& lhs,
        const Tensor& rhs,
        const Shape& expected_shape,
        const std::vector<Float32>& expected_data) {
        auto op = OperationFactory::create(make_operation_node(OP_TYPE::MatMulOp));
        const TensorDesc output_desc =
            op->forward_T({TensorDesc(lhs), TensorDesc(rhs)});

        assert(output_desc.shape_ == expected_shape);
        assert(output_desc.backend_ == Backend::CPU);
        assert(output_desc.dtype_ == Dtype::Float32);

        Tensor output(output_desc);
        Tensor& result = op->forward({&lhs, &rhs}, output);
        assert(&result == &output);
        expect_data_eq(output, expected_data);
    };

    const Tensor matrix_lhs(
        Shape({2, 3}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
    const Tensor matrix_rhs(
        Shape({3, 2}), {1, 2, 3, 4, 5, 6}, Dtype::Float32);
    expect_matmul(
        matrix_lhs,
        matrix_rhs,
        Shape({2, 2}),
        {22, 28, 49, 64});

    const Tensor vector_lhs(
        Shape({3}), {1, 2, 3}, Dtype::Float32);
    expect_matmul(
        vector_lhs,
        matrix_rhs,
        Shape({2}),
        {22, 28});

    const Tensor vector_rhs(
        Shape({3}), {1, 2, 3}, Dtype::Float32);
    expect_matmul(
        matrix_lhs,
        vector_rhs,
        Shape({2}),
        {14, 32});

    const Tensor other_vector(
        Shape({3}), {4, 5, 6}, Dtype::Float32);
    expect_matmul(
        vector_lhs,
        other_vector,
        Shape(std::vector<long>{}),
        {32});

    const Tensor batched_lhs(
        Shape({2, 2, 3}),
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        Dtype::Float32);
    const Tensor batched_rhs(
        Shape({2, 3, 2}),
        {1, 2, 3, 4, 5, 6, 1, 0, 0, 1, 1, 1},
        Dtype::Float32);
    expect_matmul(
        batched_lhs,
        batched_rhs,
        Shape({2, 2, 2}),
        {22, 28, 49, 64, 16, 17, 22, 23});

    const Tensor broadcast_lhs(
        Shape({2, 1, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8}, Dtype::Float32);
    const Tensor broadcast_rhs(
        Shape({3, 2, 2}),
        {1, 0, 0, 1, 2, 0, 0, 2, 1, 1, 1, 1},
        Dtype::Float32);
    expect_matmul(
        broadcast_lhs,
        broadcast_rhs,
        Shape({2, 3, 2, 2}),
        {1, 2, 3, 4, 2, 4, 6, 8, 3, 3, 7, 7,
         5, 6, 7, 8, 10, 12, 14, 16, 11, 11, 15, 15});

    expect_throws([&] {
        const Tensor incompatible_rhs(
            Shape({4}), {1, 2, 3, 4}, Dtype::Float32);
        auto op = OperationFactory::create(make_operation_node(OP_TYPE::MatMulOp));
        op->forward_T({TensorDesc(matrix_lhs), TensorDesc(incompatible_rhs)});
    });

    expect_throws([&] {
        const Tensor incompatible_batch_rhs(
            Shape({3, 3, 2}),
            {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1},
            Dtype::Float32);
        auto op = OperationFactory::create(make_operation_node(OP_TYPE::MatMulOp));
        op->forward_T(
            {TensorDesc(batched_lhs), TensorDesc(incompatible_batch_rhs)});
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
        {
            TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
            TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
        },
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
            {TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32)},
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
            {
                TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
                TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
            },
            {});
    });

    expect_throws([] {
        GraphBuilder().build(
            "wrong_schema",
            {Node({"a"}, {"c"}, "bad_add", OP_TYPE::AddOp)},
            {"a"},
            {"c"},
            {TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32)},
            {});
    });

    expect_throws([] {
        GraphBuilder().build(
            "wrong_matmul_schema",
            {Node({"a"}, {"c"}, "bad_matmul", OP_TYPE::MatMulOp)},
            {"a"},
            {"c"},
            {TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32)},
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
            {
                TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
                TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
            },
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
            {TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32)},
            {});
    });
}

void test_life_span_uses_last_consumer() {
    const Node producer(
        {"a", "b"}, {"shared"}, "produce_shared", OP_TYPE::AddOp);
    const Node left_branch(
        {"shared"}, {"left"}, "left_branch", OP_TYPE::ReluOp);
    const Node right_branch(
        {"shared", "scale"}, {"right"}, "right_branch", OP_TYPE::MulOp);
    const Node join(
        {"left", "right"}, {"output"}, "join_branches", OP_TYPE::AddOp);

    const Graph graph = GraphBuilder().build(
        "branching_life_span",
        {join, right_branch, left_branch, producer},
        {"a", "b", "scale"},
        {"output"},
        {
            TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
            TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
            TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
        },
        {});

    const ValueId shared_id = graph.value_id("shared");
    const auto& order = graph.executor_order();
    const auto& consumers = graph.value(shared_id).consumer();

    assert(consumers.size() == 2);

    std::vector<std::size_t> consumer_positions;
    for (NodeId consumer : consumers) {
        const auto position = std::find(order.cbegin(), order.cend(), consumer);
        assert(position != order.cend());
        consumer_positions.push_back(
            static_cast<std::size_t>(position - order.cbegin()));
    }

    const auto [first_consumer, last_consumer] = std::minmax_element(
        consumer_positions.cbegin(), consumer_positions.cend());
    assert(*first_consumer < *last_consumer);

    const auto span = graph.life_span(shared_id);
    assert(span.first < *first_consumer);
    assert(span.second == *last_consumer);
    assert(span.second != *first_consumer);

    Executor executor(graph);
    assert(executor.get_buffers().size() == 3);

    const ValueId left_id = graph.value_id("left");
    const ValueId right_id = graph.value_id("right");
    const ValueId output_id = graph.value_id("output");

    const StorageId shared_buffer = executor.get_b_id(shared_id);
    const StorageId left_buffer = executor.get_b_id(left_id);
    const StorageId right_buffer = executor.get_b_id(right_id);
    const StorageId output_buffer = executor.get_b_id(output_id);

    assert(shared_buffer == output_buffer);
    assert(shared_buffer != left_buffer);
    assert(shared_buffer != right_buffer);

    StoragePool pool;
    for (auto buffer : executor.get_buffers()) {
        pool.add(buffer);
    }

    auto shared_storage = pool.get(executor.get_buffers()[shared_buffer]);
    const Float32* shared_data = shared_storage->raw_data();
    pool.put(std::move(shared_storage), shared_buffer);

    auto output_storage = pool.get(executor.get_buffers()[output_buffer]);
    assert(output_storage->raw_data() == shared_data);

    auto left_storage = pool.get(executor.get_buffers()[left_buffer]);
    auto right_storage = pool.get(executor.get_buffers()[right_buffer]);
    assert(left_storage->raw_data() != right_storage->raw_data());
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
            TensorDesc(Shape({2, 3}), Backend::CPU, Dtype::Float32),
            TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
            TensorDesc(Shape({2, 2}), Backend::CPU, Dtype::Float32),
        },
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
    assert(graph.initializers().size() == 3);

    Executor executor(graph);
    assert(executor.get_buffers().size() == 2);

    Executor invalid_input_executor(graph);
    expect_throws([&] {
        invalid_input_executor.set_input(
            graph.value_id("weight"),
            Tensor(Shape({3, 2}), {0, 0, 0, 0, 0, 0}, Dtype::Float32));
    });

    executor.set_input(graph.value_id("x"), Tensor(Shape({2, 3}), {1, 2, 3, -1, 0, 2}, Dtype::Float32));
    executor.set_input(graph.value_id("gate"), Tensor(Shape({2, 2}), {2, 0.5, 4, -1}, Dtype::Float32));
    executor.set_input(graph.value_id("residual"), Tensor(Shape({2, 2}), {-1, 1, 2, 20}, Dtype::Float32));
    executor.run();

    const Tensor& output = executor.get_output("output");
    assert(output.shape() == Shape({2, 2}));
    assert(output.backend() == Backend::CPU);
    expect_data_eq(output, {5, 0, 2, 9});
    expect_data_eq(executor.get_tensor(graph.value_id("weight")), {1, -1, 2, 0, -1, 3});

    const std::size_t storage_count_after_first_run = executor.get_buffers().size();
    const Float32* output_data_after_first_run = output.raw_data();

    executor.set_input(graph.value_id("x"), Tensor(Shape({2, 3}), {0, 1, 2, 1, 0, 1}, Dtype::Float32));
    executor.set_input(graph.value_id("gate"), Tensor(Shape({2, 2}), {2, 0.5, 4, -1}, Dtype::Float32));
    executor.set_input(graph.value_id("residual"), Tensor(Shape({2, 2}), {-1, 1, 2, 20}, Dtype::Float32));
    executor.run();

    assert(executor.get_buffers().size() == storage_count_after_first_run);
    assert(executor.get_output("output").raw_data() == output_data_after_first_run);
    expect_data_eq(executor.get_output("output"), {1, 0, 2, 14});

    expect_throws([&] {
        executor.get_output("biased");
    });
    expect_throws([&] {
        executor.get_tensor(graph.value_id("biased"));
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
    test_binary_broadcast_ops();
    test_matmul_op();
    test_graph_validation();
    test_life_span_uses_last_consumer();
    test_complex_graph_execution();

    std::cout << "All runtime tests passed." << std::endl;
    return 0;
}
