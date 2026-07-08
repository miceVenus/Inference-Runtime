#include "tensor.hpp"
#include "addop.hpp"
#include "node.hpp"
#include "graph.hpp"
#include "executor.hpp"
#include "matmulop.hpp"

#include "operation.hpp"
#include <memory>
#include <iostream>

int main(){
    Shape shape = Shape({1, 2, 3});

    Tensor tensor   = Tensor(shape, {2, 3, 5, 6, 7, 8}, Dtype::Float32);
    Tensor tensor2  = Tensor(shape, {1, -2, 3, -4, 5, -6}, Dtype::Float32);
    Tensor tensor3  = Tensor(shape, {0, 1, 1, 0 ,1, 0}, Dtype::Float32);

    Shape m_shape1 = Shape({2, 3});
    Shape m_shape2 = Shape({3, 2});

    Tensor m_tensor2  = Tensor(m_shape1, {1, 2, 3, 4, 5, 6}, Dtype::Float32);
    Tensor m_tensor3  = Tensor(m_shape2, {1, 2, 3, 4 ,5, 6}, Dtype::Float32);



    // std::cout << tensor.shape() << std::endl;
    // print(tensor2);
    // std::cout << dtype2str(tensor.dtype())  << std::endl;
    // print(op1.forward(tensor, tensor2));

    Node add_0  = Node({"a", "b"}, {"c"}, "add_0", OP_TYPE::AddOp);
    Node add_1  = Node({"c", "a"}, {"d"}, "add_1", OP_TYPE::AddOp);
    Node mul_0  = Node({"d", "c"}, {"e"}, "mul_0", OP_TYPE::MulOp);
    Node relu_0  = Node({"e"}, {"relu_e"}, "relu_0", OP_TYPE::ReluOp);
    Node mask_0   = Node({"relu_e", "mask"}, {"masked_relu_e"}, "mask_0", OP_TYPE::MulOp);

    
    try{
        Graph toy_graph = Graph({add_0, add_1, mul_0, relu_0, mask_0}, {"a", "b", "mask"}, {"masked_relu_e"}, "toy_graph");
        // toy_graph.show_graph_info();
        Executor e = Executor(toy_graph);
        e.set_input("a", tensor);
        e.set_input("b", tensor2);
        e.set_input("mask", tensor3);
        e.run();
        
        print(e.get_output("masked_relu_e"));

        print(MatMulOp::forward(m_tensor2, m_tensor3));
        
    }   
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}