#include "tensor.hpp"
#include "addop.hpp"
#include "node.hpp"
#include <iostream>

int main(){
    Shape shape = Shape({1, 2, 3});
    AddOp op1;

    Tensor tensor = Tensor(shape, {2, 3, 5, 6, 7, 8}, Dtype::Float32);
    Tensor tensor2 = Tensor(shape, {1, 2, 3, 4, 5, 6}, Dtype::Float32);
    std::cout << tensor.shape() << std::endl;
    print(tensor2);
    std::cout << dtype2str(tensor.dtype())  << std::endl;
    print(op1.forward(tensor, tensor2));

    Node add_0 = Node({"a", "b"}, {"c"}, "add_0", OP_TYPE::AddOp);
    add_0.show();
}