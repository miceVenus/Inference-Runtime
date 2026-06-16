#include "tensor.hpp"
#include <iostream>

int main(){
    Shape shape = Shape({1, 2, 3});
    Tensor tensor = Tensor(shape);
    Tensor tensor2 = Tensor(shape, {1, 2, 3, 4, 5, 6});
    std::cout << tensor.shape() << std::endl;
    print(tensor2);
}