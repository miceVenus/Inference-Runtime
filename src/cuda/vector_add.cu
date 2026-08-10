#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void vector_add(const float* lhs, const float* rhs, float* output, int numel){
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < numel){
        output[index] = lhs[index] + rhs[index];
    }
}

int main(){
    constexpr int numel = 300000000;

    const std::size_t bytes = numel * sizeof(float);

    std::vector<float> lhs(numel);
    std::vector<float> rhs(numel);
    std::vector<float> output(numel);

    for (int i = 0; i < numel; ++i) {
        lhs[i] = static_cast<float>(i);
        rhs[i] = static_cast<float>(i * 2);
    }

    float* device_lhs = nullptr;
    float* device_rhs = nullptr;
    float* device_output = nullptr;

    cudaMalloc(&device_lhs, bytes);
    cudaMalloc(&device_rhs, bytes);
    cudaMalloc(&device_output, bytes);

    cudaMemcpy(device_lhs, lhs.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rhs, rhs.data(), bytes, cudaMemcpyHostToDevice);

    constexpr int thread_per_block = 256;
    const int blocks = (numel + thread_per_block - 1) / thread_per_block;
    vector_add<<<blocks, thread_per_block>>>(
        device_lhs,
        device_rhs,
        device_output,
        numel
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cerr
            << "Kernel launch failed: "
            << cudaGetErrorString(error)
            << '\n';

        return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(
        output.data(),
        device_output,
        bytes,
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < numel; ++i) {
        const float expected = lhs[i] + rhs[i];

        if (output[i] != expected) {
            std::cerr << "Wrong result at index " << i << '\n';
            return EXIT_FAILURE;
        }
    }

    cudaFree(device_output);
    cudaFree(device_rhs);
    cudaFree(device_lhs);

    std::cout << "CUDA vector add passed\n";
    return EXIT_SUCCESS;

}