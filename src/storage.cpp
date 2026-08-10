#include <cuda_runtime.h>

#include "storage.hpp"
#include "tensor.hpp"


using StorageId=std::size_t;


StorageDesc::StorageDesc(const TensorDesc & desc):
size_bytes(desc.shape_.numel() * dtype_size(desc.dtype_)), 
backend(desc.backend_),
numel(desc.shape_.numel()),
dtype(desc.dtype_){}


CpuStorage::CpuStorage(std::size_t numel)
:data_(numel, 0){}

CpuStorage::CpuStorage(std::vector<Float32> data)
:data_(std::move(data)){}


CpuStorage::CpuStorage(const CpuStorage &storage)
:data_(storage.data_){}


CpuStorage::CpuStorage(CpuStorage &&storage) noexcept
:data_(std::move(storage.data_)){}


Backend CpuStorage::backend() const noexcept{
    return Backend::CPU;
}
std::size_t CpuStorage::size() const noexcept{
    return data_.size();
}

std::size_t CpuStorage::size_bytes() const noexcept{
    return data_.size() * sizeof(Float32);
}

Float32 * CpuStorage::raw_data(){
    return data_.data();
}

const Float32 * CpuStorage::raw_data() const{
    return data_.data();
}

Float32& CpuStorage::operator[](std::size_t index){
    return data_.at(index);
}

const Float32& CpuStorage::operator[](std::size_t index) const{
    return data_.at(index);
}

CpuStorage& CpuStorage::operator=(CpuStorage storage) {
    data_ = std::move(storage.data_);
    return *this;
}

std::unique_ptr<Storage> CpuStorage::clone() const{
    return std::make_unique<CpuStorage>(*this);
}

void CpuStorage::fill(Float32 value){
    std::fill(data_.begin(), data_.end(), value);
}




CudaStorage::CudaStorage(std::size_t numel)
:size_(numel){
    cudaMalloc(&data_, numel * sizeof(Float32));
    fill(0);
}

CudaStorage::CudaStorage(std::vector<Float32> data)
:size_(data.size()){
    cudaMalloc(&data_, data.size() * sizeof(Float32));
    cudaMemcpy(data_, data.data(), data.size()* sizeof(Float32), cudaMemcpyHostToDevice);
}


CudaStorage::CudaStorage(const CudaStorage &storage)
:size_(storage.size()){
    cudaMemcpy(data_, storage.data_, storage.size_bytes(), cudaMemcpyDeviceToDevice);
}

CudaStorage::CudaStorage(CudaStorage &&storage) noexcept
:data_(std::move(storage.data_)), size_(storage.size()){}

Backend CudaStorage::backend() const noexcept{
    return Backend::CUDA;
}

std::size_t CudaStorage::size() const noexcept{
    return size_;
}

std::size_t CudaStorage::size_bytes() const noexcept{
    return size_ * sizeof(Float32);
}

Float32 * CudaStorage::raw_data(){
    return data_;
}

const Float32 * CudaStorage::raw_data() const{
    return data_;
}

CudaStorage& CudaStorage::operator=(CudaStorage storage){
    data_ = std::move(storage.data_);
    return *this;
}

std::unique_ptr<Storage> CudaStorage::clone() const{
    return std::make_unique<CudaStorage>(*this);
}

void CudaStorage::fill(Float32 value){
    cudaMemset(data_, value, size_bytes());
}