#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <memory>

#include "shape.hpp"
#include "storage.hpp"
#include "storage_fac.hpp"
#include "backend.hpp"

class TensorDesc;

class Tensor{
    public:
        explicit Tensor(const TensorDesc &T);
        Tensor(const TensorDesc &T, std::unique_ptr<Storage> storage);

        Tensor(const Shape & shape, Dtype dtype, Backend backend = Backend::CPU);

        Tensor(const Shape & shape, std::vector<Float32> data, Dtype dtype, Backend backend = Backend::CPU);

        Tensor(const Tensor & tensor);

        Tensor(Tensor && other) noexcept = default;

        ~Tensor() = default;

        Tensor() = default;

        Tensor& operator=(Tensor && other) noexcept = default;

        Tensor& operator=(const Tensor & tensor);

        std::unique_ptr<Storage> move_storage();

        Float32 * raw_data();

        const Float32 * raw_data() const;

        Float32& operator[](std::size_t index);

        const Float32& operator[](std::size_t index) const;



        const Shape& shape() const;

        long numel() const;

        const Dtype& dtype() const;

        Backend backend() const;

        bool is_cpu() const;

        bool is_cuda() const;

        Tensor& fill(Float32 value);

    private:
        Shape shape_;
        std::unique_ptr<Storage> storage_;
        Dtype dtype_ = Dtype::Float32;
};

class TensorDesc{
    public:
        Shape shape_;
        Backend backend_;
        Dtype dtype_;

        TensorDesc(const Shape shape, Backend backend, Dtype dtype);

        explicit TensorDesc(const Tensor & t);

        bool operator==(const TensorDesc & t_T) const{
            return (shape_ == t_T.shape_ && backend_ == t_T.backend_ && dtype_ == t_T.dtype_);
        }

        bool operator!=(const TensorDesc & t_T) const{
            return !(*this == t_T);
        }
};

void print(const Tensor &tensor);

#endif