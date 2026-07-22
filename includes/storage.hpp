#ifndef STORAGE_HPP
#define STORAGE_HPP

#include "dtype.hpp"
#include "backend.hpp"

#include <vector>
#include <cstddef>
#include <utility>
#include <memory>
#include <algorithm>

using StorageId=std::size_t;

class TensorDesc;

class StorageDesc{
    public:
        StorageDesc(const TensorDesc & desc);
        
        size_t size_bytes;
        Backend backend;

        StorageId id;
        size_t numel;
        Dtype dtype;
        size_t ava_after;
};

class Storage{
    public:

        virtual ~Storage() = default;

        virtual Backend backend() const noexcept = 0;
        virtual std::size_t size() const noexcept = 0;
        virtual std::size_t size_bytes() const noexcept = 0;

        virtual Float32 * raw_data() = 0;
        virtual const Float32 * raw_data() const = 0;

        virtual std::unique_ptr<Storage> clone() const = 0;

        virtual void fill(Float32 value) = 0;
};

class CpuStorage final : public Storage{

    public:
        explicit CpuStorage(std::size_t numel);

        CpuStorage(std::vector<Float32> data);


        CpuStorage(const CpuStorage &storage);


        CpuStorage(CpuStorage &&storage) noexcept;


        Backend backend() const noexcept override;

        std::size_t size() const noexcept override;

        std::size_t size_bytes() const noexcept override;

        Float32 * raw_data() override;

        const Float32 * raw_data() const override;

        Float32& operator[](std::size_t index);

        const Float32& operator[](std::size_t index) const;

        CpuStorage& operator=(CpuStorage storage);

        std::unique_ptr<Storage> clone() const override;

        void fill(Float32 value) override;

    private:
        std::vector<Float32> data_;

};



#endif