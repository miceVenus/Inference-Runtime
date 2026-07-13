#ifndef STORAGE_HPP
#define STORAGE_HPP

#include "dtype.hpp"
#include "backend.hpp"
#include <vector>
#include <cstddef>
#include <utility>
#include <memory>



class Storage{
    public:
        virtual ~Storage() = default;

        virtual Backend backend() const noexcept = 0;
        virtual std::size_t size() const noexcept = 0;
        virtual std::size_t size_bytes() const noexcept = 0;

        virtual Float32 * raw_data() = 0;
        virtual const Float32 * raw_data() const = 0;

        virtual std::unique_ptr<Storage> clone() const = 0;
};

class CpuStorage final : public Storage{

    public:
        explicit CpuStorage(std::size_t numel)
        :data_(numel, 0){}

        CpuStorage(std::vector<Float32> data)
        :data_(std::move(data)){}


        CpuStorage(const CpuStorage &storage)
        :data_(storage.data_){}


        CpuStorage(CpuStorage &&storage) noexcept
        :data_(std::move(storage.data_)){}


        Backend backend() const noexcept override{
            return Backend::CPU;
        }
        std::size_t size() const noexcept override{
            return data_.size();
        }

        std::size_t size_bytes() const noexcept override{
            return data_.size() * sizeof(Float32);
        }

        Float32 * raw_data() override {
            return data_.data();
        }

        const Float32 * raw_data() const override{
            return data_.data();
        }

        Float32& operator[](std::size_t index){
            return data_.at(index);
        }

        const Float32& operator[](std::size_t index) const{
            return data_.at(index);
        }

        CpuStorage& operator=(CpuStorage storage) {
            data_ = std::move(storage.data_);
            return *this;
        }

        std::unique_ptr<Storage> clone() const override{
            return std::make_unique<CpuStorage>(*this);
        }

    private:
        std::vector<Float32> data_;

};



#endif