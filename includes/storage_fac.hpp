#ifndef STORAGE_FAC_HPP
#define STORAGE_FAC_HPP


#include "storage.hpp"
#include <memory>

class StorageFac{

    public:
        static std::unique_ptr<Storage> create(std::size_t numel, Backend backend){

            switch (backend){
                case Backend::CPU:
                    return std::make_unique<CpuStorage>(numel);
                
                case Backend::CUDA:
                    return std::make_unique<CudaStorage>(numel);
            }
            throw std::runtime_error("Unknown backend");
        }


        static std::unique_ptr<Storage> create(std::vector<Float32> data, Backend backend){

            switch (backend){
                case Backend::CPU:
                    return std::make_unique<CpuStorage>(std::move(data));
                
                case Backend::CUDA:
                    return std::make_unique<CudaStorage>(std::move(data));
            }
            throw std::runtime_error("Unknown backend");
        }

    private:
};



#endif