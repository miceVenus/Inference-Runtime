#ifndef STORAGE_POOL_HPP
#define STORAGE_POOL_HPP

#include "storage.hpp"
#include "dtype.hpp"
#include "tensor.hpp"
#include "backend.hpp"
#include <vector>

class StoragePool{

    public:
        std::unique_ptr<Storage> get(TensorDesc t_T){
            switch (t_T.backend_){
                case Backend::CPU:{
                    auto ptr = select_from_cpu_pool(dtype_size(t_T.dtype_) * t_T.shape_.numel());

                    if(ptr){
                        return ptr;
                    }else{
                        return StorageFac::create(t_T.shape_.numel(), Backend::CPU);
                    }

                    break;
                }
                
                case Backend::CUDA:
                    throw std::runtime_error("CUDA STORAGE IS NOT SUPPORTED");

                default:
                    throw std::runtime_error("Unknown Backend");
            }
        }

        std::unique_ptr<Storage> get(const StorageDesc & s_desc){
            switch (s_desc.backend){
                case Backend::CPU:{
                    return std::move(cpu_pool_[s_desc.id]);
                }
                
                case Backend::CUDA:
                    throw std::runtime_error("CUDA STORAGE IS NOT SUPPORTED");

                default:
                    throw std::runtime_error("Unknown Backend");
            }
        }

        void add(StorageDesc & t){
            put(StorageFac::create(t.numel, t.backend));
        }

        void put(std::unique_ptr<Storage> ptr){
            switch (ptr->backend()){
                case Backend::CPU:{
                    cpu_pool_.push_back(std::move(ptr));
                    break;
                }
                
                case Backend::CUDA:
                    throw std::runtime_error("CUDA STORAGE IS NOT SUPPORTED");

                default:
                    throw std::runtime_error("Unknown Backend");
            }
        }

        void put(std::unique_ptr<Storage> ptr, StorageId s_id){

            switch (ptr->backend()){
                case Backend::CPU:{
                    cpu_pool_[s_id] = std::move(ptr);
                    break;
                }
                
                case Backend::CUDA:
                    throw std::runtime_error("CUDA STORAGE IS NOT SUPPORTED");

                default:
                    throw std::runtime_error("Unknown Backend");
            }
        }




    private:
        std::vector<std::unique_ptr<Storage>> cpu_pool_;

        std::unique_ptr<Storage> select_from_cpu_pool(std::size_t bytes){
            auto it = std::find_if(cpu_pool_.begin(), cpu_pool_.end(), [bytes](const auto & i){return i->size_bytes() == bytes;});
            if (it == cpu_pool_.end()) {
                return nullptr;
            }

            auto result = std::move(*it);
            cpu_pool_.erase(it);

            return result;
        }

        void put_into_cpu_pool(std::unique_ptr<Storage> ptr){
            if(ptr->backend() != Backend::CPU){
                throw std::runtime_error("the other device want to in cpu storage pool");
            }
            cpu_pool_.push_back(std::move(ptr));
        }

};


#endif