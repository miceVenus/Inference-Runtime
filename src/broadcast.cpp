#include "broadcast.hpp"

Shape broadcast_shape(const Shape& lhs, const Shape& rhs){

    auto& lhs_dims = lhs.dims();
    auto& rhs_dims = rhs.dims();

    size_t rank = std::max(lhs_dims.size(), rhs_dims.size());

    std::vector<long> aligned_lhs(rank, 1);
    std::vector<long> aligned_rhs(rank, 1);

    size_t lhs_begin = rank - lhs_dims.size();
    size_t rhs_begin = rank - rhs_dims.size();

    std::copy(lhs_dims.begin(), lhs_dims.end(), aligned_lhs.begin() + lhs_begin);
    std::copy(rhs_dims.begin(), rhs_dims.end(), aligned_rhs.begin() + rhs_begin);

    std::vector<long> dims(rank);

    for(long i = rank - 1; i >= 0; i--){

        if(aligned_lhs[i] == aligned_rhs[i]){
            dims[i] = aligned_lhs[i];
        }else if(aligned_lhs[i] == 1){
            dims[i] = aligned_rhs[i];
        }else if(aligned_rhs[i] == 1){
            dims[i] = aligned_lhs[i];
        }else {
            throw std::runtime_error("shapes are not broadcastable");
        }
    }

    return dims;
}

BroadcastPlan make_broadcast_plan(const Shape& lhs, const Shape& rhs){
    
    BroadcastPlan bp;

    auto& lhs_dims = lhs.dims();
    auto& rhs_dims = rhs.dims();

    size_t rank = std::max(lhs_dims.size(), rhs_dims.size());

    std::vector<long> aligned_lhs(rank, 1);
    std::vector<long> aligned_rhs(rank, 1);

    size_t lhs_begin = rank - lhs_dims.size();
    size_t rhs_begin = rank - rhs_dims.size();

    std::copy(lhs_dims.begin(), lhs_dims.end(), aligned_lhs.begin() + lhs_begin);
    std::copy(rhs_dims.begin(), rhs_dims.end(), aligned_rhs.begin() + rhs_begin);

    std::vector<long> dims(rank);
    std::vector<long> lhs_strides(rank);
    std::vector<long> rhs_strides(rank);

    long l_stride = 1;
    long r_stride = 1;

    for(long i = rank - 1; i >= 0; i--){

        if(aligned_lhs[i] == aligned_rhs[i]){
            dims[i] = aligned_lhs[i];
        }else if(aligned_lhs[i] == 1){
            dims[i] = aligned_rhs[i];
        }else if(aligned_rhs[i] == 1){
            dims[i] = aligned_lhs[i];
        }else {
            throw std::runtime_error("shapes are not broadcastable");
        }
        
        
        if(aligned_lhs[i] == 1){
            lhs_strides[i] = 0;
        }else{
            lhs_strides[i] = l_stride;
            l_stride *= aligned_lhs[i];
        }

        if(aligned_rhs[i] == 1){
            rhs_strides[i] = 0;
        }else{
            rhs_strides[i] = r_stride;
            r_stride *= aligned_rhs[i];
        }
    }

    bp.output_shape = std::move(dims);
    bp.lhs_strides = std::move(lhs_strides);
    bp.rhs_strides = std::move(rhs_strides);

    return bp;
}