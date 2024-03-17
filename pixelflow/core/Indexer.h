#pragma once

#include "pixelflow/core/Image.h"

namespace pixelflow {
namespace core {

// Maximum number of dimensions of ImageRef.
static constexpr int64_t MAX_DIMS = 4;


struct ImageRef {
    // The default copy constructor works on __device__ as well so we don't
    // define it explicitly. shape_[MAX_DIMS] and strides[MAX_DIMS] will be
    // copied fully.
    ImageRef() : data_ptr_(nullptr), ndims_(0), dtype_byte_size_(0) {}

    ImageRef(const Image& t) {

        data_ptr_ = const_cast<void*>(t.GetDataPtr());
        ndims_ = t.NumDims();
        dtype_byte_size_ = t.GetDType().getSize();
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = t.GetShape(i);
            byte_strides_[i] = t.GetStride(i) * dtype_byte_size_;
        }
    }



    void* data_ptr_;
    int64_t ndims_ = 0;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t byte_strides_[MAX_DIMS];

};
} //namespace core
} // namespace pixelflow