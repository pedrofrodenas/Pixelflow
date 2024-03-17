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

    /// \brief Permute (dimension shuffle) the reference to a Tensor.
    ///
    /// \param dims The desired ordering of dimensions.
    ///
    /// Note: This only affects this Tensor reference, but not the underlying
    /// Tensor.
    void Permute(const ShapeArray& dims) {

        if (dims.GetDims() != ndims_) {
            std::ostringstream oss;
            oss << "Number of dimensions mismatch " << dims.GetDims()
            << " != " << ndims_;
            LogError(oss.str().c_str());
        }

        std::vector<bool> seen_dims(ndims_, false);
        for (const int64_t &dim : dims) {
            seen_dims[dim] = true;
        }

        if (!std::all_of(seen_dims.cbegin(), seen_dims.cend(),
                        [] (const bool seen) { return seen; })) {
            std::ostringstream oss;
            oss << "Permute dims must be a permuntation from 0 to " << dims.GetDims() -1;
            LogError(oss.str().c_str());
        }

        // Map new shape
        ShapeArray new_shape(ndims_);
        ShapeArray new_byte_stride(ndims_);

        for (int64_t i = 0; i < ndims_; ++i) {
            int64_t old_dim = WrapDim(dims[i], dims.GetDims());
            new_shape[i] = shape_[old_dim];
            new_byte_stride[i] = byte_strides_[old_dim];
        }
        for (int64_t i = 0; i < ndims_; ++i) {
            shape_[i] = new_shape[i];
            byte_strides_[i] = new_byte_stride[i];
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