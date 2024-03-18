#pragma once

#include "pixelflow/core/Image.h"

// Adapted for Pixelflow from Open3D project.
// Commit e39c1e0994ac2adabd8a617635db3e35f04cce88 2023-03-10
// Documentation:
// https://www.open3d.org/docs/0.12.0/cpp_api/classopen3d_1_1core_1_1_indexer.html
//
//===- Open3d/cpp/open3d/core/Indexer.h -===//
// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

namespace pixelflow {
namespace core {

// Maximum number of dimensions of ImageRef.
static constexpr int64_t MAX_DIMS = 4;

// Maximum number of inputs of an op.
// MAX_INPUTS shall be >= MAX_DIMS to support advanced indexing.
static constexpr int64_t MAX_INPUTS = 10;

// Maximum number of outputs of an op. This number can be increased when
// necessary.
static constexpr int64_t MAX_OUTPUTS = 2;


struct ImageRef {
    // The default copy constructor works on __device__ as well so we don't
    // define it explicitly. shape_[MAX_DIMS] and strides[MAX_DIMS] will be
    // copied fully.
    ImageRef() : data_ptr_(nullptr), ndims_(0), dtype_byte_size_(0) {}

    ImageRef(const Image& t) {

        data_ptr_ = const_cast<void*>(t.GetDataPtr());
        ndims_ = t.NumDims();
        dtype_byte_size_ = t.GetDtype().getSize();
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

    inline bool IsContiguous() const {
        ShapeArray thisStride(ndims_);
        ShapeArray thisShape(ndims_);
        for (int64_t i=0; i<ndims_; ++i) {
            thisStride[i] = byte_strides_[i]/dtype_byte_size_;
            thisShape[i] = shape_[i];
        }
        return DefaultStrides(thisShape) == thisStride;
    }

    bool operator==(const ImageRef& other) const {
        bool eq = true;
        eq = eq && (data_ptr_ == other.data_ptr_);
        eq = eq && (ndims_ == other.ndims_);
        eq = eq && (dtype_byte_size_ == other.dtype_byte_size_);
        for (int64_t i = 0; i != ndims_; ++i) {
            eq = eq && (shape_[i] == other.shape_[i]);
            eq = eq && (byte_strides_[i] == other.byte_strides_[i]);
        }
        return eq;
    }

    bool operator!=(const ImageRef& other) const {
        return !(*this == other);
    }

    void* data_ptr_;
    int64_t ndims_ = 0;
    int64_t dtype_byte_size_ = 0;
    int64_t shape_[MAX_DIMS];
    int64_t byte_strides_[MAX_DIMS];

};

enum class DtypePolicy {
    NONE,        // Do not check. Expects the kernel to handle the conversion.
                 // E.g. in Copy kernel with type casting.
    ALL_SAME,    // All inputs and outputs to to have the same dtype.
    INPUT_SAME,  // All inputs have the same dtype.
    INPUT_SAME_OUTPUT_BOOL  // All inputs have the same dtype. Outputs
                            // have bool dtype.
};

class Indexer {
public:
    Indexer() {}
    Indexer(const Indexer&) = default;
    Indexer& operator=(const Indexer&) = default;

    Indexer(const std::vector<Image>& input_images,
            const Image& output_image,
            DtypePolicy dtype_policy = DtypePolicy::ALL_SAME,
            const ShapeArray& reduction_dims = {});

    Indexer(const std::vector<Image>& input_images,
            const std::vector<Image>& output_images,
            DtypePolicy dtype_policy = DtypePolicy::ALL_SAME,
            const ShapeArray& reduction_dims = {});

protected:

    /// Number of input and output Tensors.
    int64_t num_inputs_ = 0;
    int64_t num_outputs_ = 0;

    /// Array of input TensorRefs.
    ImageRef inputs_[MAX_INPUTS];

    /// Array of output TensorRefs.
    ImageRef outputs_[MAX_OUTPUTS];

    /// Array of contiguous flags for all input TensorRefs.
    bool inputs_contiguous_[MAX_INPUTS];

    /// Array of contiguous flags for all output TensorRefs.
    bool outputs_contiguous_[MAX_OUTPUTS];

    /// Indexer's global shape. The shape's number of elements is the
    /// same as GetNumWorkloads() for the Indexer.
    /// - For broadcasting, primary_shape_ is the same as the output shape.
    /// - For reduction, primary_shape_ is the same as the input shape.
    /// - Currently we don't allow broadcasting mixed with reduction. But if
    ///   broadcasting mixed with reduction is allowed, primary_shape_ is a mix
    ///   of input shape and output shape. First, fill in all omitted dimensions
    ///   (in inputs for broadcasting) and reduction dimensions (as if
    ///   keepdim=true always) with size 1. For each axis, the primary dimension
    ///   is the non-1 dimension (if both are 1, then the primary dimension is 1
    ///   in that axis).
    int64_t primary_shape_[MAX_DIMS];

    /// The default strides for primary_shape_ for internal use only. Used to
    /// compute the actual strides and ultimately the index offsets.
    int64_t primary_strides_[MAX_DIMS];

    /// Indexer's global number of dimensions.
    int64_t ndims_ = 0;

    /// Whether this iterator produces the actual output, as opposed to
    /// something that will be accumulated further. Only relevant for CUDA
    /// reductions.
    bool final_output_ = true;

    /// If the kernel should accumulate into the output. Only relevant for CUDA
    /// reductions.
    bool accumulate_ = false;
};



} //namespace core
} // namespace pixelflow