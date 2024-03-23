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

    /// Symmetrical to BroadcastRestride. Set the reduced dimensions' stride to
    /// 0 at output. Currently only support the keepdim=true case.
    static void ReductionRestride(ImageRef& dst,
                                  int64_t src_ndims,
                                  const int64_t* src_shape,
                                  const ShapeArray& reduction_dims);

    /// Get input Image data pointer based on \p workload_idx.
    ///
    /// \param input_idx Input image index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    char* GetInputPtr(int64_t input_idx,
                                         int64_t workload_idx) const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr(inputs_[input_idx],
                                  inputs_contiguous_[input_idx], workload_idx);
    }

    /// Get input Image data pointer based on \p workload_idx.
    ///
    /// \param input_idx Input tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    ///
    /// Note: Assumes that sizeof(T) matches the input's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    T* GetInputPtr(int64_t input_idx,
                   int64_t workload_idx) const {
        if (input_idx < 0 || input_idx >= num_inputs_) {
            return nullptr;
        }
        return GetWorkloadDataPtr<T>(inputs_[input_idx],
                                     inputs_contiguous_[input_idx],
                                     workload_idx);
    }

    /// Get output Image data pointer based on \p workload_idx.
    ///
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    char* GetOutputPtr(int64_t workload_idx) const {
        return GetWorkloadDataPtr(outputs_[0], outputs_contiguous_[0],
                                  workload_idx);
    }

    /// Get output Image data pointer based on \p workload_idx.
    ///
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    ///
    /// Note: Assumes that sizeof(T) matches the output's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    T* GetOutputPtr(int64_t workload_idx) const {
        return GetWorkloadDataPtr<T>(outputs_[0], outputs_contiguous_[0],
                                     workload_idx);
    }

    /// Get output Image data pointer based on \p workload_idx.
    ///
    /// \param output_idx Output tensor index.
    /// \param workload_idx The index of the compute workload, similar to
    /// thread_id, if a thread only processes one workload.
    template <typename T>
    T* GetOutputPtr(int64_t output_idx,
                    int64_t workload_idx) const {
        return GetWorkloadDataPtr<T>(outputs_[output_idx],
                                     outputs_contiguous_[output_idx],
                                     workload_idx);
    }


    /// Returns the total number of workloads (e.g. computations) needed for
    /// the op. The scheduler schedules these workloads to run on parallel
    /// threads.
    ///
    /// For non-reduction ops, NumWorkloads() is the same as number of output
    /// elements (e.g. for broadcasting ops).
    ///
    /// For reduction ops, NumWorkLoads() is the same as the number of input
    /// elements. Currently we don't allow mixing broadcasting and reduction in
    /// one op kernel.
    int64_t NumWorkloads() const;

protected:

    /// Merge adjacent dimensions if either dim is 1 or if:
    /// shape[n] * stride[n] == shape[n + 1]
    void CoalesceDimensions();

    // Permute reduction dimensions to front.
    // TODO: Sort the dimensions based on strides in ascending orderto improve
    // thread coalescing.
    void ReorderDimensions(const ShapeArray& reduction_dims);

    /// Update primary_strides_ based on primary_shape_.
    void UpdatePrimaryStrides();

    /// Update input_contiguous_ and output_contiguous_.
    void UpdateContiguousFlags();

    /// Broadcast src to dst by setting shape 1 to omitted dimensions and
    /// setting stride 0 to brocasted dimensions.
    ///
    /// Note that other approaches may also work. E.g. one could set src's shape
    /// to exactly the same as dst's shape. In general, if a dimension is of
    /// size 1, the stride have no effect in computing offsets; or likewise if a
    /// dimension has stride 0, the shape have no effect in computing offsets.
    ///
    /// [Before]
    ///                 Omitted
    ///                 |       Broadcast
    ///                 |       |   No broadcast
    ///                 |       |   |
    ///                 V       V   V
    /// src.shape_:   [     2,  1,  1,  3]
    /// src.strides_: [     3,  3,  3,  1]
    /// dst.shape_:   [ 2,  2,  2,  1,  3]
    /// dst.strides_: [12,  6,  3,  3,  1]
    ///
    /// [After]
    /// src.shape_:   [ 1,  2,  1,  1,  3]
    /// src.strides_: [ 0,  3,  0,  3,  1]
    ///
    /// \param src The source TensorRef to be broadcasted.
    /// \param dst_ndims Number of dimensions to be broadcasted to.
    /// \param dst_shape Shape to be broadcasted to.
    static void BroadcastRestride(ImageRef& src,
                                  int64_t dst_ndims,
                                  const int64_t* dst_shape);

    /// Get data pointer from a TensorRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    char* GetWorkloadDataPtr(const ImageRef& tr,
                             bool tr_contiguous,
                             int64_t workload_idx) const {
        // For 0-sized input reduction op, the output Tensor
        // workload_idx == 1 > NumWorkloads() == 0.
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<char*>(tr.data_ptr_) +
                   workload_idx * tr.dtype_byte_size_;
        } else {
            int64_t offset = 0;
            for (int64_t i = 0; i < ndims_; ++i) {
                offset += workload_idx / primary_strides_[i] *
                          tr.byte_strides_[i];
                workload_idx = workload_idx % primary_strides_[i];
            }
            return static_cast<char*>(tr.data_ptr_) + offset;
        }
    }

    /// Get data pointer from a ImageRef with \p workload_idx.
    /// Note: can be optimized by computing all input ptrs and output ptr
    /// together.
    ///
    /// Note: Assumes that sizeof(T) matches the data's dtype size, but does
    /// not check this constraint for performance reasons.
    template <typename T>
    T* GetWorkloadDataPtr(const ImageRef& tr,
                          bool tr_contiguous,
                          int64_t workload_idx) const {
        // For 0-sized input reduction op, the output Tensor
        // workload_idx == 1 > NumWorkloads() == 0.
        if (workload_idx < 0) {
            return nullptr;
        }
        if (tr_contiguous) {
            return static_cast<T*>(tr.data_ptr_) + workload_idx;
        } else {
            int64_t offset = 0;
            for (int64_t i = 0; i < ndims_; ++i) {
                // (workload_idx / primary_strides_[i]) siempre que workload_idx no sea
                // mayor que los strides de la dimension procesada dara 0, sino dara 1
                // y el offset sera igual al bytestride de esa dimension
                offset += workload_idx / primary_strides_[i] *
                          tr.byte_strides_[i];
                // Cuando el calculo anterior sea igual a byte_stride[i]
                // se reiniciara workload_idx a 0 para la siguiente
                // dimension
                workload_idx = workload_idx % primary_strides_[i];
            }
            return static_cast<T*>(static_cast<void*>(
                    static_cast<char*>(tr.data_ptr_) + offset));
        }
    }

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