#pragma once

#include <type_traits>


#include "pixelflow/core/Broadcasting.h"
#include "pixelflow/core/Device.h"
#include "pixelflow/core/ShapeArray.h"
#include "pixelflow/core/dtypes.h"
#include "pixelflow/core/Blob.h"
#include "pixelflow/core/MemoryManager.h"
#include "pixelflow/core/TensorKey.h"
#include "pixelflow/core/Dispatch.h"

namespace pixelflow {
namespace core {

class Image : public IsDevice{

public:

    Image() {}

    // Contiguos memory Image constructor
    Image(const ShapeArray& shape,
          PfType dtype,
          const Device& device)
              : shape_(shape),
                strides_(DefaultStrides(shape)),
                dtype_(dtype),
                blob_(std::make_shared<Blob>(shape.NumElements()*dtype.getSize(),device)) {
        data_ptr_ = blob_->GetDataPtr();
    }

    // Constructor with initial values
    template<typename T>
    Image(const std::vector<T>& init_vals,
          const ShapeArray& shape,
          PfType dtype,
          const Device& device)
              : Image(shape, dtype, device) {

        // Check if datatype provided match T datatype
        if (dtype.getSize() != sizeof(T)) {
            LogError("init_vals datatype not match provided dtype");
        }

        // TODO: Check if names of datatype match

        // Check if values of vector are C Stardar-layout types
        if (!std::is_pod<T>()) {
            LogError("Object in vector is not a C basic type (POD)");
        }

        // TODO:
        MemoryManager::MemcpyFromHost(blob_->GetDataPtr(),
                                      blob_->GetDevice(),
                               init_vals.data(),
                             init_vals.size() * dtype.getSize());
    }

    /// The fully specified constructor. Since you're responsible for creating
    /// the Blob, take care of Blob's deleter if the memory is allocated
    /// elsewhere. See Blob.h for more details.
    Image(const ShapeArray& shape,
          const ShapeArray& stride,
          void* data_ptr,
          PfType dtype,
          const std::shared_ptr<Blob>& blob)
              : shape_(shape), strides_(stride),
                data_ptr_(data_ptr), dtype_(dtype),
                blob_(blob) {}

    inline bool IsContiguous() const {
        return strides_ == DefaultStrides(shape_);
    }

    /// Returns a contiguous Image containing the same data in the same device.
    /// If self tensor is already contiguous, the same underlying memory will be
    /// used.
    Image Contiguous() const;

    Device GetDevice() const override;

    inline ShapeArray Shape() const {return shape_;}

    inline ShapeArray Stride() const {return strides_;}

    inline void* GetDataPtr() {return data_ptr_;}
    inline const void* GetDataPtr() const {return data_ptr_;}
    inline PfType GetDtype() const {return dtype_;}

    inline int64_t GetShape(int64_t dim) const {
        return shape_[WrapDim(dim, NumDims())];
    }

    inline int64_t GetStride(int64_t dim) const {
        return strides_[WrapDim(dim, NumDims())];
    }

    inline int64_t NumElements() const { return shape_.NumElements(); }

    inline int64_t NumDims() const { return shape_.GetDims(); }

    /// Pythonic __getitem__ for image.
    ///
    /// Returns a view of the original image, if TensorKey is
    /// TensorKeyMode::Index or TensorKeyMode::Slice. Returns a copy if the
    /// TensorKey contains TensorKeyMode::IndexTensor (advanced indexing).
    ///
    /// For example, in numpy:
    /// ```python
    /// t = np.empty((4, 5), dtype=np.float32)
    /// t1 = t[2]
    /// t2 = t[0:4:2]
    /// ```
    ///
    /// ```cpp
    /// Image t({4, 5}, core::Float32);
    /// Image t1 = t.GetItem(TensorIndex(2));
    /// Image t2 = t.GetItem(TensorSlice(0, 4, 2));
    /// ```
    Image GetItem(const TensorKey& tk) const;

    /// Pythonic __getitem__ for image.
    ///
    /// Returns a view of the original image, if TensorKey only contains
    /// TensorKeyMode::Index or TensorKeyMode::Slice. Returns a copy if the
    /// TensorKey contains IndexTensor (advanced indexing).
    ///
    /// For example, in numpy:
    /// ```python
    /// t = np.empty((4, 5), dtype=np.float32)
    /// t1 = t[1, 0:4:2]
    /// ```
    ///
    /// ```cpp
    /// Tensor t({4, 5}, core::Float32);
    /// Tensor t1 = t.GetItem({TensorIndex(2), TensorSlice(0, 4, 2)});
    /// ```
    ///
    Image GetItem(const std::vector<TensorKey>& tks) const;

    /// Slice Image.
    ///
    /// \param dim The dimension to slice.
    /// \param start The start index (inclusive).
    /// \param stop The end index (exclusive).
    /// \param step Pick one element for every \p step elements.
    Image Slice(int64_t dim, int64_t start, int64_t stop, int64_t step) const;

    /// Extract the i-th Image along the first axis, returning a new view.
    Image operator[](int64_t i) const;

    /// Extract the \p idx -th sub-image in dimension \p dim. After
    /// IndexExtract, the dimension \p dim will be removed.
    Image IndexExtract(int64_t dim, int64_t idx) const;

    /// Returns a image with the specified \p device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new tensor is always created; if false, the copy
    /// is avoided when the original image is already on the targeted device.
    Image To(const Device& device, bool copy = false) const;

    std::string ToString(bool with_suffix = true,
                         const std::string& indent = "") const;

    std::string ScalarPtrToString(const void* ptr) const;



    /// Iterator for Image.
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Image;
        using pointer = value_type*;
        using reference = value_type;  // Typically Image&, but a image slice
        // creates a new Image object with
        // shared memory.

        // Iterator must be constructible, copy-constructible, copy-assignable,
        // destructible and swappable.
        Iterator(pointer image, int64_t index);
        Iterator(const Iterator&);
        ~Iterator();
        Iterator& operator++();
        Iterator operator++(int);
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    /// Returns the beginning of the tensor iterator. The iterator iterates over
    /// the first dimension of the tensor. The generated tensor slices share the
    /// same memory with the original tensor.
    Iterator begin();

    /// Returns the end of the tensor iterator. The iterator iterates over the
    /// first dimension of the tensor. The generated tensor slices share the
    /// same memory with the original tensor.
    Iterator end();


protected:
    /// The shape of the image
    ShapeArray shape_ = {0};

    ShapeArray strides_ = {1};

    /// Data pointer pointing to the beginning element of the Image
    void* data_ptr_ = nullptr;

    /// DataType
    PfType dtype_ = PfType::Undefined;

    /// Underlying memory buffer for Tensor.
    std::shared_ptr<Blob> blob_ = nullptr;
};

}
}



