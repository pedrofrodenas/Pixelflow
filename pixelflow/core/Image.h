#pragma once

#include <type_traits>


#include "Broadcasting.h"
#include "pixelflow/core/Device.h"
#include "pixelflow/core/ShapeArray.h"
#include "pixelflow/core/dtypes.h"
#include "pixelflow/core/Blob.h"
#include "pixelflow/core/MemoryManager.h"

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
                blob_(std::make_shared<Blob>(shape.NumElems()*dtype.getSize(),device)) {
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

    Device GetDevice() const override;

    inline ShapeArray Shape() const {return shape_;}

    inline void* GetDataPtr() {return data_ptr_;}
    inline const void* GetDataPtr() const {return data_ptr_;}

    inline int64_t NumElements() const { return shape_.NumElems(); }

    inline int64_t NumDims() const { return shape_.GetDims(); }


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



