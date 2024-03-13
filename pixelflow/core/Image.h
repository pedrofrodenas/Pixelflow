#pragma once

#include <type_traits>


#include "Broadcasting.h"
#include "pixelflow/core/Device.h"
#include "pixelflow/core/ShapeArray.h"
#include "pixelflow/core/dtypes.h"
#include "pixelflow/core/Blob.h"

namespace pixelflow {
namespace core {

class Image : public IsDevice{

public:

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



