#pragma once
#include "Broadcasting.h"
#include "pixelflow/core/Device.h"
#include "pixelflow/core/ShapeArray.h"
#include "pixelflow/core/dtypes.h"
#include "pixelflow/core/Blob.h"

namespace pixelflow {
namespace core {

class Image : public IsDevice{

public:

    Image(const ShapeArray& shape,
          PfType dtype,
          const Device& device)
              : shape_(shape),
                strides_(DefaultStrides(shape)),
                dtype_(dtype),
                blob_(std::make_shared<Blob>(shape.NumElems()*dtype.getSize(),device)) {
        data_ptr_ = blob_->GetDataPtr();
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



