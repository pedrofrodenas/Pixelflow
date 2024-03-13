#include "pixelflow/core/Image.h"

namespace pixelflow {
namespace core {

    Device Image::GetDevice() const {

        if (blob_ == nullptr) {
            LogError("Blob is null, there is no blob of memory yet");
        }
        return blob_->GetDevice();
    }


} // namespace core
} // namespace pixelflow
