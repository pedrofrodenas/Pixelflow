#include "pixelflow/core/Image.h"

namespace pixelflow {
namespace core {

    Device Image::GetDevice() const {

        if (blob_ == nullptr) {
            LogError("Blob is null, there is no blob of memory yet");
        }
        return blob_->GetDevice();
    }

    struct Image::Iterator::Impl {
        Image* image_;
        int64_t index_;
        Image image_slice_;  // Stores temporary tensor slice with shared memory
        // as the original tensor. This allows taking the &
        // of the tensor for Iterator::operator->.
    };

    Image::Iterator::Iterator(pointer image, int64_t index)
    : impl_(std::make_unique<Impl>()) {
        impl_->image_ = image;
        impl_->index_ = index;
    }

    Image::Iterator::Iterator(const Image::Iterator& other)
        : impl_(std::make_unique<Impl>()) {
        impl_->image_ = other.impl_->image_;
        impl_->index_ = other.impl_->index_;
    }

    // Empty destructor since Impl is incomplete type in Tensor.h.
    // https://stackoverflow.com/a/34073093/1255535
    Image::Iterator::~Iterator() {}

    Image::Iterator& Image::Iterator::operator++() {
        impl_->index_++;
        return *this;
    }

    Image::Iterator Image::Iterator::operator++(int) {
        Iterator tmp(impl_->image_, impl_->index_);
        impl_->index_++;
        return tmp;
    }

    bool Image::Iterator::operator==(const Image::Iterator& other) const {
        return impl_->image_ == other.impl_->image_ &&
               impl_->index_ == other.impl_->index_;
    }

    bool Image::Iterator::operator!=(const Image::Iterator& other) const {
        return !(*this == other);
    }



    Image::Iterator Image::begin() {
        if (NumDims() == 0) {
            LogError("Cannot iterate a scalar (0-dim) tensor.");
        }
        return Iterator(this, 0);
    }

    Image::Iterator Image::end() {
        if (NumDims() == 0) {
            LogError("Cannot iterate a scalar (0-dim) tensor.");
        }
        return Iterator(this, shape_[0]);
    }


} // namespace core
} // namespace pixelflow
