#include "pixelflow/core/Image.h"
#include "pixelflow/core/kernel/Copy.h"



namespace pixelflow {
namespace core {

    Image Image::SetItem(const Image& value) {
        this->AsRvalue() = value;
        return *this;
    }

    Image Image::Slice(int64_t dim, int64_t start, int64_t stop, int64_t step) const {

        if (shape_.GetDims() == 0) {
            LogError("Image is 0-dim");
        }

        if (step <= 0) {
            LogError("Step cannot be 0 or less");
        }

        // Convert dim to positive counterpart
        dim = WrapDim(dim, shape_.GetDims());

        // Firstly warp start
        if (start < 0) {
            start += shape_[dim];
        }
        // If after warping is still negative, put the default value
        if (start < 0) {
            start = 0;
        }
        // If start is greater than lenght of image in this dimensions, empty image
        else if (start >= shape_[dim]) {
            start = shape_[dim];
        }

        // Firstly warp stop
        if (stop < 0) {
            stop += shape_[dim];
        }
        // If stop is behing start, empty image
        if (stop <= start) {
            stop = start;
        }
        else if (stop >= shape_[dim]) {
            stop = shape_[dim];
        }

        // Get data pointer to the possition selected
        // treat the pointer as pointing to a sequence of bytes
        // (char is 1 byte in C++). This way, you can perform arithmetic
        // operations on the pointer.
        void* new_data_ptr = static_cast<char*>(data_ptr_) + (start * strides_[dim] * dtype_.getSize());

        ShapeArray new_shape = shape_;
        ShapeArray new_stride = strides_;

        // Update shape and stride
        new_shape[dim] = (stop - start + step - 1) / step;
        new_stride[dim] = strides_[dim] * step;

        return Image(new_shape, new_stride, new_data_ptr, dtype_, blob_);
    }

    Image Image::operator[](int64_t i) const { return IndexExtract(0, i); }

    Image Image::IndexExtract(int64_t dim, int64_t idx) const {

        if (shape_.size() == 0) {
            LogError("Cannot index a 0 sized Image");
        }

        dim = WrapDim(dim, NumDims());
        idx = WrapDim(idx, shape_[dim]);

        // Get the original shape and stride
        ShapeArray new_shape(shape_);
        ShapeArray new_stride(strides_);

        // Delete data until dim
        new_shape.erase(new_shape.begin() + dim);
        new_stride.erase(new_stride.begin() + dim);

        void* new_data_ptr = static_cast<char*>(data_ptr_) + ( strides_[dim] * dtype_.ByteSize() * idx);

        return {new_shape, new_stride, new_data_ptr, dtype_, blob_};
    }

    Image Image::Contiguous() const {
        if (IsContiguous()) {
            return *this;
        }
        else {
            return To(GetDevice(), true);
        }
    }

    Image Image::GetItem(const TensorKey &tk) const {
        if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
            return IndexExtract(0, tk.GetIndex());
        } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
            if (NumDims() == 0) {
                LogError("Cannot slice a scalar (0-dim) image.");
            }
            TensorKey tk_new = tk.InstantiateDimSize(shape_[0]);
            return Slice(0, tk_new.GetStart(), tk_new.GetStop(), tk_new.GetStep());
        } else if (tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor) {
            LogError("TensorKey::TensorKeyMode::IndexTensor not implemented yet");
        } else {
            LogError("Internal error: wrong TensorKeyMode.");
        }
    }

    Image Image::GetItem(const std::vector<TensorKey>& tks) const {
        if (std::any_of(tks.begin(), tks.end(), [](const TensorKey& tk) {
                return tk.GetMode() == TensorKey::TensorKeyMode::IndexTensor;
            })) {
            LogError("IndexTensor is not implemented in GetItem");
        }

        Image t = *this;
        int64_t slice_dim = 0;
        for (const TensorKey& tk : tks) {
            if (tk.GetMode() == TensorKey::TensorKeyMode::Index) {
                t = t.IndexExtract(slice_dim, tk.GetIndex());
            } else if (tk.GetMode() == TensorKey::TensorKeyMode::Slice) {
                TensorKey tk_new = tk.InstantiateDimSize(t.shape_[slice_dim]);
                t = t.Slice(slice_dim, tk_new.GetStart(), tk_new.GetStop(),
                            tk_new.GetStep());
                slice_dim++;
            } else {
                LogError("Internal error: wrong TensorKeyMode.");
            }
        }
        return t;
    }

    Image Image::To(const Device &device, bool copy) const {
        // If we don't want to copy Image and resides in the same device
        // we return the same Image
        if (!copy && GetDevice() == device) {
            return *this;
        }

        Image dst_image(shape_, dtype_, device);
        kernel::Copy(*this, dst_image);
        return dst_image;
    }

    std::string Image::ToString(bool with_suffix,
                                const std::string& indent) const {
        std::ostringstream rc;
        if (IsCUDA() || !IsContiguous()) {
            Image host_contiguous_tensor = Contiguous().To(Device("CPU:0"));
            rc << host_contiguous_tensor.ToString(false, indent);
        } else {
            if (shape_.NumElements() == 0) {
                rc << indent;
                rc << "0-element Tensor";
            } else if (shape_.size() == 0) {
                rc << indent;
                rc << ScalarPtrToString(data_ptr_);
            } else if (shape_.size() == 1) {
                const char* ptr = static_cast<const char*>(data_ptr_);
                rc << "[";
                std::string delim = "";
                int64_t element_byte_size = dtype_.ByteSize();
                for (int64_t i = 0; i < shape_.NumElements(); ++i) {
                    rc << delim << ScalarPtrToString(ptr);
                    delim = " ";
                    ptr += element_byte_size;
                }
                rc << "]";
            } else {
                rc << "[";
                std::string delim = "";
                std::string child_indent = "";
                for (int64_t i = 0; i < shape_[0]; ++i) {
                    rc << delim << child_indent
                       << this->operator[](i).ToString(false, indent + " ");
                    delim = ",\n";
                    child_indent = indent + " ";
                }
                rc << "]";
            }
        }
        if (with_suffix) {
            std::ostringstream ra;
            ra << "\nTensor[shape=" << shape_.ToString() << ", stride="
            << strides_.ToString() << ", " << dtype_.ToString() << ", "
            << GetDevice().ToString() << "]";
            rc << ra.str();
        }
        return rc.str();
    }

    std::string Image::ScalarPtrToString(const void* ptr) const {
        std::ostringstream out;
        if (dtype_ == core::Bool) {
            out << *static_cast<const unsigned char*>(ptr) ? "True" : "False";
        }
        else {
            DISPATCH_DTYPE_TO_TEMPLATE(dtype_, [&]() {
                out << " " << *static_cast<const scalar_t*>(ptr) << " ";
            });
        }
        return out.str();
    }

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
