#pragma once

#include "pixelflow/core/MemoryManager.h"

namespace pixelflow {
namespace core {

class Blob {

public:
    /// Construct Blob on a specified device.
    ///
    /// \param byte_size Size of the blob in bytes.
    /// \param device Device where the blob resides.
    Blob(int64_t byte_size, const Device &device)
        : deleter_(nullptr),
          data_ptr_(MemoryManager::Malloc(byte_size, device)),
          device_(device) {};

    /// Construct Blob with externally managed memory.
    ///
    /// \param device Device where the blob resides.
    /// \param data_ptr Pointer the blob's beginning.
    /// \param deleter The deleter function is called at Blob's destruction to
    /// notify the external memory manager that the memory is no longer needed.
    /// It's up to the external manager to free the memory.
    Blob(const Device& device,
         void* data_ptr,
         const std::function<void(void*)>& deleter)
        : deleter_(deleter), data_ptr_(data_ptr), device_(device) {}

    ~Blob() {
        if (deleter_) {
            deleter_(nullptr);
        }
        else {
            MemoryManager::Free(data_ptr_, device_);
        }
    }

    Device GetDevice() const {return device_;}

    void* GetDataPtr() {return data_ptr_;}
    const void* GetDataPtr() const { return data_ptr_; }


protected:
    /// For externally managed memory, deleter != nullptr.
    std::function<void(void*)> deleter_ = nullptr;
    void* data_ptr_ = nullptr;
    Device device_;
};


} // namespace core
} // namespace pixelflow


