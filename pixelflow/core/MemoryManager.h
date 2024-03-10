#pragma once

#include "pixelflow/core/Device.h"

namespace pixelflow {
namespace core {

// Adapted for Pixelflow from Open3D project.
// Commit e39c1e0994ac2adabd8a617635db3e35f04cce88 2023-03-10
// Documentation:
// https://www.open3d.org/docs/0.15.1/cpp_api/classopen3d_1_1core_1_1_memory_manager.html
//
//===- Open3d/cpp/open3d/core/MemoryManager.h -===//
// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
class MemoryManagerDevice {
public:
    virtual ~MemoryManagerDevice() = default;

    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    virtual void* Malloc(size_t byte_size, const Device& device) = 0;

    /// Frees previously allocated memory at address \p ptr on device \p device.
    virtual void Free(void* ptr, const Device& device) = 0;

    /// Copies \p num_bytes bytes of memory at address \p src_ptr on device
    /// \p src_device to address \p dst_ptr on device \p dst_device.
    virtual void Memcpy(void* dst_ptr,
                        const Device& dst_device,
                        const void* src_ptr,
                        const Device& src_device,
                        size_t num_bytes) = 0;
};

class MemoryManagerCPU : public MemoryManagerDevice {
public:
    void* Malloc(size_t byte_size, const Device& device) override;

    void Free(void* ptr, const Device& device) override;

    void Memcpy(void* dst_ptr,
                const Device& dst_device,
                void* src_ptr,
                const Device& src_device,
                size_t num_bytes);
};


} // namespace core
} // namespace pixelflow

