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
#pragma once
#include <cstdint>

#include "pixelflow/core/Device.h"
#include "pixelflow/utility/Logging.h"

namespace pixelflow {
namespace core {



    /// Run a function in parallel on CPU or CUDA.
///
/// \param device The device for the parallel for loop to run on.
/// \param n The number of workloads.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
///
/// \note This is optimized for uniform work items, i.e. where each call to \p
/// func takes the same time.
/// \note If you use a lambda function, capture only the required variables
/// instead of all to prevent accidental race conditions. If you want the
/// kernel to be used on both CPU and CUDA, capture the variables by value.
template <typename func_t>
void ParallelFor(const Device& device, int64_t n, const func_t& func) {
#ifdef __CUDACC__
    ParallelForCUDA_(device, n, func);
#else
    ParallelForCPU_(device, n, func);
#endif
}

/// Run a function in parallel on CPU.
template <typename func_t>
void ParallelForCPU_(const Device& device, int64_t n, const func_t& func) {
    if (!device.IsCPU()) {
        LogError("ParallelFor for CPU cannot run on device");
    }
    if (n == 0) {
        return;
    }

#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < n; ++i) {
        func(i);
    }
}

}
} // 