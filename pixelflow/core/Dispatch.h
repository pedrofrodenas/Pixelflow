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

#include "pixelflow/utility/Logging.h"
#include "pixelflow/core/dtypes.h"

/// Call a numerical templated function based on Dtype. Wrap the function to
/// a lambda function to use DISPATCH_DTYPE_TO_TEMPLATE.
///
/// Before:
///     if (dtype == core::Float32) {
///         func<float>(args);
///     } else if (dtype == core::Float64) {
///         func<double>(args);
///     } else ...
///
/// Now:
///     DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
///        func<scalar_t>(args);
///     });
///
/// Inspired by:
///     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#define DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, ...)                   \
    [&] {                                                        \
        if (DTYPE == pixelflow::core::Float32) {                    \
            using scalar_t = float;                              \
            return __VA_ARGS__();                                \
        } else if (DTYPE == pixelflow::core::Float64) {             \
            using scalar_t = double;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == pixelflow::core::Int8) {                \
            using scalar_t = int8_t;                             \
            return __VA_ARGS__();                                \
        } else if (DTYPE == pixelflow::core::Int32) {               \
            using scalar_t = int32_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == pixelflow::core::Int64) {               \
            using scalar_t = int64_t;                            \
            return __VA_ARGS__();                                \
        } else if (DTYPE == pixelflow::core::UInt8) {               \
            using scalar_t = uint8_t;                            \
            return __VA_ARGS__();                                \
        } else {                                                 \
            LogError("Unsupported data type."); \
        }                                                        \
    }()

#define DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(DTYPE, ...)    \
    [&] {                                                   \
        if (DTYPE == pixelflow::core::Bool) {                  \
            using scalar_t = bool;                          \
            return __VA_ARGS__();                           \
        } else {                                            \
            DISPATCH_DTYPE_TO_TEMPLATE(DTYPE, __VA_ARGS__); \
        }                                                   \
    }()