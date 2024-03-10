#pragma once

#include <algorithm>

#include "pixelflow/core/ShapeArray.h"

namespace pixelflow {
namespace core {

    ShapeArray ExpandDims(const ShapeArray& shape, int64_t ndims);

    bool IsBroadcastable(const ShapeArray& l, const ShapeArray& r);

    ShapeArray BroadcastShape(const ShapeArray& l, const ShapeArray& r);

    /// Computes default stride for row-mayor continuous memory
    ShapeArray DefaultStrides(const ShapeArray& shape);

} // namespace core
} // namespace pixelflow