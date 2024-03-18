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

    /// \brief Wrap around negative \p dim.
    ///
    /// E.g. If max_dim == 5, dim -1 will be converted to 4.
    ///
    /// \param dim Dimension index
    /// \param max_dim Maximum dimension index
    int64_t WrapDim(int64_t dim, int64_t max_dim);

    /// \brief Returns the shape after reduction.
    ///
    /// \param src_shape shape to reduce
    /// \param dims A list of dimensions to be reduced.
    /// \param keepdim If true, the reduced dims will be retained as size 1.
    ShapeArray ReductionShape(const ShapeArray& src_shape,
                              const ShapeArray& dims,
                              bool keepdim);

} // namespace core
} // namespace pixelflow