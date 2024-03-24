#include "pixelflow/core/Broadcasting.h"

namespace pixelflow {
namespace core {

    ShapeArray ExpandDims(const ShapeArray& shape, int64_t ndims) {

        int64_t currentDims = shape.GetDims();

        if ( currentDims > ndims)
        {
            std::ostringstream oss;
            oss << "Cannot expand " << currentDims << " to " << ndims;
            LogError(oss.str().c_str());
        }
        ShapeArray expandedArray(ndims, 1);

        std::copy(shape.rbegin(), shape.rend(), expandedArray.rbegin());
        return expandedArray;
    }

    bool IsBroadcastable(const ShapeArray& l, const ShapeArray& r) {

        int64_t ldims = l.GetDims();
        int64_t rdims = r.GetDims();

        if ((ldims == 0) || (rdims == 0)) {
            return true;
        }

        // Get the ShapeArray that has less dimensions
        int64_t minDim = std::min(ldims, rdims);

        for (int i = 0 ; i != minDim; ++i) 
        {   
            int64_t cLDim = l[ldims - i - 1];
            int64_t cRDim = r[rdims - i - 1];
            if (!((cLDim == cRDim) || (cLDim == 1) || (cRDim == 1)))
            {
                return false;
            }
        }
        return true;
    }

    ShapeArray BroadcastShape(const ShapeArray &l, const ShapeArray& r){

        if (!IsBroadcastable(l, r))
        {
            std::ostringstream oss;
            oss << "Shape: " << l.ToString() << " is not broadcastable "
            "with Shape: " << r.ToString();
            LogError(oss.str().c_str());
        }

        int64_t lDims = l.GetDims();
        int64_t rDims = r.GetDims();

        int64_t maxShape = std::max(lDims, rDims);

        ShapeArray lExpanded = ExpandDims(l, maxShape);
        ShapeArray rExpanded = ExpandDims(r, maxShape);

        ShapeArray outputArray(maxShape);
        for (size_t i = 0; i != maxShape; ++i)
        {
            if (lExpanded[i] == 1)
            {
                outputArray[i] = rExpanded[i];
            }
            else if (rExpanded[i] == 1)
            {
                outputArray[i] = lExpanded[i];
            }
            else if (lExpanded[i] == rExpanded[i])
            {
                outputArray[i] = lExpanded[i];
            }
            else {
                LogError("Some bug happend in BroadcastShape in BroadCasting.cpp");
            }
        }
        return outputArray;
    }

    bool CanBeBroadcastedTo(const ShapeArray &l, const ShapeArray &r) {
        if (!IsBroadcastable(l, r)) {
            return false;
        }
        else {
            return (BroadcastShape(l, r) == r);
        }
    }


    ShapeArray DefaultStrides(const ShapeArray& shape) {
        ShapeArray strides(static_cast<int64_t>(shape.size()));
        int64_t stride_size = 1;
        for (auto i = static_cast<int64_t>(shape.size()); i > 0; --i) {
            strides[i - 1] = stride_size;
            // Handles 0-sized dimensions
            stride_size *= std::max<int64_t>(shape[i - 1], 1);
        }
        return strides;
    }

    ShapeArray ReductionShape(const ShapeArray &src_shape,
                              const ShapeArray &dims,
                              bool keepdim) {

        int64_t src_dims = src_shape.GetDims();
        ShapeArray out_shape(src_shape);

        if (keepdim) {
            for (const int64_t& dim : dims) {
                out_shape[WrapDim(dim, src_dims)] = 1;
            }
        }
        else {
            // If dim i is reduced, dims_mask[i] == true.
            std::vector<bool> dims_mask(src_dims, false);
            for (const int64_t dim : dims) {
                if (dims_mask[WrapDim(dim, src_dims)]) {
                    LogError("Repeated reduction dimension");
                }
                dims_mask[WrapDim(dim, src_dims)] = true;
            }
            // Copy to out_shape only possitions not specified in dims
            int64_t to_fill = 0;
            for (int64_t i = 0; i != src_dims; ++i) {
                if (!dims_mask[i]) {
                    out_shape[to_fill] = out_shape[i];
                    to_fill++;
                }
            }
            out_shape.resize(to_fill);
        }
        return out_shape;
    }


    int64_t WrapDim(int64_t dim, const int64_t max_dim) {

        if (max_dim <= 0) {
            LogError("max_dim should be greater than zero");
        }

        int64_t min = -max_dim;

        if (dim < min || dim > max_dim) {
            std::ostringstream rc;
            rc << "Index out-of-range: dim == " << dim
            << ", but it must satisfy " << min << " <= dim <="
            << max_dim;
            LogError(rc.str().c_str());
        }

        if (dim < 0) {
            dim += max_dim;
        }
        return dim;
    }




} // namespace core
} // namespace pixelflow