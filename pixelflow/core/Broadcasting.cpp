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
            oss << "Shape: " << l.Shape() << " is not broadcastable "
            "with Shape: " << r.Shape();
            LogError(oss.str().c_str());
        }

        int64_t lDims = l.GetDims();
        int64_t rDims = r.GetDims();


        int64_t maxShape = std::max(lDims, rDims);

        ShapeArray largestArray;
        
        return l;
    }



} // namespace core
} // namespace pixelflow