#include "pixelflow/core/Broadcasting.h"

namespace pixelflow {
namespace core {

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



} // namespace core
} // namespace pixelflow