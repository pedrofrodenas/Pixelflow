#include "pixelflow/core/kernel/Copy.h"
#include "pixelflow/core/Image.h"
#include "pixelflow/core/Broadcasting.h"
#include "pixelflow/core/Indexer.h"
#include "pixelflow/core/Dispatch.h"

namespace pixelflow {
namespace core {

namespace kernel {

void Copy(const Image& src, Image& dst) {

    // Check if shape match or are broadcastable
    if (!CanBeBroadcastedTo(src.Shape(), dst.Shape())) {
        std::ostringstream oss;
        oss << "Shape: " << src.Shape().ToString()
        << " cannot be broadcasted to: "
        << dst.Shape().ToString();
        LogError(oss.str().c_str());
    }

    // Copy in CUDA is not implemented
    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();
    if (src_device.IsCPU() && dst_device.IsCPU()) {
        CopyCPU(src, dst);
    }
    else {
        LogError("Copy: Unimplemented device");
    }
}

void CopyCPU(const Image& src, Image &dst) {

    // src and dst have been checked to have the same shape, dtype, device
    PfType src_dtype = src.GetDtype();
    PfType dst_dtype = dst.GetDtype();

    if (src.Shape() == dst.Shape() && 
        src_dtype == dst_dtype && 
        src.IsContiguous() && dst.IsContiguous()) {
            MemoryManager::Memcpy(dst.GetDataPtr(), dst.GetDevice(),
                                  src.GetDataPtr(), src.GetDevice(), 
                                  src.NumElements()*src_dtype.ByteSize());
        }
    else {
        
        // Create an Indexer object
        Indexer indexer({src}, dst, DtypePolicy::NONE);
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                using src_t = scalar_t;
        });
    }
}



}  // namespace kernel
}  // namespace core
}  // namespace pixelflow