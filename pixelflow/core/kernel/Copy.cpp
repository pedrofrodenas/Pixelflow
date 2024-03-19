#include "pixelflow/core/kernel/Copy.h"


namespace pixelflow {
namespace core {
namespace kernel {

void Copy(const Image &src, Image &dst) {

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
        // CopyCPU
    }
    else {
        LogError("Copy: Unimplemented device");
    }




}


}  // namespace kernel
}  // namespace core
}  // namespace pixelflow