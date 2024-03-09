
#include "cudaUtils.h"
#include <iostream>
#include "pixelflow/core/cudaUtils.h"
#include "pixelflow/utility/Logging.h"

namespace pixelflow {
namespace core {
namespace cuda {

int DeviceCount() {
#ifdef CUDA_ENABLED
    try { 
        int num_devices;
        PIXELFLOW_CUDA_CHECK(cudaGetDeviceCount(&num_devices));
        return num_devices;
    }
    // This function is also used to detect CUDA support in our Python code.
    // Thus, catch any errors if no GPU is available.
    catch (const std::runtime_error&) {
        return 0;
    }
#else
    return 0;
#endif
}

#ifdef CUDA_ENABLED
void __PIXELFLOW_CUDA_CHECK(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        LogError_(file, line, cudaGetErrorString(err));
    }
}
#endif

} // namespace cuda
} // namespace core
} // namespace pixelflow








