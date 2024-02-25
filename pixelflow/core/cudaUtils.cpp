#ifdef CUDA_ENABLED

#include "cudaUtils.h"


#include <iostream>

#include "../utility/Logging.h"

namespace pf {
namespace cuda {

int DeviceIds() {
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
}

void __PIXELFLOW_CUDA_CHECK(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        LogError(file, line, cudaGetErrorString(err));
    }
}
} // namespace cuda
} // namespace pixelflow

#endif







