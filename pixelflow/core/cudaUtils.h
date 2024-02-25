#ifdef CUDA_ENABLED
#pragma once

#include "cuda_runtime.h"


#define PIXELFLOW_CUDA_CHECK(err) \
    pf::cuda::__PIXELFLOW_CUDA_CHECK(err, __FILE__, __LINE__)


namespace pf {
namespace cuda {

/// Returns the number of available CUDA devices. Returns 0 if Pixelflow is not
/// compiled with CUDA support.
int DeviceIds();

void __PIXELFLOW_CUDA_CHECK(cudaError_t err, const char* file, const int line);

} // namespace cuda
}  // namespace pf



#endif