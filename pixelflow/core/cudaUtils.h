#pragma once
#include "pixelflow/pfconfig.h"
#ifdef CUDA_ENABLED

#include "cuda_runtime.h"

#define PIXELFLOW_CUDA_CHECK(err) \
    pixelflow::core::__PIXELFLOW_CUDA_CHECK(err, __FILE__, __LINE__)


namespace pixelflow {
namespace core {


void __PIXELFLOW_CUDA_CHECK(cudaError_t err, const char* file, const int line);

} // namespace core
}  // namespace pixelflow

#else

namespace pixelflow {
namespace core {

#define PIXELFLOW_CUDA_CHECK(err)

} // namespace core
} // namespace pixelflow
#endif


namespace pixelflow {
namespace core {
/// Returns the number of available CUDA devices. Returns 0 if Pixelflow is not
/// compiled with CUDA support.
int DeviceCount();
}
}





