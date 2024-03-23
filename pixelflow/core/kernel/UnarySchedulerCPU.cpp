
#include "pixelflow/core/Indexer.h"
#include "pixelflow/core/Device.h"
#include "pixelflow/core/ParallelFor.h"

namespace pixelflow {
namespace core {
namespace kernel {

template <typename element_func_t>
static void LaunchUnaryEWKernel(const Indexer& indexer,
                                const element_func_t& element_func) {
    ParallelFor(Device("CPU:0"), indexer.NumWorkloads(),
                [&indexer, &element_func](int64_t i) {
                    element_func(indexer.GetInputPtr(0, i),
                                 indexer.GetOutputPtr(i));
                });
}

} // kernel
} // core
} // pixelflow