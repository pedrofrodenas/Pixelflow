#include "pixelflow/core/ImgSize.h"

namespace pixelflow {
namespace core {

    ImgSize::ImgSize(const std::initializer_list<int64_t>& dim_sizes)
        : parent_v(dim_sizes) {}
}
}