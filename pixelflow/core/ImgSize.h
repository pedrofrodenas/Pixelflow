#pragma once

#include "pixelflow/core/SmallVector.h"

namespace pixelflow {
namespace core {

class ImgSize;

class ImgSize : public SmallVector<int64_t, 4> {
public:
    using parent_v = SmallVector<int64_t, 4>;

    ImgSize() {}

    ImgSize(const std::initializer_list<int64_t>& dim_sizes);

    ImgSize(const std::vector<int64_t>& dim_sizes);

    ImgSize(const ImgSize& other);
};

} 
}