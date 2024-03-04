#include "pixelflow/core/ImgSize.h"

namespace pixelflow {
namespace core {

    ImgSize::ImgSize(const std::initializer_list<int64_t>& dim_sizes)
        : parent_v(dim_sizes) {}

    ImgSize::ImgSize(const std::vector<int64_t>& dim_sizes)
        : parent_v(dim_sizes.cbegin(), dim_sizes.cend()) {}

    ImgSize::ImgSize(const ImgSize& other)
        : parent_v(other) {}

    ImgSize& ImgSize::operator=(const ImgSize& v) {
        static_cast<parent_v*>(this)->operator=(v);
        return *this;
    }

    ImgSize& ImgSize::operator=(ImgSize&& v) {
        static_cast<parent_v*>(this)->operator=(v);
        return *this;
    }

    int64_t ImgSize::NumElems() const {
    if (this->size() == 0) {
        return 1;
    }
    auto f = [this](const int64_t& a, const int64_t& b) -> int64_t{return std::multiplies<int64_t>()(a, b);};

    return std::accumulate(
            this->begin(), this->end(), 1LL, f);
    }

    

    
}
}