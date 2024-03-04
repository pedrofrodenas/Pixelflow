#pragma once

#include <numeric>
#include "pixelflow/core/SmallVector.h"
#include "pixelflow/utility/Logging.h"


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

    template <class InputIterator>
    ImgSize(InputIterator first, InputIterator last)
        : parent_v(first, last) {}

    // Assigment operator, returns a reference to ImgSize
    // object and it takes a constant reference to ImgSize
    // We are going to call superclass's assigment
    // operator
    ImgSize& operator=(const ImgSize& v);

    // Move operator (without const)
    ImgSize& operator=(ImgSize&& v);

    int64_t NumElems() const;

    int64_t GetDims() const;

    std::string Shape();
};

} 
}