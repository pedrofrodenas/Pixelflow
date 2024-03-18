#include "pixelflow/core/ShapeArray.h"

namespace pixelflow {
namespace core {

    ShapeArray::ShapeArray(const std::initializer_list<int64_t>& dim_sizes)
        : parent_v(dim_sizes) {}

    ShapeArray::ShapeArray(const std::vector<int64_t>& dim_sizes)
        : parent_v(dim_sizes.cbegin(), dim_sizes.cend()) {}

    ShapeArray::ShapeArray(const ShapeArray& other)
        : parent_v(other) {}

    ShapeArray::ShapeArray(int64_t numElements)
        : parent_v(numElements) {}

    ShapeArray::ShapeArray(int64_t nElements, int64_t value)
        : parent_v(nElements, value) {}

    ShapeArray& ShapeArray::operator=(const ShapeArray& v) {
        static_cast<parent_v*>(this)->operator=(v);
        return *this;
    }

    ShapeArray& ShapeArray::operator=(ShapeArray&& v) {
        static_cast<parent_v*>(this)->operator=(v);
        return *this;
    }

    int64_t ShapeArray::NumElements() const {
    if (this->size() == 0) {
        return 1;
    }
    auto f = [this](const int64_t& a, const int64_t& b) -> int64_t{return std::multiplies<int64_t>()(a, b);};

    return std::accumulate(
            this->begin(), this->end(), 1LL, f);
    }

    int64_t ShapeArray::GetDims() const {
        return this->size();
    }

    std::string ShapeArray::ToString() const {
        std::ostringstream oss;
        oss << "[" << Join(this->begin(), this->end(), ", ") << "]";
        return oss.str();
    }
} // namespace core
} // namespace pixelflow