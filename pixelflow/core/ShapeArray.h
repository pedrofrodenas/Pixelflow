#pragma once

#include <numeric>
#include "pixelflow/core/SmallVector.h"
#include "pixelflow/utility/Logging.h"


namespace pixelflow {
namespace core {

class ShapeArray;

class ShapeArray : public SmallVector<int64_t, 4> {
public:
    using parent_v = SmallVector<int64_t, 4>;

    ShapeArray() {}

    ShapeArray(const std::initializer_list<int64_t>& dim_sizes);

    ShapeArray(const std::vector<int64_t>& dim_sizes);

    ShapeArray(const ShapeArray& other);

    ShapeArray(int64_t numElements);

    ShapeArray(int64_t nElements, int64_t value);

    template <class InputIterator>
    ShapeArray(InputIterator first, InputIterator last)
        : parent_v(first, last) {}

    // Assigment operator, returns a reference to ShapeArray
    // object and it takes a constant reference to ShapeArray
    // We are going to call superclass's assigment
    // operator
    ShapeArray& operator=(const ShapeArray& v);

    // Move operator (without const)
    ShapeArray& operator=(ShapeArray&& v);

    int64_t NumElements() const;

    int64_t GetDims() const;

    std::string Shape() const;
};

} // namespace core
} // namespace pixelflow