#pragma once

#include <algorithm>

#include "pixelflow/core/ShapeArray.h"

namespace pixelflow {
namespace core {

bool IsBroadcastable(const ShapeArray& l, const ShapeArray& r);

} // namespace core
} // namespace pixelflow