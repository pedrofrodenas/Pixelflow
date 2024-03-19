#pragma once

#include "pixelflow/core/Image.h"
#include "pixelflow/core/Broadcasting.h"

namespace pixelflow {
namespace core {
namespace kernel {

void Copy(const Image& src, Image& dst);


}  // namespace kernel
}  // namespace core
}  // namespace pixelflow