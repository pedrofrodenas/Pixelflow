#pragma once

class Image;

namespace pixelflow {
namespace core {

namespace kernel {

void Copy(const Image& src, Image& dst);

void CopyCPU(const Image& src, Image& dst);


}  // namespace kernel
}  // namespace core
}  // namespace pixelflow