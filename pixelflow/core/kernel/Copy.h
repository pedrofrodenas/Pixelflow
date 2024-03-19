#pragma once



namespace pixelflow {
namespace core {

class Image;

namespace kernel {

void Copy(const Image& src, Image& dst);

void CopyCPU(const Image& src, Image& dst);


}  // namespace kernel
}  // namespace core
}  // namespace pixelflow