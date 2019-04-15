#ifndef COLOR_SPACE_CUH_
#define COLOR_SPACE_CUH_

#include "cudaSurface2D.h"

namespace domain_transform {

__global__ void RGB2YCbCr(cua::CudaSurface2D<uchar4> image) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image.width_ && y < image.height_) {
    // These magic values are taken from https://en.wikipedia.org/wiki/YCbCr
    const uchar4 val = image.get(x, y);
    const uchar4 y_cb_cr_val = make_uchar4(
        val.x * 0.299 + 0.587 * val.y + 0.114 * val.z,
        128 - val.x * 0.168736 - 0.331264 * val.y + 0.5 * val.z,
        128 + val.x * 0.5 - 0.418688 * val.y - 0.081312 * val.z, val.w);
    image.set(x, y, y_cb_cr_val);
  }
}

__global__ void RGB2YYY(cua::CudaSurface2D<uchar4> image) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image.width_ && y < image.height_) {
    // These magic values are taken from https://en.wikipedia.org/wiki/YCbCr
    const uchar4 val = image.get(x, y);
    const unsigned char y_val = val.x * 0.299 + 0.587 * val.y + 0.114 * val.z;
    image.set(x, y, make_uchar4(y_val, y_val, y_val, val.w));
  }
}

}  // namespace domain_transform

#endif  // COLOR_SPACE_CUH_
