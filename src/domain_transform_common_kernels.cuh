#ifndef DOMAIN_TRANSFORM_COMMON_KERNELS_CUH_
#define DOMAIN_TRANSFORM_COMMON_KERNELS_CUH_

#include "domain_transform_common.h"

namespace domain_transform {

template <typename CudaType1, typename CudaType2>
__global__ void CopyVariable(const ImageDim image_dim, const CudaType1 src,
                             CudaType2 dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    dst.set(x, y, src.get(x, y));
  }
}

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_COMMON_KERNELS_CUH_
