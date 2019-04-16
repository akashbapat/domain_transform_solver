#ifndef DOMAIN_TRANSFORM_COMMON_KERNELS_CUH_
#define DOMAIN_TRANSFORM_COMMON_KERNELS_CUH_

#include "domain_transform_common.h"

namespace domain_transform {
namespace {

constexpr float kINV255 = 0.00392156863;
constexpr float kINV255Sq = 0.0000153787;

}  // namespace

template <typename SRC_TYPE, typename DST_TYPE>
__global__ void CopyVariable(const ImageDim image_dim, SRC_TYPE src,
                             DST_TYPE dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    dst.set(x, y, src.get(x, y));
  }
}

template <typename SRC_TYPE, typename DST_TYPE>
__global__ void CopyVariableTranspose(const ImageDim image_dim, SRC_TYPE src,
                                      DST_TYPE dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    dst.set(y, x, src.get(x, y));
  }
}

__device__ __forceinline__ float Uchar4SumOfAbsDiffNormalize(uchar4 v1,
                                                             uchar4 v2) {
  return (fabsf(v1.x - v2.x) + fabsf(v1.y - v2.y) + fabsf(v1.z - v2.z) +
          fabsf(v1.w - v2.w)) *
         kINV255;
}

__device__ __forceinline__ float Uchar4SumOfSquaredDiffNormalize(uchar4 v1,
                                                                 uchar4 v2) {
  return ((v1.x - v2.x) * (v1.x - v2.x) * 1.0f +
          (v1.y - v2.y) * (v1.y - v2.y) * 1.0f +
          (v1.z - v2.z) * (v1.z - v2.z) * 1.0f +
          (v1.w - v2.w) * (v1.w - v2.w) * 1.0f) *
         kINV255Sq;
}

template <typename SRC_TYPE, typename DST_TYPE>
__global__ void Compute_dHdx(const ImageDim image_dim,
                             const float sigma_x_div_sigma_r,
                             const SRC_TYPE color_image, DST_TYPE dHdx) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    // Compute the backward difference.
    const typename SRC_TYPE::Scalar curr = color_image.get(x, y);
    const typename SRC_TYPE::Scalar backward = color_image.get(x - 1, y);
    const typename DST_TYPE::Scalar grad =
        sqrt(Uchar4SumOfSquaredDiffNormalize(curr, backward) *
                 sigma_x_div_sigma_r * sigma_x_div_sigma_r +
             1);
    dHdx.set(x, y, grad);
  }
}

template <typename SRC_TYPE, typename DST_TYPE>
__global__ void Compute_dVdy(const ImageDim image_dim,
                             const float sigma_y_div_sigma_r,
                             const SRC_TYPE color_image, DST_TYPE dVdy) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    // Compute the backward difference.
    const typename SRC_TYPE::Scalar curr = color_image.get(x, y);
    const typename SRC_TYPE::Scalar backward = color_image.get(x, y - 1);
    const typename DST_TYPE::Scalar grad =
        sqrt(Uchar4SumOfSquaredDiffNormalize(curr, backward) *
                 sigma_y_div_sigma_r * sigma_y_div_sigma_r +
             1);
    dVdy.set(x, y, grad);
  }
}

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_COMMON_KERNELS_CUH_
