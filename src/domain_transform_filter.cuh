#ifndef DOMAIN_TRANSFORM_FILTER_2D_CUH_
#define DOMAIN_TRANSFORM_FILTER_2D_CUH_

namespace domain_transform {

// Linear serach along rows, can be made better using binary search.
template <typename CudaArrayType>
__device__ __forceinline__ void LinearSearchLowerAndUpperBound2DX(
    const ImageDim image_dim, const CudaArrayType* ct_dir, const float sigma,
    const int pos_x, const int pos_y, int* lower_bound, int* upper_bound) {
  const float val = ct_dir->get(pos_x, pos_y);
  *lower_bound = pos_x - 1;
  *upper_bound = *lower_bound + 1;

  for (int i = pos_x; i >= 0; i--) {
    if (val - sigma > ct_dir->get(i, pos_y)) {
      *lower_bound = i;
      break;
    }
  }

  for (int i = pos_x; i < image_dim.width; i++) {
    if (val + sigma < ct_dir->get(i, pos_y)) {
      *upper_bound = i - 1;
      break;
    }
  }
}

template <typename CudaArrayType>
__global__ void FilterX(const ImageDim image_dim, CudaArrayType ct_dir,
                        const float sigma_x, CudaArrayType intergrated_var,
                        CudaArrayType var) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    int lower_bound;
    int upper_bound;

    LinearSearchLowerAndUpperBound2DX(image_dim, &ct_dir, sigma_x, x, y,
                                      &lower_bound, &upper_bound);
    const float var_val = (intergrated_var.get(upper_bound, y) -
                           intergrated_var.get(lower_bound, y)) /
                          (upper_bound - lower_bound);

    var.set(x, y, var_val);
  }
}

template <typename CudaArrayType>
__global__ void FilterY(const ImageDim image_dim, CudaArrayType ct_dir,
                        const float sigma_y, CudaArrayType intergrated_var,
                        CudaArrayType var) {
  // Note that the x and y direction is interchanged.
  const int y = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    int lower_bound;
    int upper_bound;

    LinearSearchLowerAndUpperBound2DX(image_dim, &ct_dir, sigma_y, y, x,
                                      &lower_bound, &upper_bound);

    const float var_val = (intergrated_var.get(upper_bound, x) -
                           intergrated_var.get(lower_bound, x)) /
                          (upper_bound - lower_bound);
    var.set(x, y, var_val);
  }
}

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_FILTER_2D_CUH_
