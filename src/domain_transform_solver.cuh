#ifndef DOMAIN_TRANSFORM_SOLVER_CUH_
#define DOMAIN_TRANSFORM_SOLVER_CUH_


#include "cudaSurface2D.h"

#include "domain_transform_filter.cuh"
#include "domain_transform_filter_params.h"
#include "domain_transform_optimize.h"

namespace domain_transform {

__device__ inline float GaussianLossGradient(const float confidence,
                                             const float var,
                                             const float target) {
  return 2 * confidence * (var - target);
}

__device__ inline float CharbonnierLossGradient(const float confidence,
                                                const float var,
                                                const float target) {
  const float x = (var - target);
  return confidence * x / sqrt(x * x + 0.000001);
}

__device__ inline float Step(const float lambda, const float bilateral_grad,
                             const float grad_target) {
  return -(lambda * bilateral_grad + 0.5 * grad_target);
}

template <typename... Args>
struct EmptyLoss {
  __device__ float operator()(const int x, const int y, Args... args) {
    return 0;
  }
};

template <RobustLoss loss_type, bool update_monitor_flag,
          typename CudaArrayType, typename LossTerm = EmptyLoss<float> >
__global__ void OptimizeX(const ImageDim image_dim, CudaArrayType const_var,
                          CudaArrayType target, CudaArrayType confidence,
                          CudaArrayType ct_dir, const float sigma_z,
                          const CudaArrayType intergrated_const_var,
                          const float lambda, CudaArrayType var,
                          LossTerm loss_term = LossTerm()) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    int lower_bound;
    int upper_bound;

    LinearSearchLowerAndUpperBound2DX(image_dim, &ct_dir, sigma_z, x, y,
                                      &lower_bound, &upper_bound);

    const float bilateral_grad = (intergrated_const_var.get(upper_bound, y) -
                                  intergrated_const_var.get(lower_bound, y)) /
                                 (upper_bound - lower_bound);

    float grad_target;
    switch (loss_type) {
      case RobustLoss::L2: {
        grad_target = GaussianLossGradient(
            confidence.get(x, y) / (upper_bound - lower_bound),
            const_var.get(x, y), target.get(x, y));
        break;
      }
      case RobustLoss::CHARBONNIER: {
        grad_target = CharbonnierLossGradient(
            confidence.get(x, y) / (upper_bound - lower_bound),
            const_var.get(x, y), target.get(x, y));

        break;
      }
    }

    grad_target += loss_term.operator()(x, y, const_var.get(x, y));

    const float var_val =
        const_var.get(x, y) +
        Step(lambda,
             GaussianLossGradient(0.5, const_var.get(x, y), bilateral_grad),
             grad_target);
    var.set(x, y, var_val);
  }
}

template <RobustLoss loss_type, typename CudaArrayType>
__global__ void OptimizeY(const ImageDim image_dim, CudaArrayType const_var,
                          CudaArrayType target, CudaArrayType confidence,
                          CudaArrayType ct_dir, const float sigma_z,
                          const CudaArrayType intergrated_const_var,
                          const float lambda, CudaArrayType var) {
  const int y = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    int lower_bound;
    int upper_bound;

    LinearSearchLowerAndUpperBound2DX(image_dim, &ct_dir, sigma_z, y, x,
                                      &lower_bound, &upper_bound);

    const float bilateral_grad = (intergrated_const_var.get(upper_bound, x) -
                                  intergrated_const_var.get(lower_bound, x)) /
                                 (upper_bound - lower_bound);

    const float var_val =
        const_var.get(x, y) +
        Step(lambda,
             GaussianLossGradient(0.5, const_var.get(x, y), bilateral_grad), 0);
    var.set(x, y, var_val);
  }
}

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_SOLVER_CUH_
