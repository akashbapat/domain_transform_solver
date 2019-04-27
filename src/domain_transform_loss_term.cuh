#ifndef DOMAIN_TRANSFORM_LOSS_TERM_CUH_
#define DOMAIN_TRANSFORM_LOSS_TERM_CUH_

#include "domain_transform_common_kernels.cuh"

#include "cudaSurface2D.h"

namespace domain_transform {

struct PhotometricL2LossTerm {
  float center_offset;
  float lambda;
  float grad_alpha;
  float grad_beta;  // (gradient + beta) * alpha = actual_gradient.

  cua::CudaSurface2D<uchar4> left_image;
  cua::CudaSurface2D<uchar4> right_image;
  cua::CudaArray2D<float> gradient;

  __device__ float operator()(const int x, const int y, const float d) {
    if (center_offset + x - d >= 0 &&
        center_offset + x - d < gradient.Width()) {
      const typename cua::CudaSurface2D<uchar4>::Scalar val1 =
          left_image.get(x, y);
      const typename cua::CudaSurface2D<uchar4>::Scalar val2 =
          right_image.get(static_cast<int>(center_offset + x - d), y);
      const float grad_right_val =
          gradient.get(static_cast<int>(center_offset + x - d), y);

      const float loss_grad =
          lambda * 2 * Uchar4SumOfAbsDiffNormalize(val1, val2) *
          ((grad_right_val * grad_right_val + grad_beta) * grad_alpha);

      return loss_grad;

    } else {
      return 0;
    }
  }

  PhotometricL2LossTerm(const float center_offset_val, const float lambda_val,
                        cua::CudaSurface2D<uchar4> term_1,
                        cua::CudaSurface2D<uchar4> term_2,
                        cua::CudaArray2D<float> gradient_image,
                        const float grad_alpha_val, const float grad_beta_val)
      : left_image(term_1),
        right_image(term_2),
        gradient(gradient_image),
        center_offset(center_offset_val),
        lambda(lambda_val),
        grad_alpha(grad_alpha_val),
        grad_beta(grad_beta_val) {}
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_LOSS_TERM_CUH_
