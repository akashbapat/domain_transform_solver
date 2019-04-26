#include "domain_transform_stereo.h"

#include <iostream>

#include "cudaArray2D.h"

#include "domain_transform_common_functions.h"
#include "domain_transform_common_kernels.cuh"
#include "domain_transform_filter_struct.cuh"
#include "domain_transform_loss_term.cuh"
#include "domain_transform_solver.cuh"
#include "domain_transform_solver_struct.cuh"
#include "error_types.h"

namespace domain_transform {

void DomainTransformStereo::InitRightFrame(const float center_offset,
                                           const COLOR_SPACE &color_space,
                                           void *color_image) {
  center_offset_ = center_offset;
  right_filter_->InitFrame(color_space, color_image);
}

void DomainTransformStereo::ProcessRightFrame() {
  right_filter_->ComputeColorSpaceDifferential(filter_params_);
  right_filter_->IntegrateColorDifferentials();
}

void DomainTransformStereo::SetImageDim(const ImageDim &image_dim) {
  if (image_dim.width > 0 && image_dim.width <= max_image_dims_.width &&
      image_dim.height > 0 && image_dim.height <= max_image_dims_.height) {
    image_dims_ = image_dim;
    right_filter_->SetImageDim(image_dim);
  } else {
    std::cerr << "Invalid image dims WxH:" << image_dim.width << " x "
              << image_dim.height << std::endl;
  }
}

DomainTransformStereo::DomainTransformStereo(const ImageDim &max_image_dims)
    : DomainTransformSolver(max_image_dims),
      right_filter_(new DomainTransformFilter(max_image_dims)) {}

void DomainTransformStereo::Optimize(
    const DomainOptimizeParams &optimize_params,
    const float overwrite_target_above_conf) {
  dim3 block_dim, grid_dim;
  ComputeBlockAndGridDim2D<false>(image_dims_, &block_dim, &grid_dim);

  cua::CudaArray2D<float> filtered_var = filter_struct_->var_buffer;
  cua::CudaArray2D<float> var = filter_struct_->var;
  CopyVariable<<<grid_dim, block_dim>>>(image_dims_, solver_struct_->target,
                                        var);
  GPU_CHECK(cudaPeekAtLastError());

  filtered_var.Fill(0.0f);

  constexpr int kDMIter = 1;

  for (int i = 0; i < optimize_params.num_iterations; i++) {
    for (int j = 0; j < kDMIter; j++) {
      float sigma = optimize_params.sigma_z;

      if (j != 0) {
        const float multiplier = 3 * std::pow(2, (kDMIter - (j + 1))) /
                                 std::sqrt(std::pow(4, kDMIter) - 1);

        sigma *= multiplier;
      }
      if (overwrite_target_above_conf > 0 && overwrite_target_above_conf < 1) {
        CopyVariableFlagGreaterThanThresh<<<grid_dim, block_dim>>>(
            image_dims_, overwrite_target_above_conf,
            solver_struct_->confidence, solver_struct_->target, var);
      }
      IntegrateVariable(image_dims_, var, &filter_struct_->summed_area_x);

      const float sigma_x_div_sigma_r =
          filter_params_.sigma_r / filter_params_.sigma_x;

      const float grad_alpha = sigma_x_div_sigma_r * sigma_x_div_sigma_r;
      const float grad_beta = -1;

      std::cerr << " ccenter_offset_ " << center_offset_ << ", grad_alpha "
                << grad_alpha << ", grad_beta " << grad_beta
                << ", optimize_params.photo_consis_lambda "
                << optimize_params.photo_consis_lambda << std::endl
                << std::endl;

#if 1

      PhotometricL2LossTerm loss_term(
          center_offset_, optimize_params.photo_consis_lambda,
          filter_struct_->color_image,
          right_filter_->FilterStructPtr()->color_image,
          right_filter_->FilterStructPtr()->dHdx, grad_alpha, grad_beta);

      printf("lambda %f, grad_alpha %f, grad_beta %f, center_offset %f \n\n",
             loss_term.lambda, loss_term.grad_alpha, loss_term.grad_beta,
             loss_term.center_offset);
#else
      PhotometricL2LossTermTest loss_term(center_offset_,
                                          optimize_params.photo_consis_lambda,
                                          grad_alpha, grad_beta);

      printf("lambda %f, grad_alpha %f, grad_beta %f, center_offset %f \n\n",
             loss_term.lambda, loss_term.grad_alpha, loss_term.grad_beta,
             loss_term.center_offset);
#endif
      switch (optimize_params.loss) {
        case RobustLoss::CHARBONNIER: {
          OptimizeX<RobustLoss::CHARBONNIER><<<grid_dim, block_dim>>>(
              image_dims_, var, solver_struct_->target,
              solver_struct_->confidence, filter_struct_->ct_H, sigma,
              optimize_params.step_size, filter_struct_->summed_area_x,
              optimize_params.lambda, filtered_var, loss_term);

          GPU_CHECK(cudaPeekAtLastError());
          break;
        }
        case RobustLoss::L2: {
          OptimizeX<RobustLoss::L2><<<grid_dim, block_dim>>>(
              image_dims_, var, solver_struct_->target,
              solver_struct_->confidence, filter_struct_->ct_H, sigma,
              optimize_params.step_size, filter_struct_->summed_area_x,
              optimize_params.lambda, filtered_var, loss_term);
          GPU_CHECK(cudaPeekAtLastError());
          break;
        }
      }

      //  -----------------------------------------------------------
      // Swap buffers.
      cua::CudaArray2D<float> tmp = filtered_var;
      filtered_var = var;
      var = tmp;
      if (overwrite_target_above_conf > 0 && overwrite_target_above_conf < 1) {
        CopyVariableFlagGreaterThanThresh<<<grid_dim, block_dim>>>(
            image_dims_, overwrite_target_above_conf,
            solver_struct_->confidence, solver_struct_->target, var);
      }
      IntegrateVariable(image_dims_, var, &filter_struct_->summed_area_y,
                        &filter_struct_->parallel_scan_transpose);

      dim3 block_dim_t, grid_dim_t;
      ComputeBlockAndGridDim2D<true>(image_dims_, &block_dim_t, &grid_dim_t);

      switch (optimize_params.loss) {
        case RobustLoss::CHARBONNIER: {
          OptimizeY<RobustLoss::CHARBONNIER><<<grid_dim_t, block_dim_t>>>(
              image_dims_, var, solver_struct_->target,
              solver_struct_->confidence, filter_struct_->ct_V, sigma,
              optimize_params.step_size, filter_struct_->summed_area_y,
              optimize_params.lambda, filtered_var);
          GPU_CHECK(cudaPeekAtLastError());
          break;
        }
        case RobustLoss::L2: {
          OptimizeY<RobustLoss::L2><<<grid_dim_t, block_dim_t>>>(
              image_dims_, var, solver_struct_->target,
              solver_struct_->confidence, filter_struct_->ct_V, sigma,
              optimize_params.step_size, filter_struct_->summed_area_y,
              optimize_params.lambda, filtered_var);
          GPU_CHECK(cudaPeekAtLastError());
          break;
        }
      }

      //  -----------------------------------------------------------
      // Swap buffers.
      tmp = filtered_var;
      filtered_var = var;
      var = tmp;
    }
  }

  CopyVariable<<<grid_dim, block_dim>>>(image_dims_, var, filter_struct_->var);
  GPU_CHECK(cudaPeekAtLastError());
}

DomainTransformStereo::~DomainTransformStereo() {}

}  // namespace domain_transform
