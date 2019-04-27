#include "domain_transform_solver.h"

#include <iostream>
#include <limits>

#include "cudaArray2D.h"

#include "color_space.cuh"
#include "cub.cuh"
#include "cumsum_kernels2d.cuh"
#include "domain_transform_common_functions.h"
#include "domain_transform_common_kernels.cuh"
#include "domain_transform_filter.cuh"
#include "domain_transform_filter_struct.cuh"
#include "domain_transform_solver.cuh"
#include "domain_transform_solver_struct.cuh"
#include "error_types.h"

namespace domain_transform {

template <typename CudaArrayType1, typename CudaArrayType2,
          typename CudaArrayType3>
__global__ void ComputeConfidenceKernel(const ImageDim image_dim,
                                        const float variance_scale,
                                        const int left_clear_width,
                                        CudaArrayType1 dtz2, CudaArrayType2 dtz,
                                        CudaArrayType3 confidence) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 0 && y >= 0 && x < image_dim.width && y < image_dim.height) {
    if (x > left_clear_width) {
      const float dtz_val = dtz.get(x, y);
      const float variance = dtz2.get(x, y) - dtz_val * dtz_val;
      const float conf = variance < 0 || !isfinite(variance)
                             ? 0
                             : expf(-variance * variance_scale);

      confidence.set(x, y, conf);
    } else {
      confidence.set(x, y, 0);
    }
  }
}

void DomainTransformSolver::InitFrame(const COLOR_SPACE &color_space,
                                      void *color_image, float *target,
                                      float *confidence) {
  DomainTransformFilter::InitFrame(color_space, color_image);

  solver_struct_->target.View(0, 0, image_dims_.width, image_dims_.height) =
      target;
  GPU_CHECK(cudaPeekAtLastError());
  if (confidence == nullptr) {
    solver_struct_->confidence.Fill(1.0f);
  } else {
    solver_struct_->confidence.View(0, 0, image_dims_.width,
                                    image_dims_.height) = confidence;
    GPU_CHECK(cudaPeekAtLastError());
  }
}

void DomainTransformSolver::Download(const ImageType &image_type,
                                     void *image) const {
  switch (image_type) {
    case ImageType::TARGET: {
      solver_struct_->target.View(0, 0, image_dims_.width, image_dims_.height)
          .CopyTo(reinterpret_cast<float *>(image));
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case ImageType::CONFIDENCE: {
      solver_struct_->confidence
          .View(0, 0, image_dims_.width, image_dims_.height)
          .CopyTo(reinterpret_cast<float *>(image));
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case ImageType::OPTIMIZED_QUANTITY: {
      filter_struct_->var.View(0, 0, image_dims_.width, image_dims_.height)
          .CopyTo(reinterpret_cast<float *>(image));
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    default: { DomainTransformFilter::Download(image_type, image); }
  }
}

void DomainTransformSolver::ComputeColorSpaceDifferential(
    const DomainFilterParams &domain_filter_params) {
  filter_params_ = domain_filter_params;
  DomainTransformFilter::ComputeColorSpaceDifferential(filter_params_);
}

DomainTransformSolver::DomainTransformSolver(const ImageDim &max_image_dims)
    : DomainTransformFilter(max_image_dims),
      solver_struct_(new DomainTransformSolverStruct(max_image_dims.width,
                                                     max_image_dims.height)) {}

void DomainTransformSolver::Optimize(
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

      switch (optimize_params.loss) {
        case RobustLoss::CHARBONNIER: {
          OptimizeX<RobustLoss::CHARBONNIER><<<grid_dim, block_dim>>>(
              image_dims_, var, solver_struct_->target,
              solver_struct_->confidence, filter_struct_->ct_H, sigma,
              optimize_params.step_size, filter_struct_->summed_area_x,
              optimize_params.lambda, filtered_var);

          GPU_CHECK(cudaPeekAtLastError());
          break;
        }
        case RobustLoss::L2: {
          OptimizeX<RobustLoss::L2><<<grid_dim, block_dim>>>(
              image_dims_, var, solver_struct_->target,
              solver_struct_->confidence, filter_struct_->ct_H, sigma,
              optimize_params.step_size, filter_struct_->summed_area_x,
              optimize_params.lambda, filtered_var);
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

DomainTransformSolver::~DomainTransformSolver() {}

void DomainTransformSolver::ComputeConfidence(
    const DomainFilterParamsVec<1> &conf_params,
    const int left_side_clear_width) {
  // Compute the confidence in an edge aware way using the domain trasform as a
  // means of local variance estimator. See
  // \url{https://drive.google.com/file/d/0B4nuwEMaEsnmdEREcjhlSXM2NGs/view}
  // Barron and Poole ECCV16 supplement page 6 for more details.

  // Assumes that the differentials have already been taken.

  // Compute DT(Z) at confidence.
  // Compute DT(Z^2) at confidence_square.
  // Compute  find variance = DT(Z^2) - DT(Z)^2 as confidence.
  dim3 block_dim, grid_dim;
  ComputeBlockAndGridDim2D<false>(image_dims_, &block_dim, &grid_dim);

  DomainFilterParams local_filter_params;
  local_filter_params.sigma_x = conf_params.sigma_x;
  local_filter_params.sigma_y = conf_params.sigma_y;
  local_filter_params.sigma_r = conf_params.sigma_r;

  const int kNimDTFilterIteration = 3;
  Filter(local_filter_params, kNimDTFilterIteration, &solver_struct_->target,
         &solver_struct_->confidence);
  SquareVariable<<<grid_dim, block_dim>>>(
      image_dims_, solver_struct_->target,
      solver_struct_->confidence_square_buffer);
  GPU_CHECK(cudaPeekAtLastError());

  Filter(local_filter_params, kNimDTFilterIteration,
         &solver_struct_->confidence_square_buffer,
         &solver_struct_->confidence_square);

  const float variance_scale =
      0.5f / (conf_params.sigmas[0] * conf_params.sigmas[0]);
  ComputeConfidenceKernel<<<grid_dim, block_dim>>>(
      image_dims_, variance_scale, left_side_clear_width,
      solver_struct_->confidence_square, solver_struct_->confidence,
      solver_struct_->confidence);
  GPU_CHECK(cudaPeekAtLastError());
}

void DomainTransformSolver::ClearAll() {
  DomainTransformFilter::ClearAll();

  solver_struct_->confidence.Fill(0.0f);
  solver_struct_->confidence_square.Fill(0.0f);
  solver_struct_->confidence_square_buffer.Fill(0.0f);
  solver_struct_->target.Fill(0.0f);
}

void DomainTransformSolver::IntegrateColorDifferentials() {
  DomainTransformFilter::IntegrateColorDifferentials();
}

}  // namespace domain_transform
