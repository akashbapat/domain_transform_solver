#include "domain_transform_filter.h"

#include <cmath>
#include <iostream>

#include "cudaArray2D.h"
#include "cudaSurface2D.h"

#include "color_space.cuh"
#include "domain_transform_common_functions.h"
#include "domain_transform_common_kernels.cuh"
#include "domain_transform_filter.cuh"
#include "domain_transform_filter_struct.cuh"
#include "error_types.h"

namespace domain_transform {

void DomainTransformFilter::InitFrame(const COLOR_SPACE &color_space,
                                      void *color_image) {
  // Copy inputs to GPU.
  filter_struct_->color_image.View(0, 0, image_dims_.width,
                                   image_dims_.height) =
      reinterpret_cast<uchar4 *>(color_image);
  GPU_CHECK(cudaPeekAtLastError());

  switch (color_space) {
    case COLOR_SPACE::RGB: {
      break;
    }
    case COLOR_SPACE::YCbCr: {
      dim3 block_dim, grid_dim;
      ComputeBlockAndGridDim2D<false>(image_dims_, &block_dim, &grid_dim);

      RGB2YCbCr<<<grid_dim, block_dim>>>(image_dims_,
                                         filter_struct_->color_image);
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case COLOR_SPACE::YYY: {
      dim3 block_dim, grid_dim;
      ComputeBlockAndGridDim2D<false>(image_dims_, &block_dim, &grid_dim);

      RGB2YYY<<<grid_dim, block_dim>>>(image_dims_,
                                       filter_struct_->color_image);
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
  }
}

void DomainTransformFilter::Download(const ImageType &image_type,
                                     void *image) const {
  switch (image_type) {
    case ImageType::COLOR_IMAGE: {
      filter_struct_->color_image
          .View(0, 0, image_dims_.width, image_dims_.height)
          .CopyTo(reinterpret_cast<uchar4 *>(image));
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case ImageType::DIFFERENTIAL: {
      filter_struct_->dHdx.View(0, 0, image_dims_.width, image_dims_.height)
          .CopyTo(reinterpret_cast<float *>(image));
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
    case ImageType::INTEGRAL: {
      filter_struct_->ct_H.View(0, 0, image_dims_.width, image_dims_.height)
          .CopyTo(reinterpret_cast<float *>(image));
      GPU_CHECK(cudaPeekAtLastError());
      break;
    }
  }
}

DomainTransformFilter::DomainTransformFilter(const ImageDim &max_image_dims)
    : filter_struct_(new DomainTransformFilterStruct(max_image_dims.width,
                                                     max_image_dims.height)),
      max_image_dims_(max_image_dims),
      image_dims_(max_image_dims) {
  // (TODO:akashbapat) Convert the linear search to binary.
  // (TODO:akashbapat) Convert the gradient descent to conjugate gradients.
}

void DomainTransformFilter::IntegrateColorDifferentials() {
  IntegrateVariable(image_dims_, filter_struct_->dHdx, &filter_struct_->ct_H);

  IntegrateVariable(image_dims_, filter_struct_->dVdy, &filter_struct_->ct_V,
                    &filter_struct_->parallel_scan_transpose);
}

void DomainTransformFilter::ComputeColorSpaceDifferential(
    const DomainFilterParams &filter_params) {
  dim3 block_dim, grid_dim;
  ComputeBlockAndGridDim2D<false>(image_dims_, &block_dim, &grid_dim);
  const float sigma_x_div_sigma_r =
      filter_params.sigma_x / filter_params.sigma_r;
  Compute_dHdx<<<grid_dim, block_dim>>>(image_dims_, sigma_x_div_sigma_r,
                                        filter_struct_->color_image,
                                        filter_struct_->dHdx);
  GPU_CHECK(cudaPeekAtLastError());

  const float sigma_y_div_sigma_r =
      filter_params.sigma_y / filter_params.sigma_r;

  Compute_dVdy<<<grid_dim, block_dim>>>(image_dims_, sigma_y_div_sigma_r,
                                        filter_struct_->color_image,
                                        filter_struct_->dVdy);
  GPU_CHECK(cudaPeekAtLastError());
}

DomainTransformFilter::~DomainTransformFilter() {}

void DomainTransformFilter::SetImageDim(const ImageDim &image_dim) {
  if (image_dim.width > 0 && image_dim.width <= max_image_dims_.width &&
      image_dim.height > 0 && image_dim.height <= max_image_dims_.height) {
    image_dims_ = image_dim;
  } else {
    std::cerr << "Invalid image dims WxH:" << image_dim.width << " x "
              << image_dim.height << std::endl;
  }
}

void DomainTransformFilter::Filter(const DomainFilterParams &filter_params,
                                   const int num_iter,
                                   const cua::CudaArray2D<float> *input,
                                   cua::CudaArray2D<float> *output) {
  ComputeColorSpaceDifferential(filter_params);
  IntegrateColorDifferentials();

  dim3 block_dim, grid_dim;
  ComputeBlockAndGridDim2D<false>(image_dims_, &block_dim, &grid_dim);

  cua::CudaArray2D<float> filtered_var = filter_struct_->var_buffer;
  cua::CudaArray2D<float> var = filter_struct_->var;
  CopyVariable<<<grid_dim, block_dim>>>(image_dims_, *input, var);

  GPU_CHECK(cudaPeekAtLastError());

  filtered_var.Fill(0.0f);

  for (int iter = 0; iter < num_iter; iter++) {
    float sigma_x = filter_params.sigma_x;
    float sigma_y = filter_params.sigma_y;

    if (iter != 0) {
      const float multiplier = 3 * std::pow(2, (num_iter - (iter + 1))) /
                               std::sqrt(std::pow(4, num_iter) - 1);

      sigma_x *= multiplier;
      sigma_y *= multiplier;
    }

    IntegrateVariable(image_dims_, var, &filter_struct_->summed_area_x);

    FilterX<<<grid_dim, block_dim>>>(image_dims_, filter_struct_->ct_H, sigma_x,
                                     filter_struct_->summed_area_x,
                                     filtered_var);

    GPU_CHECK(cudaPeekAtLastError());

    //  -----------------------------------------------------------
    // Swap buffers.
    cua::CudaArray2D<float> tmp = filtered_var;
    filtered_var = var;
    var = tmp;

    IntegrateVariable(image_dims_, var, &filter_struct_->summed_area_y,
                      &filter_struct_->parallel_scan_transpose);

    dim3 block_dim_t, grid_dim_t;
    ComputeBlockAndGridDim2D<true>(image_dims_, &block_dim_t, &grid_dim_t);

    FilterY<<<grid_dim_t, block_dim_t>>>(image_dims_, filter_struct_->ct_V,
                                         sigma_y, filter_struct_->summed_area_y,
                                         filtered_var);

    GPU_CHECK(cudaPeekAtLastError());
    //  -----------------------------------------------------------
    // Swap buffers.
    tmp = filtered_var;
    filtered_var = var;
    var = tmp;
  }

  CopyVariable<<<grid_dim, block_dim>>>(image_dims_, var, *output);
  GPU_CHECK(cudaPeekAtLastError());
}

void DomainTransformFilter::Filter(const DomainFilterParams &filter_params,
                                   const float *input, float *output,
                                   const int num_iter) {
  filter_struct_->var = input;
  Filter(filter_params, num_iter, &filter_struct_->var, &filter_struct_->var);

  filter_struct_->var.View(0, 0, image_dims_.width, image_dims_.height)
      .CopyTo(output);

  GPU_CHECK(cudaPeekAtLastError());
}

void DomainTransformFilter::ClearAll() {
  filter_struct_->color_image.Fill(make_uchar4(0, 0, 0, 0));
  filter_struct_->dHdx.Fill(0.0f);
  filter_struct_->dVdy.Fill(0.0f);
  filter_struct_->ct_H.Fill(0.0f);
  filter_struct_->ct_V.Fill(0.0f);

  // These change every iteration.
  filter_struct_->summed_area_x.Fill(0.0f);
  filter_struct_->summed_area_y.Fill(0.0f);
  filter_struct_->var.Fill(0.0f);
  filter_struct_->var_buffer.Fill(0.0f);

  filter_struct_->parallel_scan_transpose.Fill(0.0f);
}

DomainTransformFilterStruct *DomainTransformFilter::FilterStructPtr() {
  return filter_struct_.get();
}

}  // namespace domain_transform
