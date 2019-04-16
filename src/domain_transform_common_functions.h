#ifndef DOMAIN_TRANSFORM_COMMON_FUNCTIONS_H_
#define DOMAIN_TRANSFORM_COMMON_FUNCTIONS_H_

#include <cmath>
#include <iostream>

// Defines dim3.
#include <vector_types.h>

#include "domain_transform_common.h"
#include "domain_transform_common_kernels.cuh"

#include "cudaArray2D.h"
#include "cumsum_kernels2d.cuh"
#include "error_types.h"

namespace domain_transform {

template <const bool Transpose>
void ComputeBlockAndGridDim2D(const ImageDim &options, dim3 *block_dim,
                              dim3 *grid_dim) {
  constexpr int x_th = 32;
  constexpr int y_th = 32;
  if (Transpose) {
    *block_dim = dim3(x_th, y_th, 1);
    *grid_dim = dim3(std::ceil(options.height / (x_th * 1.0f)),
                     std::ceil(options.width / (y_th * 1.0f)), 1);

  } else {
    *block_dim = dim3(x_th, y_th, 1);
    *grid_dim = dim3(std::ceil(options.width / (x_th * 1.0f)),
                     std::ceil(options.height / (y_th * 1.0f)), 1);
  }
}

void IfInvalidSetTo(const float set_val, float *check_val) {
  if (std::isnan(*check_val) || std::isinf(*check_val)) {
    std::cerr << "Invalid value of sigma params " << *check_val
              << ", resetting to " << set_val << std::endl;
    *check_val = set_val;
  }
}

void IntegrateVariableParallel(const ImageDim &image_dim,
                               const cua::CudaArray2D<float> &var,
                               cua::CudaArray2D<float> *summed_area) {
  // Parameterize a tile-scanning policy
  constexpr int ITEMS_PER_THREAD = 4;
  const int num_rows = image_dim.height;

  if (image_dim.width < 32 * ITEMS_PER_THREAD) {
    std::cerr << "Image is too small WxH: " << image_dim.width << "x"
              << image_dim.height << "." << std::endl;
    std::cerr << "The smallest image supported is 128x128." << std::endl;

    std::exit(-1);

  } else if (image_dim.width < 64 * ITEMS_PER_THREAD) {
    constexpr int BLOCK_THREADS = 32;
    SegmentedScanParallel2DHorizontal<
        BLOCK_THREADS,
        ITEMS_PER_THREAD><<<dim3(1, num_rows, 1), dim3(BLOCK_THREADS, 1, 1)>>>(
        image_dim, var, *summed_area);
    GPU_CHECK(cudaPeekAtLastError());

  } else if (image_dim.width < 128 * ITEMS_PER_THREAD) {
    constexpr int BLOCK_THREADS = 64;
    SegmentedScanParallel2DHorizontal<
        BLOCK_THREADS,
        ITEMS_PER_THREAD><<<dim3(1, num_rows, 1), dim3(BLOCK_THREADS, 1, 1)>>>(
        image_dim, var, *summed_area);
    GPU_CHECK(cudaPeekAtLastError());
  } else {
    constexpr int BLOCK_THREADS = 128;

    SegmentedScanParallel2DHorizontal<
        BLOCK_THREADS,
        ITEMS_PER_THREAD><<<dim3(1, num_rows, 1), dim3(BLOCK_THREADS, 1, 1)>>>(
        image_dim, var, *summed_area);
    GPU_CHECK(cudaPeekAtLastError());
  }
}

void IntegrateVariable(const ImageDim &image_dims,
                       const cua::CudaArray2D<float> &var,
                       cua::CudaArray2D<float> *summed_area,
                       cua::CudaArray2D<float> *tmp = nullptr) {
  if (tmp != nullptr) {
    dim3 block_dim, grid_dim;
    ComputeBlockAndGridDim2D<false>(image_dims, &block_dim, &grid_dim);

    CopyVariableTranspose<<<grid_dim, block_dim>>>(image_dims, var, *tmp);

    GPU_CHECK(cudaPeekAtLastError());
    IntegrateVariableParallel(image_dims.Transpose(), *tmp, summed_area);
  } else {
    IntegrateVariableParallel(image_dims, var, summed_area);
  }
}

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_COMMON_FUNCTIONS_H_
