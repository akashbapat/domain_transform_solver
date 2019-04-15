#ifndef DOMAIN_TRANSFORM_COMMON_FUNCTIONS_H_
#define DOMAIN_TRANSFORM_COMMON_FUNCTIONS_H_

#include <cmath>
#include <iostream>

// Defines dim3.
#include <vector_types.h> 

#include "domain_transform_common.h"

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

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_COMMON_FUNCTIONS_H_
