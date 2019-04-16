#ifndef DOMAIN_TRANSFORM_FILTER_STRUCT_H_
#define DOMAIN_TRANSFORM_FILTER_STRUCT_H_

#include "cudaArray2D.h"
#include "cudaSurface2D.h"

namespace domain_transform {

struct DomainTransformFilterStruct {
  // Holds the 4 channel color image, all channels are used so the input should
  // have 255 (or constant) in all pixels if last channel is not being used.
  cua::CudaSurface2D<uchar4> color_image;

  // dHdx = sqrt( 1 + \sum_k I'^2(x)), ie the integrand in Eqn 5 in the paper
  // Bapat and Frahm, The Domain Transform Solver. The prime(') represents
  // differential in x direction.
  cua::CudaArray2D<float> dHdx;

  // dHdx = sqrt( 1 + \sum_k I'^2(y)), ie the integrand in Eqn 5 in the paper
  // Bapat and Frahm, The Domain Transform Solver.  The prime(') represents
  // differential in y direction.
  cua::CudaArray2D<float> dVdy;

  // ct_H = DT(u) in Eqn 5 in the paper Bapat and Frahm, The Domain Transform
  // Solver but in x direction. This is the domain transform value, ie integral
  // of dHdx.
  cua::CudaArray2D<float> ct_H;

  // ct_V = DT(u) in Eqn 5 in the paper Bapat and Frahm, The Domain Transform
  // Solver but in y direction. This is the domain transform value, ie integral
  // of dVdy.
  cua::CudaArray2D<float> ct_V;

  // These change every iteration.
  // This is a buffer to integrate the solution in x direction for fast
  // computation of box blur with variable sizes.
  cua::CudaArray2D<float> summed_area_x;

  // Same as above, but in y direction.  Note that the size of this buffer is
  // transposed.
  cua::CudaArray2D<float> summed_area_y;

  // This is a buffer which saves the input and at the end of the optimization
  // also holds the result.
  cua::CudaArray2D<float> var;
  cua::CudaArray2D<float> var_buffer;

  // This is a buffer useful for computing integrals in parallel. Note that the
  // size of this buffer is transposed.
  cua::CudaArray2D<float> parallel_scan_transpose;

  DomainTransformFilterStruct(const int max_width, const int max_height)
      : color_image(max_width, max_height),
        dHdx(max_width, max_height),
        dVdy(max_width, max_height),
        ct_H(max_width, max_height),
        ct_V(max_height, max_width),
        summed_area_x(max_width, max_height),
        summed_area_y(max_height, max_width),
        var(max_width, max_height),
        var_buffer(max_width, max_height),
        parallel_scan_transpose(max_height, max_width) {}
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_FILTER_STRUCT_H_
