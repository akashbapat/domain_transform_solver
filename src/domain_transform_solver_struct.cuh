#ifndef DOMAIN_TRANSFORM_SOLVER_STRUCT_H_
#define DOMAIN_TRANSFORM_SOLVER_STRUCT_H_

#include "cudaArray2D.h"
#include "error_types.h"

namespace domain_transform {
// DomainTransformSolverStruct holds the cuda memory buffers which are used to
// store input confidence and to automatically compute confidence if not
// provided as input.
struct DomainTransformSolverStruct {
  // Stores the confidence values.
  cua::CudaArray2D<float> confidence;

  // Used to compute the confidence.
  cua::CudaArray2D<float> confidence_square;

  // Used to compute the confidence.
  cua::CudaArray2D<float> confidence_square_buffer;

  // Input target map used to initialize the solution.
  cua::CudaArray2D<float> target;

  DomainTransformSolverStruct(const int max_width, const int max_height)
      : confidence(max_width, max_height),
        confidence_square(max_width, max_height),
        confidence_square_buffer(max_width, max_height),
        target(max_width, max_height) {
    // Check for successful allocation of memory.
    GPU_CHECK(cudaPeekAtLastError());
  }
};

}  // namespace domain_transform

#endif  // DOMAIN_TRANSFORM_SOLVER_STRUCT_H_
