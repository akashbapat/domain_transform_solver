#ifndef ERROR_TYPES_CUDA_H_
#define ERROR_TYPES_CUDA_H_

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define GPU_RETURN_IF_ERROR(err)                                      \
  {                                                                   \
    \
if(err != cudaSuccess) {                                              \
      return util::CudaError(std::string(cudaGetErrorString(err)) +   \
                             " in file " + std::string(__FILE__) +    \
                             " at line " + std::to_string(__LINE__)); \
    }                                                                 \
  \
}

#define GPU_CHECK(ans) \
  { util::cuda::GPUAssert((ans), std::string(__FILE__), __LINE__); }

namespace util {
namespace cuda {

inline void GPUAssert(const cudaError_t& err, const std::string file,
                      int line) {
  if (err != cudaSuccess) {
    std::cerr << std::string(cudaGetErrorString(err)) + " in file " +
                     std::string(file) + " at line " + std::to_string(line)
              << std::endl;
    std::exit(err);
  }
}

}  // namespace cuda
}  // namespace util

#endif  // ERROR_TYPES_CUDA_H_
