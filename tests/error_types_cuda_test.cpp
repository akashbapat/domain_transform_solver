#include "error_types.h"

#include <gtest/gtest.h>

namespace util {
namespace cuda {

TEST(GPUAssertTest, GPUAssertDiesOnError) {
  EXPECT_EXIT(GPUAssert(cudaErrorMemoryAllocation, "", 0),
              ::testing::ExitedWithCode(cudaErrorMemoryAllocation), "");
}

TEST(GPUAssertTest, GPUAssertPassesOnSuccess) {
  GPUAssert(cudaSuccess, "", 0);
  std::cerr << "Reaches here." << std::endl;
}

}  // namespace cuda
}  // namespace util
