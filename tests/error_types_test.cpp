#include "error_types.h"

#include <gtest/gtest.h>

namespace util {

TEST(ErrorTest, Print) {
  constexpr ERROR er = ERROR::UNKNOWN;
  std::cerr << er << "." << std::endl;
}

TEST(ExecutionStateTest, InternalErrorConstructorAndPrint) {
  ExecutionState state = InternalError("This is internal error message.");
  std::cerr << state << std::endl;
}

TEST(ExecutionStateTest, InvalidArgumentConstructorAndPrint) {
  ExecutionState state =
      InvalidArgument("This is invalid argument error message.");
  std::cerr << state << std::endl;
}

TEST(ExecutionStateTest, OutOfRangeConstructorAndPrint) {
  ExecutionState state = OutOfRange("This is out of range error message.");
  std::cerr << state << std::endl;
}

TEST(ExecutionStateTest, UnknownErrorConstructorAndPrint) {
  ExecutionState state = UnknownError("This is unknown error message.");
  std::cerr << state << std::endl;
}

TEST(ExecutionStateTest, IOErrorConstructorAndPrint) {
  ExecutionState state = IOError("This is input-output error message.");
  std::cerr << state << std::endl;
}

TEST(ExecutionStateTest, CudaErrorConstructorAndPrint) {
  ExecutionState state = CudaError("This is Cuda error message.");
  std::cerr << state << std::endl;
}

TEST(ExecutionStateTest, SuccessConstructorAndPrint) {
  ExecutionState state = Success();
  std::cerr << state << std::endl;
}

TEST(ExecutionStateTest, IsInternalError) {
  ExecutionState state = InternalError("This is internal error message.");
  ASSERT_TRUE(state.IsInternalError())
      << "Constructor and error type query does not match for "
      << ERROR::INTERNAL_ERROR << ".";
}

TEST(ExecutionStateTest, IsInvalidArgumentError) {
  ExecutionState state =
      InvalidArgument("This is invalid argument error message.");
  ASSERT_TRUE(state.IsInvalidArgumentError())
      << "Constructor and error type query does not match for "
      << ERROR::INVALID_ARGUMENT << ".";
}

TEST(ExecutionStateTest, IsOutOfRangeError) {
  ExecutionState state = OutOfRange("This is out of range error message.");
  ASSERT_TRUE(state.IsOutOfRangeError())
      << "Constructor and error type query does not match for "
      << ERROR::OUT_OF_RANGE << ".";
}

TEST(ExecutionStateTest, IsUnknownError) {
  ExecutionState state = UnknownError("This is unknown error message.");
  ASSERT_TRUE(state.IsUnknownError())
      << "Constructor and error type query does not match for "
      << ERROR::UNKNOWN << ".";
}

TEST(ExecutionStateTest, IsIOError) {
  ExecutionState state = IOError("This is input-output error message.");
  ASSERT_TRUE(state.IsIOError())
      << "Constructor and error type query does not match for "
      << ERROR::IO_ERROR << ".";
}

TEST(ExecutionStateTest, IsCudaError) {
  ExecutionState state = CudaError("This is Cuda error message.");
  ASSERT_TRUE(state.IsCudaError())
      << "Constructor and error type query does not match for "
      << ERROR::CUDA_ERROR << ".";
}

TEST(ExecutionStateTest, Success) {
  ExecutionState state = Success();
  ASSERT_TRUE(state.IsSuccessful())
      << "Constructor and error type query does not match for "
      << ERROR::SUCCESS << ".";
}

}  // namespace util
