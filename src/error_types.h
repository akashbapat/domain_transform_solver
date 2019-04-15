#ifndef ERROR_TYPES_H_
#define ERROR_TYPES_H_

#include <ostream>
#include <string>

#include "error_types_cuda.h"

#define RETURN_IF_ERROR(err) \
  {                          \
    \
if(!err.IsSuccessful()) {    \
      return err;            \
    }                        \
  \
}

namespace util {

enum class ERROR {
  SUCCESS,
  UNKNOWN,
  INVALID_ARGUMENT,
  OUT_OF_RANGE,
  INTERNAL_ERROR,
  IO_ERROR,
  CUDA_ERROR
};

std::ostream &operator<<(std::ostream &os, const ERROR &obj);

class ExecutionState {
 public:
  ExecutionState(ERROR err, std::string error_msg);
  ExecutionState() = delete;
  bool IsInternalError() const;
  bool IsInvalidArgumentError() const;
  bool IsOutOfRangeError() const;
  bool IsUnknownError() const;
  bool IsSuccessful() const;
  bool IsIOError() const;
  bool IsCudaError() const;

  friend std::ostream &operator<<(std::ostream &os,
                                  const ExecutionState &state);

 private:
  ERROR error_;
  std::string error_msg_;
};

// Convenient constructors for ExecutionState.
ExecutionState InternalError(const std::string msg);
ExecutionState InvalidArgument(const std::string msg);
ExecutionState OutOfRange(const std::string msg);
ExecutionState UnknownError(const std::string msg);
ExecutionState IOError(const std::string msg);
ExecutionState CudaError(const std::string msg);
ExecutionState Success();

}  // namespace util

#endif  // ERROR_TYPES_H_
