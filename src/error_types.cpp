#include "error_types.h"

namespace util {

std::ostream &operator<<(std::ostream &os, const ERROR &obj) {
  std::string msg;
  switch (obj) {
    case ERROR::SUCCESS: {
      msg = "Success";
      break;
    }

    case ERROR::UNKNOWN: {
      msg = "Unknown error encountered";
      break;
    }

    case ERROR::INVALID_ARGUMENT: {
      msg = "Invalid argument";
      break;
    }
    case ERROR::OUT_OF_RANGE: {
      msg = "Out of range error";
      break;
    }
    case ERROR::INTERNAL_ERROR: {
      msg = "Internal error";
      break;
    }
    case ERROR::IO_ERROR: {
      msg = "Input-output error";
      break;
    }
    case ERROR::CUDA_ERROR: {
      msg = "Cuda error";
      break;
    }
  }
  return os << msg;
}

ExecutionState::ExecutionState(ERROR err, std::string error_msg)
    : error_(err), error_msg_(error_msg) {}

bool ExecutionState::IsInternalError() const {
  return (error_ == ERROR::INTERNAL_ERROR);
}
bool ExecutionState::IsInvalidArgumentError() const {
  return (error_ == ERROR::INVALID_ARGUMENT);
}
bool ExecutionState::IsOutOfRangeError() const {
  return (error_ == ERROR::OUT_OF_RANGE);
}

bool ExecutionState::IsUnknownError() const {
  return (error_ == ERROR::UNKNOWN);
}

bool ExecutionState::IsIOError() const { return (error_ == ERROR::IO_ERROR); }

bool ExecutionState::IsCudaError() const {
  return (error_ == ERROR::CUDA_ERROR);
}

bool ExecutionState::IsSuccessful() const { return (error_ == ERROR::SUCCESS); }

ExecutionState InternalError(const std::string msg) {
  return ExecutionState(ERROR::INTERNAL_ERROR, msg);
}

ExecutionState InvalidArgument(const std::string msg) {
  return ExecutionState(ERROR::INVALID_ARGUMENT, msg);
}

ExecutionState OutOfRange(const std::string msg) {
  return ExecutionState(ERROR::OUT_OF_RANGE, msg);
}

ExecutionState UnknownError(const std::string msg) {
  return ExecutionState(ERROR::UNKNOWN, msg);
}

ExecutionState IOError(const std::string msg) {
  return ExecutionState(ERROR::IO_ERROR, msg);
}

ExecutionState CudaError(const std::string msg) {
  return ExecutionState(ERROR::CUDA_ERROR, msg);
}

ExecutionState Success() { return ExecutionState(ERROR::SUCCESS, ""); }

std::ostream &operator<<(std::ostream &os, const ExecutionState &state) {
  return os << state.error_ << ": " << state.error_msg_ << std::endl;
}

}  // namespace util
