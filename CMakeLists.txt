cmake_minimum_required(VERSION 3.1)

project(DomainTransformSolver)

set (CMAKE_CXX_STANDARD 11)

option(BUILD_SHARED_LIBS "Build dynamic library" ON)

option(BUILD_WITH_CUDA "Build with CUDA" ON)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Optimization level." FORCE)
message("Setting the build to Debug")
endif()

# This option builds CLIs(command-line interfaces) which are helpful to run 
# local or small programs.
option(BUILD_CLI "Build CLIs" ON)

# Useful for testing, turn off if not using test code.
option(BUILD_TESTING "Build tests" ON)
################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

if(EXISTS ${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
    include(${CMAKE_SOURCE_DIR}/LocalConfig.cmake)
endif()


################################################################################
# Find packages
################################################################################

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
endif(BUILD_TESTING)

find_package(OpenCV REQUIRED)
set(OPENCV_LIBRARIES opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_calib3d)

if(BUILD_CLI)
# Useful for passing commandline arguments to program.
  find_package(gflags REQUIRED)
  include_directories(${GFLAGS_INCLUDE_DIR})
endif(BUILD_CLI)

if(BUILD_WITH_CUDA)

  find_package(CUDA REQUIRED)

  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  include_directories("${CUDA_INCLUDE_DIRS}")

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -O3 --expt-extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets)
  
  # These include_directories are here due to a bug in cmake for including INTERFACE libraries for cuda.
  include_directories(ext/cudaArray/libcua)
  include_directories(ext/cub/cub)
endif(BUILD_WITH_CUDA)
################################################################################
# Compiler specific configuration
################################################################################
#
if (CMAKE_COMPILER_IS_GNUCC)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -pedantic -Werror -Wno-sign-compare -Wno-variadic-macros")
else()
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-variadic-macros")
endif()

endif()
#
################################################################################
# Add sources
################################################################################
add_subdirectory(ext)
add_subdirectory(src)
include_directories(src)

if(BUILD_TESTING)
add_subdirectory(tests)
endif(BUILD_TESTING)
