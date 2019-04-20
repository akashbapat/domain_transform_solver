FILE(GLOB_RECURSE DTS_HEADERS "*.h")
add_custom_target(headers SOURCES ${DTS_HEADERS})

#-------------------------------------DTS code---------------------------------
add_library(error_types error_types.cpp)

if(BUILD_WITH_CUDA)
#
  # Add definitions for error types if using cuda.
  target_compile_definitions(error_types PUBLIC USE_CUDA)
  target_link_libraries(error_types ${CUDA_LIBRARIES})

  cuda_add_library(hfbs hfbs.h hfbs.cu)
  target_link_libraries(hfbs cuda_array error_types)

  cuda_add_library(domain_transform_filter domain_transform_filter.h domain_transform_filter.cu)
  target_link_libraries(domain_transform_filter cuda_array cub error_types)

  cuda_add_library(domain_transform_solver domain_transform_solver.h domain_transform_solver.cu)
  target_link_libraries(domain_transform_solver domain_transform_filter cuda_array)
#
#
#
  if(BUILD_CLI)
  #
    add_executable(benchmark_depth_super_res_cli benchmark_depth_super_res_cli.cpp)
    target_link_libraries(benchmark_depth_super_res_cli gflags domain_transform_solver ${OPENCV_LIBRARIES} hfbs)
  #
  endif(BUILD_CLI)
#
#
#
endif(BUILD_WITH_CUDA)
