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

  cuda_add_library(domain_transform_filter domain_transform_filter.h domain_transform_filter.cu)
  target_link_libraries(domain_transform_filter cub error_types)

  cuda_add_library(domain_transform_solver domain_transform_solver.h domain_transform_solver.cu)
  target_link_libraries(domain_transform_solver domain_transform_filter)
#
endif(BUILD_WITH_CUDA)
