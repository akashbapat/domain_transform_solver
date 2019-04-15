FILE(GLOB_RECURSE DTS_HEADERS "*.h")
add_custom_target(headers SOURCES ${DTS_HEADERS})

#-------------------------------------DTS code---------------------------------
add_library(error_types error_types.cpp)

if(BUILD_WITH_CUDA)
#
  # Add definitions for error types if using cuda.
  target_compile_definitions(error_types PUBLIC USE_CUDA)
  target_link_libraries(error_types ${CUDA_LIBRARIES})
#
endif(BUILD_WITH_CUDA)
