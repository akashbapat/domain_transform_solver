enable_testing()
include(CTest)

link_directories("../src")

set(GTEST_LIBRARY gtest gtest_main)

###########################################################
function(TEST_IT TEST_IT_NAME)
  set(options)
  set(one_value_args CMD_LINE_ARGS)
  set(multi_value_args)
 
  cmake_parse_arguments(TEST_IT_PREFIX "${options}"  "${one_value_args}" "${multi_value_args}" ${ARGN})
#
    if(NOT ${GTEST_FOUND})
        message(FATAL_ERROR "GTesting libraries not found. Aborting...")
    endif() # NOT GTEST_LIBRARY
#
    # Setup target
 
    target_link_libraries(${TEST_IT_NAME} ${GTEST_LIBRARY})
    add_test(NAME ${TEST_IT_NAME}
	COMMAND ${TEST_IT_NAME} ${TEST_IT_PREFIX_CMD_LINE_ARGS})
#
endfunction() # TEST_IT
###########################################################
#
#
#
###########################################################
add_custom_target(copy_testdata_dir ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/testdata
		${CMAKE_CURRENT_BINARY_DIR}/testdata
                 )
###########################################################
add_executable(error_types_test error_types_test.cpp)
target_link_libraries(error_types_test error_types)
TEST_IT(error_types_test)
###########################################################
add_executable(domain_transform_filter_test domain_transform_filter_test.cpp)
target_link_libraries(domain_transform_filter_test domain_transform_filter ${OPENCV_LIBRARIES})
add_dependencies(domain_transform_filter_test copy_testdata_dir)
TEST_IT(domain_transform_filter_test)
###########################################################
add_executable(domain_transform_solver_test domain_transform_solver_test.cpp)
target_link_libraries(domain_transform_solver_test domain_transform_solver ${OPENCV_LIBRARIES})
add_dependencies(domain_transform_solver_test copy_testdata_dir)
TEST_IT(domain_transform_solver_test)
###########################################################

