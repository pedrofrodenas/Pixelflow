cmake_minimum_required (VERSION 3.21)
include(CMakePrintHelpers)

project (Pixelflow
	VERSION "0.0.2"
	DESCRIPTION "computer vision library,"
)

find_package(CUDA QUIET)
cmake_print_variables(CUDA_FOUND)
cmake_print_variables(WITH_CUDA)
if(CUDA_FOUND AND WITH_CUDA)
    set(CUDA_ENABLED ON)
    enable_language(CUDA)
endif()

add_subdirectory ("src")

configure_file(cmake/templates/pfconfig.h.in pfconfig.h)


file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY ${CMAKE_SOURCE_DIR}/figs DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")