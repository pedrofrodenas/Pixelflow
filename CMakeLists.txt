cmake_minimum_required (VERSION 3.21)

project (Pixelflow
	VERSION "0.0.1"
	DESCRIPTION "computer vision library,"
)
add_subdirectory ("src")

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY ${CMAKE_SOURCE_DIR}/figs DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")