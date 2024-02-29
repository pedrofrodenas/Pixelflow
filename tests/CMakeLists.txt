

add_executable(tests main.cpp test.c)
target_link_libraries(tests Pixelflow)
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR})


if(CUDA_ENABLED)

add_executable(cudaTest testCuda.cpp)
target_link_libraries(cudaTest pixelflow_cuda)
target_include_directories(cudaTest PUBLIC ${CMAKE_SOURCE_DIR})

endif()
