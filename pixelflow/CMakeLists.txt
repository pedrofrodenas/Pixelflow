set (pixelflow_SOURCES
    "color/color.c"
    "spatial-filters/linear.c"
    "core/math.c"
    "core/normalization.c"
    "core/operation.c"
    "core/resize.c"
    "panorama/harris_image.c"
    "panorama/panorama_image.c"
    "opticalflow/flow_image.c"
    "neural/classifier.c"
    "args.c"
    "data.c"
    "image_opencv.cpp"
    "list.c"
    "load_image.c"
    "matrix.c"
    "core/cudaUtils.cpp"
)

# Configure a header file to pass the version settings to the source code
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Pixelflow.h.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/Pixelflow.h")

configure_file(${PROJECT_SOURCE_DIR}/cmake/templates/pfconfig.h.in ${CMAKE_CURRENT_SOURCE_DIR}/pfconfig.h)

add_library (Pixelflow SHARED ${pixelflow_SOURCES})

target_compile_features (Pixelflow
	PRIVATE cxx_std_11
)

SET_TARGET_PROPERTIES(Pixelflow PROPERTIES SUFFIX .so)


cmake_print_variables(CMAKE_SOURCE_DIR)

# Add header files to library
target_include_directories (Pixelflow PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


# if(CUDA_ENABLED)
     
#      set(CUDA_SRC ${CUDA_SRC} core/cudaUtils.cpp)
# endif()


# if(CUDA_ENABLED)
#     add_library(pixelflow_cuda SHARED ${CUDA_SRC})

#     target_include_directories(pixelflow_cuda PRIVATE
#         ${CMAKE_SOURCE_DIR}
#         ${PROJECT_BINARY_DIR}
#         SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#         )

#     target_link_libraries(pixelflow_cuda PUBLIC Pixelflow ${CUDA_LIBRARIES})

#     # add_executable(imageBlur cuda/imageBlur.cu)
#     # target_link_libraries (imageBlur pixelflow_cuda)
#     # target_include_directories(imageBlur PUBLIC ${CMAKE_SOURCE_DIR})
# endif()

# Moves library and Cmake files to installation folder
install(TARGETS Pixelflow EXPORT PixelflowTargets
        RUNTIME DESTINATION ${Pixelflow_INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${Pixelflow_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${Pixelflow_INSTALL_LIB_DIR}
)

# Get header folder hierarchy and copies them to installation folder
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${Pixelflow_INSTALL_INCLUDE_DIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh"
)
