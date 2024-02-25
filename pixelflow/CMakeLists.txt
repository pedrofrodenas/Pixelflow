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
    "test.c"
)

# Configure a header file to pass the version settings to the source code
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Pixelflow.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/Pixelflow.h")

configure_file(${PROJECT_SOURCE_DIR}/cmake/templates/pfconfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/pfconfig.h)

add_library (Pixelflow SHARED ${pixelflow_SOURCES})

target_compile_features (Pixelflow
	PRIVATE cxx_std_11
)

SET_TARGET_PROPERTIES(Pixelflow PROPERTIES SUFFIX .so)


cmake_print_variables(CMAKE_CURRENT_BINARY_DIR)

# Add header files to library
target_include_directories (Pixelflow PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_executable(tests main.c)
target_link_libraries(tests Pixelflow)
target_include_directories(tests PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

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



if(CUDA_ENABLED)
     #collect CUDA files
     FILE(GLOB_RECURSE CUDA_SRC  *.cu)
     FILE(GLOB_RECURSE CUDA_HDR *.h)
     
     set(CUDA_SRC ${CUDA_SRC} core/cudaUtils.cpp)
     

endif()


if(CUDA_ENABLED)
    add_library(pixelflow_cuda SHARED ${CUDA_SRC})
    target_include_directories(pixelflow_cuda
        PUBLIC "../include"
        ${PROJECT_BINARY_DIR}
        )

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${PIXELFLOW_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.cuh"
    )

    cmake_print_variables(PIXELFLOW_INSTALL_INCLUDE_DIR)

    target_include_directories(pixelflow_cuda SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(pixelflow_cuda PUBLIC Pixelflow ${CUDA_LIBRARIES})
    target_compile_definitions(pixelflow_cuda PUBLIC MYMATH_USE_CUDA)


    add_executable(mMulShared cuda/imageBlur.cu)
    target_include_directories(mMulShared PRIVATE "../include")
    target_link_libraries (mMulShared PRIVATE Pixelflow)

    # add_executable(test test.cpp)
    # target_include_directories(test PRIVATE "../include" ${CUDA_HDR})
    # target_link_libraries (test PRIVATE pixelflow_cuda)
endif()