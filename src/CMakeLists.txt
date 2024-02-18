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

add_library (Pixelflow SHARED ${pixelflow_SOURCES})

target_compile_features (Pixelflow
	PRIVATE cxx_std_11
)

SET_TARGET_PROPERTIES(Pixelflow PROPERTIES SUFFIX .so)

target_include_directories (Pixelflow
	PUBLIC "../include"
    ${PROJECT_BINARY_DIR}
)

add_executable(tests main.c)
target_link_libraries(tests Pixelflow)

if(CUDA_ENABLED)
     #collect CUDA files
     FILE(GLOB_RECURSE CUDA_SRC  *.cu)
endif()


if(CUDA_ENABLED)
    add_library(pixelflow_cuda SHARED ${CUDA_SRC})
    target_include_directories(pixelflow_cuda
        PUBLIC "../include"
        ${PROJECT_BINARY_DIR}
        )

    target_include_directories(pixelflow_cuda SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(pixelflow_cuda PUBLIC Pixelflow ${CUDA_LIBRARIES})
    target_compile_definitions(pixelflow_cuda PUBLIC MYMATH_USE_CUDA)


    add_executable(mMulShared cuda/imageBlur.cu)
    target_include_directories(mMulShared PRIVATE "../include")
    target_link_libraries (mMulShared PRIVATE Pixelflow)
endif()