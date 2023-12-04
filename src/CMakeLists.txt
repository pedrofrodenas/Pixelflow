add_library (Pixelflow SHARED
    "color/color.c"
    "spatial-filters/linear.c"
    "core/math.c"
    "core/normalization.c"
    "core/operation.c"
    "core/resize.c"
    "hw3/harris_image.c"
    "hw3/panorama_image.c"
    "hw4/flow_image.c"
    "hw5/classifier.c"
    "args.c"
    "data.c"
    "image_opencv.cpp"
    "list.c"
    "load_image.c"
    "matrix.c"
    "test.c"
)

target_compile_features (Pixelflow
	PRIVATE cxx_std_11
)

SET_TARGET_PROPERTIES(Pixelflow PROPERTIES SUFFIX .so)

target_include_directories (Pixelflow
	PUBLIC "../include"
)

add_executable(tests main.c)
target_link_libraries(tests Pixelflow)