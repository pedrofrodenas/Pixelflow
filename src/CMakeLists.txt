add_library (Pixelflow SHARED
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

target_compile_features (Pixelflow
	PRIVATE cxx_std_11
)

SET_TARGET_PROPERTIES(Pixelflow PROPERTIES SUFFIX .so)

target_include_directories (Pixelflow
	PUBLIC "../include"
)

add_executable(tests main.c)
target_link_libraries(tests Pixelflow)