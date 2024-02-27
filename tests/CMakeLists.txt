

add_executable(tests main.cpp test.c)
target_link_libraries(tests Pixelflow)
target_include_directories(tests PUBLIC ${CMAKE_SOURCE_DIR})
