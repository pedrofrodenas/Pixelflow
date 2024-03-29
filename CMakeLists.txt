cmake_minimum_required (VERSION 3.21)
include(CMakePrintHelpers)

project (Pixelflow
	VERSION "0.0.2"
	DESCRIPTION "computer vision library,"
    LANGUAGES C CXX
)

option(WITH_CUDA          "Build the CUDA module"                    OFF)

# Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(Pixelflow_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(Pixelflow_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(Pixelflow_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    # Put resources in */share/
    set(PIXELFLOW_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(Pixelflow_INSTALL_INCLUDE_DIR include)
    set(Pixelflow_INSTALL_BIN_DIR bin)
    set(Pixelflow_INSTALL_LIB_DIR lib)
    # Put resources in */bin, with executables / DLLs
    set(PIXELFLOW_INSTALL_CMAKE_DIR CMake)
endif()

if(WITH_CUDA)
    set(CUDA_ENABLED ON)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

cmake_print_variables(WITH_CUDA)

add_subdirectory ("pixelflow")
add_subdirectory ("tests")


file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
file(COPY ${CMAKE_SOURCE_DIR}/figs DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${PIXELFLOW_INSTALL_CMAKE_DIR})
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::)