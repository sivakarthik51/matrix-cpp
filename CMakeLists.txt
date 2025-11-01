cmake_minimum_required(VERSION 3.18)

# Project only uses CXX initially
project(MatrixLib VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable CUDA
option(ENABLE_CUDA "Build CUDA accelerated version" OFF)

# CPU library
add_library(matrix STATIC src/matrix.cpp)
target_include_directories(matrix PUBLIC include)

# Optional CUDA library
if(ENABLE_CUDA)
    enable_language(CUDA)
    add_library(matrix_cuda STATIC src/matrix_cuda.cu)
    target_include_directories(matrix_cuda PUBLIC include)
    set_target_properties(matrix_cuda PROPERTIES
        CUDA_STANDARD 17
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Tests
enable_testing()
add_subdirectory(tests)

# Installation
install(TARGETS matrix DESTINATION lib)
if(ENABLE_CUDA)
    install(TARGETS matrix_cuda DESTINATION lib)
endif()
install(DIRECTORY include/ DESTINATION include/matrixlib)
