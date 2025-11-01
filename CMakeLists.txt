cmake_minimum_required(VERSION 3.16)
project(MatrixLib VERSION 1.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files (optional)
set(SOURCES
    src/matrix.cpp
)

# Build static library
add_library(matrix STATIC ${SOURCES})

# Include directories for consumers
target_include_directories(matrix PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Enable testing
enable_testing()
add_subdirectory(tests)

# Install rules
install(TARGETS matrix DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/matrixlib)
