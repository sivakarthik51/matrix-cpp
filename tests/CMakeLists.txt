# Compile Catch2 main once
add_library(catch_main STATIC catch_main.cpp)

# CPU tests
add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE matrix catch_main)
add_test(NAME MatrixTests COMMAND test_matrix)

# CUDA tests
if(TARGET matrix_cuda)
    add_executable(test_matrix_cuda test_matrix_cuda.cpp)
    target_link_libraries(test_matrix_cuda PRIVATE matrix_cuda catch_main)
    add_test(NAME MatrixCudaTests COMMAND test_matrix_cuda)
endif()
