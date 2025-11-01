# Executable for unit tests
add_executable(test_matrix
    test_matrix.cpp
    catch_amalgamated.cpp
)

# Link the library
target_link_libraries(test_matrix PRIVATE matrix)

# Add to CTest
add_test(NAME MatrixTests COMMAND test_matrix)
