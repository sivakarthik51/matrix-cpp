#include "catch_amalgamated.cpp"
#include "matrix_cuda.hpp"
#include <vector>
#include <iostream>
#include <cmath>

using namespace matrixlib;

// Helper for floating-point comparison
template <typename T>
bool approx_equal(T a, T b, T eps = 1e-5)
{
    return std::fabs(a - b) < eps;
}

TEST_CASE("MatrixCUDA basic operations", "[cuda]")
{

    // Test 2x3 matrices
    size_t rows = 2, cols = 3;
    std::vector<float> hA = {1, 2, 3, 4, 5, 6};
    std::vector<float> hB = {0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
    std::vector<float> hC(rows * cols);

    // Create CUDA matrices
    MatrixCUDA<float> A(rows, cols);
    MatrixCUDA<float> B(rows, cols);

    A.copyFromCPU(hA.data());
    B.copyFromCPU(hB.data());

    SECTION("Addition")
    {
        A.add(B);
        A.copyToCPU(hC.data());

        REQUIRE(approx_equal(hC[0], 1.5f));
        REQUIRE(approx_equal(hC[1], 3.5f));
        REQUIRE(approx_equal(hC[2], 5.5f));
        REQUIRE(approx_equal(hC[3], 7.5f));
        REQUIRE(approx_equal(hC[4], 9.5f));
        REQUIRE(approx_equal(hC[5], 11.5f));
    }

    // Optional: add more tests like multiplication, scalar ops
}
