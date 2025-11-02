#include "catch_amalgamated.hpp"
#include "../include/matrix.hpp"

using namespace matrixlib;

TEST_CASE("Matrix basic arithmetic with same type", "[matrix-basic]")
{
    Matrix<int> A{{1, 2}, {3, 4}};
    Matrix<int> B{{5, 6}, {7, 8}};

    auto C = A + B;
    REQUIRE(C(0, 0) == 6);
    REQUIRE(C(1, 1) == 12);

    auto D = A * 2;
    REQUIRE(D(0, 1) == 4);
    REQUIRE(D(1, 0) == 6);

    auto E = A / 2;
    REQUIRE(E(0, 0) == 0); // integer division
}

TEST_CASE("Matrix arithmetic with mixed types", "[matrix-mixed]")
{
    Matrix<int> A{{1, 2}, {3, 4}};
    Matrix<double> B{{0.5, 1.5}, {2.5, 3.5}};

    auto C = A + B;
    REQUIRE(std::is_same<decltype(C), Matrix<double>>::value);
    REQUIRE(C(0, 0) == Catch::Approx(1.5));
    REQUIRE(C(1, 1) == Catch::Approx(7.5));

    auto D = 2.0 * A;
    REQUIRE(std::is_same<decltype(D), Matrix<double>>::value);
    REQUIRE(D(0, 0) == Catch::Approx(2.0));
    REQUIRE(D(1, 1) == Catch::Approx(8.0));

    auto E = A / 2.0;
    REQUIRE(E(0, 0) == Catch::Approx(0.5));
}

TEST_CASE("Matrix-scalar and scalar-matrix subtraction", "[matrix-subtract]")
{
    Matrix<int> A{{1, 2}, {3, 4}};

    auto B = 10 - A;
    REQUIRE(B(0, 0) == 9);
    REQUIRE(B(1, 1) == 6);

    auto C = A - 1.5;
    REQUIRE(std::is_same<decltype(C), Matrix<double>>::value);
    REQUIRE(C(0, 0) == Catch::Approx(-0.5));
}

TEST_CASE("Matrix streaming", "[matrix-print]")
{
    Matrix<int> A{{1, 2}, {3, 4}};
    std::ostringstream oss;
    oss << A;
    auto str = oss.str();

    REQUIRE(str.find("1") != std::string::npos);
    REQUIRE(str.find("4") != std::string::npos);
}

TEST_CASE("Matrix multiplication (integer)", "[matrix-mul-int]")
{
    Matrix<int> A{{1, 2, 3},
                  {4, 5, 6}};
    Matrix<int> B{{7, 8},
                  {9, 10},
                  {11, 12}};

    auto C = A * B; // 2x2 result
    REQUIRE(C.rows() == 2);
    REQUIRE(C.cols() == 2);

    // Expected:
    // [ 58, 64 ]
    // [139, 154]
    REQUIRE(C(0, 0) == 58);
    REQUIRE(C(0, 1) == 64);
    REQUIRE(C(1, 0) == 139);
    REQUIRE(C(1, 1) == 154);
}

TEST_CASE("Matrix multiplication (mixed types)", "[matrix-mul-mixed]")
{
    Matrix<int> A{{1, 2},
                  {3, 4}};
    Matrix<double> B{{0.5, 1.5},
                     {2.5, 3.5}};

    auto C = A * B;

    REQUIRE(std::is_same<decltype(C), Matrix<double>>::value);
    REQUIRE(C.rows() == 2);
    REQUIRE(C.cols() == 2);

    // Expected results:
    // [[1*0.5 + 2*2.5, 1*1.5 + 2*3.5],
    //  [3*0.5 + 4*2.5, 3*1.5 + 4*3.5]]
    REQUIRE(C(0, 0) == Catch::Approx(5.5));
    REQUIRE(C(0, 1) == Catch::Approx(8.5));
    REQUIRE(C(1, 0) == Catch::Approx(11.5));
    REQUIRE(C(1, 1) == Catch::Approx(18.5));
}

TEST_CASE("Matrix multiplication dimension mismatch throws", "[matrix-mul-error]")
{
    Matrix<int> A{{1, 2}, {3, 4}};
    Matrix<int> B{{1, 2, 3}};

    REQUIRE_THROWS_AS(A * B, std::runtime_error);
}
