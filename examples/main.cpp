#include <iostream>
#include "matrix.hpp"

using namespace matrixlib;

int main()
{
    Matrix<int> A{{1, 2}, {3, 4}};
    Matrix<double> B{{1.5, 2.5}, {3.5, 4.5}};

    auto C = A + B;   // Matrix<double>
    auto D = 2.5 * A; // Matrix<double>
    auto E = B - 1;   // Matrix<double>
    auto F = 10 - A;  // Matrix<int>
    auto G = A / 2.0; // Matrix<double>

    std::cout << "A + B:\n"
              << C << "\n\n";
    std::cout << "2.5 * A:\n"
              << D << "\n\n";
    std::cout << "B - 1:\n"
              << E << "\n\n";
    std::cout << "10 - A:\n"
              << F << "\n\n";
    std::cout << "A / 2.0:\n"
              << G << "\n";
}
