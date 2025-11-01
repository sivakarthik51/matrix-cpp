# 🧮 MatrixLib

<!-- Badges -->
[![Build Status](https://github.com/sivakarthik51/matrix-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/sivakarthik51/matrix-cpp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/<YOUR_USERNAME>/matrixlib?label=latest)](https://github.com/<YOUR_USERNAME>/matrixlib/releases)


A lightweight, header-only **C++17 matrix library** supporting generic numeric types (`int`, `float`, `double`, etc.) with type promotion, operator overloading, and easy integration.

MatrixLib is designed to be **simple, modern, and cross-platform**, with clean build and test automation using **Makefiles** and **GitHub Actions CI**.

---

## ✨ Features

- ✅ **Generic template design** — supports `int`, `float`, `double`, or mixed arithmetic
- ⚙️ **Automatic type promotion** (e.g., `Matrix<int> * Matrix<double>` → `Matrix<double>`)
- ➕ **Operator overloading** for:
  - Scalar arithmetic (`+`, `-`, `*`, `/`)
  - Matrix–matrix operations
  - Mixed scalar/matrix arithmetic
- 🧾 **Stream output (`<<`)** for easy printing and debugging
- 🧪 **Unit testing** via [Catch2](https://github.com/catchorg/Catch2)
- 🌍 **Cross-platform builds** via Makefile (Linux, macOS, Windows/MSYS2)
- 🏗️ **CI/CD pipeline** for automated build, test, and release packaging

---

## ⚙️ Building the Library

```bash
# Build in Release mode
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Run tests
ctest --output-on-failure

```

## 🔹 Installing the library and headers

```bash
cmake --install . --prefix /usr/local   # Linux/macOS
cmake --install . --prefix C:/MatrixLib # Windows
```

This installs:

Headers → include/matrixlib

Static library → libmatrix.a (or matrix.lib on Windows)


## 🧪 Running Tests

All tests are located in tests/test_matrix.cpp and use Catch2.

To run them manually:

```bash
./tests/test_matrix
```

Output example:

```scss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MatrixLib Test Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All tests passed (15 assertions in 7 test cases)
```

## 💡 Example Usage

```cpp
#include "matrix.hpp"
#include <iostream>
using namespace matrixlib;

int main() {
    Matrix<int> A{{1, 2, 3}, {4, 5, 6}};
    Matrix<double> B{{0.5, 1.5, 2.5}, {3.5, 4.5, 5.5}};

    auto C = A + B;  // automatic type promotion → Matrix<double>
    auto D = C * 2.0;

    std::cout << "Matrix C:\n" << C << "\n\n";
    std::cout << "Matrix D:\n" << D << std::endl;
}
```

Output:

```
Matrix C:
[[1.5, 3.5, 5.5]
 [7.5, 9.5, 11.5]]

Matrix D:
[[3, 7, 11]
 [15, 19, 23]]
```

## 📦 Installation

You can use MatrixLib as a header-only library:

1. Copy the include/ directory into your project

2. Include it:

```cpp
#include "matrix.hpp"
```

3. Compile with:

```bash
g++ -std=c++17 -Iinclude examples/main.cpp -o main
```

Or link the static library built by the Makefile:

```bash
make
g++ -std=c++17 main.cpp -Lbuild -lmatrix -Iinclude -o main
```

## 🧠 Requirements
- C++17 compiler
- GCC 9+
- Clang 10+
- MSVC (via MSYS2 MinGW64)
- (Optional) Catch2 for testing

## 📝 License
MIT License © 2025

You’re free to use, modify, and distribute this library for personal or commercial projects.
