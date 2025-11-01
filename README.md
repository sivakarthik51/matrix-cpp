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

### 🔹 Linux / macOS

```bash
make test
```
This builds the test binary (tests/test_matrix) and runs all tests.

### 🔹 Windows (MSYS2 MINGW64)

```bash
pacman -S --needed make mingw-w64-x86_64-gcc
make test
```
The CI automatically sets up make and g++ if you use the provided GitHub Actions workflow.


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

## 🧱 Build System
MatrixLib uses a simple GNU Makefile with the following targets:

| Command      | Description               |
| ------------ | ------------------------- |
| `make`       | Builds the library        |
| `make test`  | Builds and runs all tests |
| `make clean` | Removes build artifacts   |

The Makefile works with:

* GCC (g++)
* Clang
* MSYS2 / MinGW on Windows

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
- make
- (Optional) Catch2 for testing

## 📝 License
MIT License © 2025

You’re free to use, modify, and distribute this library for personal or commercial projects.
