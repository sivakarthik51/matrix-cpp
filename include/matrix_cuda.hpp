#pragma once
#include <cstddef>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace matrixlib
{

    template <typename T>
    class MatrixCUDA
    {
    public:
        MatrixCUDA(size_t rows, size_t cols);
        ~MatrixCUDA();

        void copyFromCPU(const T *host_data);
        void copyToCPU(T *host_data) const;

        void add(const MatrixCUDA &other);
        void multiply(const MatrixCUDA &other);

    private:
        T *device_data = nullptr;
        size_t rows, cols;
    };

} // namespace matrixlib
