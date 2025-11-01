#include "matrix_cuda.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace matrixlib
{

    template <typename T>
    __global__ void addKernel(T *a, const T *b, size_t n)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n)
            a[idx] += b[idx];
    }

    template <typename T>
    MatrixCUDA<T>::MatrixCUDA(size_t r, size_t c) : rows(r), cols(c)
    {
        cudaMalloc(&device_data, rows * cols * sizeof(T));
    }

    template <typename T>
    MatrixCUDA<T>::~MatrixCUDA()
    {
        if (device_data)
            cudaFree(device_data);
    }

    template <typename T>
    void MatrixCUDA<T>::copyFromCPU(const T *host_data)
    {
        cudaMemcpy(device_data, host_data, rows * cols * sizeof(T), cudaMemcpyHostToDevice);
    }

    template <typename T>
    void MatrixCUDA<T>::copyToCPU(T *host_data) const
    {
        cudaMemcpy(host_data, device_data, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
    }

    template <typename T>
    void MatrixCUDA<T>::add(const MatrixCUDA &other)
    {
        size_t N = rows * cols;
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        addKernel<<<blocks, threads>>>(device_data, other.device_data, N);
        cudaDeviceSynchronize();
    }

    // Explicit instantiations
    template class MatrixCUDA<int>;
    template class MatrixCUDA<float>;
    template class MatrixCUDA<double>;

} // namespace matrixlib
