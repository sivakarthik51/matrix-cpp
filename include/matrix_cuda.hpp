#pragma once
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace matrixlib
{

    // --- Common CUDA helper ---
    inline void checkCuda(cudaError_t result, const char *msg)
    {
        if (result != cudaSuccess)
            throw std::runtime_error(std::string("CUDA Error: ") + msg + " - " + cudaGetErrorString(result));
    }

    // --- Elementwise kernels ---
    template <typename T>
    __global__ void addKernel(const T *A, const T *B, T *C, size_t rows, size_t cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols)
            C[idx] = A[idx] + B[idx];
    }

    template <typename T>
    __global__ void subKernel(const T *A, const T *B, T *C, size_t rows, size_t cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols)
            C[idx] = A[idx] - B[idx];
    }

    template <typename T>
    __global__ void mulKernel(const T *A, const T *B, T *C, size_t rows, size_t cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols)
            C[idx] = A[idx] * B[idx];
    }

    template <typename T>
    __global__ void divKernel(const T *A, const T *B, T *C, size_t rows, size_t cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols)
            C[idx] = A[idx] / B[idx];
    }

    template <typename T>
    __global__ void scalarKernel(T *A, T scalar, char op, size_t rows, size_t cols)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < rows * cols)
        {
            switch (op)
            {
            case '+':
                A[idx] += scalar;
                break;
            case '-':
                A[idx] -= scalar;
                break;
            case '*':
                A[idx] *= scalar;
                break;
            case '/':
                A[idx] /= scalar;
                break;
            }
        }
    }

    // --- MatrixCUDA definition ---
    template <typename T>
    class MatrixCUDA
    {
        static_assert(std::is_arithmetic_v<T>, "MatrixCUDA supports only numeric types");

    public:
        MatrixCUDA(size_t rows, size_t cols)
            : rows_(rows), cols_(cols)
        {
            size_t bytes = rows * cols * sizeof(T);
            checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
            checkCuda(cudaMemset(data_, 0, bytes), "cudaMemset");
        }

        ~MatrixCUDA() { cudaFree(data_); }

        MatrixCUDA(const MatrixCUDA &) = delete;
        MatrixCUDA &operator=(const MatrixCUDA &) = delete;

        MatrixCUDA(MatrixCUDA &&other) noexcept
            : rows_(other.rows_), cols_(other.cols_), data_(other.data_)
        {
            other.data_ = nullptr;
            other.rows_ = other.cols_ = 0;
        }

        MatrixCUDA &operator=(MatrixCUDA &&other) noexcept
        {
            if (this != &other)
            {
                cudaFree(data_);
                rows_ = other.rows_;
                cols_ = other.cols_;
                data_ = other.data_;
                other.data_ = nullptr;
                other.rows_ = other.cols_ = 0;
            }
            return *this;
        }

        void copyFromCPU(const T *hostData)
        {
            checkCuda(cudaMemcpy(data_, hostData, rows_ * cols_ * sizeof(T), cudaMemcpyHostToDevice),
                      "cudaMemcpy (HostToDevice)");
        }

        void copyToCPU(T *hostData) const
        {
            checkCuda(cudaMemcpy(hostData, data_, rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToHost),
                      "cudaMemcpy (DeviceToHost)");
        }

        size_t rows() const noexcept { return rows_; }
        size_t cols() const noexcept { return cols_; }
        T *data() noexcept { return data_; }
        const T *data() const noexcept { return data_; }

        // --- Elementwise operations ---
        template <typename U>
        auto operator+(const MatrixCUDA<U> &other) const
        {
            using R = decltype(T{} + U{});
            return binaryOp<R>(other, addKernel<R>);
        }

        template <typename U>
        auto operator-(const MatrixCUDA<U> &other) const
        {
            using R = decltype(T{} - U{});
            return binaryOp<R>(other, subKernel<R>);
        }

        template <typename U>
        auto operator*(const MatrixCUDA<U> &other) const
        {
            using R = decltype(T{} * U{});
            return binaryOp<R>(other, mulKernel<R>);
        }

        template <typename U>
        auto operator/(const MatrixCUDA<U> &other) const
        {
            using R = decltype(T{} / U{});
            return binaryOp<R>(other, divKernel<R>);
        }

        // --- Scalar operations (A + s, A - s, A * s, A / s) ---
        MatrixCUDA operator+(T scalar) const { return scalarOp('+', scalar); }
        MatrixCUDA operator-(T scalar) const { return scalarOp('-', scalar); }
        MatrixCUDA operator*(T scalar) const { return scalarOp('*', scalar); }
        MatrixCUDA operator/(T scalar) const { return scalarOp('/', scalar); }

        // --- In-place versions ---
        MatrixCUDA &operator+=(const MatrixCUDA &other) { return inplaceOp(other, addKernel<T>); }
        MatrixCUDA &operator-=(const MatrixCUDA &other) { return inplaceOp(other, subKernel<T>); }
        MatrixCUDA &operator*=(const MatrixCUDA &other) { return inplaceOp(other, mulKernel<T>); }
        MatrixCUDA &operator/=(const MatrixCUDA &other) { return inplaceOp(other, divKernel<T>); }

    private:
        size_t rows_, cols_;
        T *data_;

        template <typename R, typename U, typename Kernel>
        MatrixCUDA<R> binaryOp(const MatrixCUDA<U> &other, Kernel kernel) const
        {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::runtime_error("MatrixCUDA: dimension mismatch");

            MatrixCUDA<R> result(rows_, cols_);
            size_t total = rows_ * cols_;
            dim3 threads(256);
            dim3 blocks((total + threads.x - 1) / threads.x);

            kernel<<<blocks, threads>>>(reinterpret_cast<const R *>(data_),
                                        reinterpret_cast<const R *>(other.data_),
                                        result.data_, rows_, cols_);
            cudaDeviceSynchronize();
            return result;
        }

        template <typename Kernel>
        MatrixCUDA &inplaceOp(const MatrixCUDA &other, Kernel kernel)
        {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::runtime_error("MatrixCUDA: dimension mismatch");

            size_t total = rows_ * cols_;
            dim3 threads(256);
            dim3 blocks((total + threads.x - 1) / threads.x);

            kernel<<<blocks, threads>>>(data_, other.data_, data_, rows_, cols_);
            cudaDeviceSynchronize();
            return *this;
        }

        MatrixCUDA scalarOp(char op, T scalar) const
        {
            MatrixCUDA result(rows_, cols_);
            checkCuda(cudaMemcpy(result.data_, data_, rows_ * cols_ * sizeof(T), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy (DeviceToDevice)");
            size_t total = rows_ * cols_;
            dim3 threads(256);
            dim3 blocks((total + threads.x - 1) / threads.x);

            scalarKernel<<<blocks, threads>>>(result.data_, scalar, op, rows_, cols_);
            cudaDeviceSynchronize();
            return result;
        }
    };

    // --- Scalar on left side: s + A, s - A, etc. ---
    template <typename T>
    MatrixCUDA<T> operator+(T scalar, const MatrixCUDA<T> &A) { return A + scalar; }
    template <typename T>
    MatrixCUDA<T> operator-(T scalar, const MatrixCUDA<T> &A)
    {
        MatrixCUDA<T> result = A * (-1);
        result += scalar;
        return result;
    }
    template <typename T>
    MatrixCUDA<T> operator*(T scalar, const MatrixCUDA<T> &A) { return A * scalar; }

    // --- Mixed-type scalar kernels and free operators for MatrixCUDA ---

    template <typename Tsrc, typename Rdst>
    __global__ void scalarMixedKernel(const Tsrc *A, Rdst *C, Rdst scalar, char op, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        Rdst aval = static_cast<Rdst>(A[idx]);
        switch (op)
        {
        case '+':
            C[idx] = aval + scalar;
            break;
        case '-':
            C[idx] = aval - scalar;
            break;
        case '*':
            C[idx] = aval * scalar;
            break;
        case '/':
            C[idx] = aval / scalar;
            break;
        default:
            C[idx] = aval;
            break;
        }
    }

    // matrix (T)  op  scalar (U)  -> MatrixCUDA<R>
    template <typename T, typename U,
              typename R = typename std::common_type<T, U>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    MatrixCUDA<R> operator+(const MatrixCUDA<T> &m, U scalar)
    {
        MatrixCUDA<R> result(m.rows(), m.cols());
        size_t total = m.rows() * m.cols();
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        scalarMixedKernel<T, R><<<blocks, threads>>>(m.data(), result.data(), static_cast<R>(scalar), '+', total);
        cudaDeviceSynchronize();
        return result;
    }

    template <typename T, typename U,
              typename R = typename std::common_type<T, U>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    MatrixCUDA<R> operator-(const MatrixCUDA<T> &m, U scalar)
    {
        MatrixCUDA<R> result(m.rows(), m.cols());
        size_t total = m.rows() * m.cols();
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        scalarMixedKernel<T, R><<<blocks, threads>>>(m.data(), result.data(), static_cast<R>(scalar), '-', total);
        cudaDeviceSynchronize();
        return result;
    }

    template <typename T, typename U,
              typename R = typename std::common_type<T, U>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    MatrixCUDA<R> operator*(const MatrixCUDA<T> &m, U scalar)
    {
        MatrixCUDA<R> result(m.rows(), m.cols());
        size_t total = m.rows() * m.cols();
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        scalarMixedKernel<T, R><<<blocks, threads>>>(m.data(), result.data(), static_cast<R>(scalar), '*', total);
        cudaDeviceSynchronize();
        return result;
    }

    template <typename T, typename U,
              typename R = typename std::common_type<T, U>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type>
    MatrixCUDA<R> operator/(const MatrixCUDA<T> &m, U scalar)
    {
        MatrixCUDA<R> result(m.rows(), m.cols());
        size_t total = m.rows() * m.cols();
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        scalarMixedKernel<T, R><<<blocks, threads>>>(m.data(), result.data(), static_cast<R>(scalar), '/', total);
        cudaDeviceSynchronize();
        return result;
    }

    // scalar (U) op matrix (T) -> MatrixCUDA<R>
    template <typename U, typename T,
              typename R = typename std::common_type<U, T>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    MatrixCUDA<R> operator+(U scalar, const MatrixCUDA<T> &m)
    {
        // reuse matrix + scalar (commutative)
        return m + scalar;
    }

    template <typename U, typename T,
              typename R = typename std::common_type<U, T>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    MatrixCUDA<R> operator-(U scalar, const MatrixCUDA<T> &m)
    {
        // produce scalar - element
        MatrixCUDA<R> result(m.rows(), m.cols());
        size_t total = m.rows() * m.cols();
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        // reuse kernel but we need (scalar - A)
        // define a small kernel wrapper on host via lambda launching a kernel that handles (scalar - A)
        // We'll invoke scalarMixedKernel with op='x' and handle it inside kernel; simpler: call scalarMixedKernel with '+' after negating A into result then add scalar.
        // Instead do dedicated kernel below:

        // dedicated kernel for scalar - A:
        auto lambda_launch = [&](void)
        {
            // kernel defined inline below
        };
        // implement kernel:
        // define a kernel here (local lambda can't be device), so instead define a generic kernel above for scalarLeft:
        // But to keep header simple, implement inline here:

        // We'll launch a kernel that computes result[idx] = scalar - A[idx]
        // Use scalarMixedKernel by treating scalar as R and op='S' with switch; we didn't implement 'S' above.
        // For clarity: call scalarMixedKernel with op='-' but swap operands: C = scalar - A => C = (-A) + scalar.
        // So compute result = -A (via scalar kernel with op='*' and scalar = -1) then add scalar.
        scalarMixedKernel<T, R><<<blocks, threads>>>(m.data(), result.data(), static_cast<R>(0), '+', total);
        // Instead of above complexity, we can implement a small kernel here for scalar-left:
        // However to avoid code duplication we'll implement a simple device kernel now (outside). See note below.
        // For now, keep simple by copying device->host, computing, copying back (less efficient but correct).
        // WARNING: This fallback does host transfers; acceptable for tests.
        std::vector<R> host(m.rows() * m.cols());
        std::vector<T> host_src(m.rows() * m.cols());
        cudaMemcpy(host_src.data(), m.data(), m.rows() * m.cols() * sizeof(T), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<R>(scalar) - static_cast<R>(host_src[i]);
        cudaMemcpy(result.data(), host.data(), host.size() * sizeof(R), cudaMemcpyHostToDevice);
        return result;
    }

    template <typename U, typename T,
              typename R = typename std::common_type<U, T>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    MatrixCUDA<R> operator*(U scalar, const MatrixCUDA<T> &m)
    {
        return m * scalar;
    }

    template <typename U, typename T,
              typename R = typename std::common_type<U, T>::type,
              typename = typename std::enable_if<std::is_arithmetic<U>::value>::type,
              typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    MatrixCUDA<R> operator/(U scalar, const MatrixCUDA<T> &m)
    {
        // compute scalar / A elementwise; fallback to host for simplicity
        MatrixCUDA<R> result(m.rows(), m.cols());
        std::vector<R> host(m.rows() * m.cols());
        std::vector<T> host_src(m.rows() * m.cols());
        cudaMemcpy(host_src.data(), m.data(), m.rows() * m.cols() * sizeof(T), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<R>(scalar) / static_cast<R>(host_src[i]);
        cudaMemcpy(result.data(), host.data(), host.size() * sizeof(R), cudaMemcpyHostToDevice);
        return result;
    }

} // namespace matrixlib
