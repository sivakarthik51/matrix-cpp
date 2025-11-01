#pragma once
#include <vector>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <iostream>

namespace matrixlib
{

    // Helper alias for restricting to arithmetic types
    template <typename T>
    using EnableIfArithmetic = typename std::enable_if<std::is_arithmetic<T>::value>::type;

    // ================= Matrix Class =================
    template <typename T, typename = EnableIfArithmetic<T>>
    class Matrix
    {
    public:
        Matrix(std::size_t rows, std::size_t cols)
            : rows_(rows), cols_(cols), data_(rows * cols) {}

        Matrix(std::size_t rows, std::size_t cols, const T &value)
            : rows_(rows), cols_(cols), data_(rows * cols, value) {}

        Matrix(std::initializer_list<std::initializer_list<T>> values)
        {
            rows_ = values.size();
            cols_ = values.begin()->size();
            data_.reserve(rows_ * cols_);
            for (const auto &row : values)
            {
                if (row.size() != cols_)
                    throw std::runtime_error("Inconsistent row size");
                data_.insert(data_.end(), row.begin(), row.end());
            }
        }

        // Element access
        T &operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
        const T &operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

        std::size_t rows() const { return rows_; }
        std::size_t cols() const { return cols_; }
        const std::vector<T> &data() const { return data_; }

        // ===== Matrix-Matrix (same type) =====
        Matrix operator+(const Matrix &other) const
        {
            check_same_size(other);
            Matrix result(rows_, cols_);
            for (std::size_t i = 0; i < data_.size(); ++i)
                result.data_[i] = data_[i] + other.data_[i];
            return result;
        }

        Matrix operator-(const Matrix &other) const
        {
            check_same_size(other);
            Matrix result(rows_, cols_);
            for (std::size_t i = 0; i < data_.size(); ++i)
                result.data_[i] = data_[i] - other.data_[i];
            return result;
        }

        Matrix operator*(const Matrix &other) const
        {
            if (cols_ != other.rows_)
                throw std::runtime_error("Matrix dimension mismatch");
            Matrix result(rows_, other.cols_, T{});
            for (std::size_t i = 0; i < rows_; ++i)
                for (std::size_t j = 0; j < other.cols_; ++j)
                    for (std::size_t k = 0; k < cols_; ++k)
                        result(i, j) += (*this)(i, k) * other(k, j);
            return result;
        }

        // ===== Matrix-Scalar (same type) =====
        Matrix operator+(const T &scalar) const
        {
            return apply_scalar([&](T x)
                                { return x + scalar; });
        }
        Matrix operator-(const T &scalar) const
        {
            return apply_scalar([&](T x)
                                { return x - scalar; });
        }
        Matrix operator*(const T &scalar) const
        {
            return apply_scalar([&](T x)
                                { return x * scalar; });
        }
        Matrix operator/(const T &scalar) const
        {
            return apply_scalar([&](T x)
                                { return x / scalar; });
        }

        Matrix &operator+=(const T &scalar)
        {
            for (auto &x : data_)
                x += scalar;
            return *this;
        }
        Matrix &operator-=(const T &scalar)
        {
            for (auto &x : data_)
                x -= scalar;
            return *this;
        }
        Matrix &operator*=(const T &scalar)
        {
            for (auto &x : data_)
                x *= scalar;
            return *this;
        }
        Matrix &operator/=(const T &scalar)
        {
            for (auto &x : data_)
                x /= scalar;
            return *this;
        }

    private:
        std::size_t rows_{}, cols_{};
        std::vector<T> data_;

        void check_same_size(const Matrix &other) const
        {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::runtime_error("Matrix size mismatch");
        }

        template <typename F>
        Matrix apply_scalar(F func) const
        {
            Matrix result(rows_, cols_);
            for (std::size_t i = 0; i < data_.size(); ++i)
                result.data_[i] = func(data_[i]);
            return result;
        }
    };

    // ================= Type-Promotion Overloads =================

    // ---- Matrix + Matrix (mixed types) ----
    template <
        typename T1, typename T2,
        typename R = typename std::common_type<T1, T2>::type,
        typename = EnableIfArithmetic<T1>,
        typename = EnableIfArithmetic<T2>>
    Matrix<R> operator+(const Matrix<T1> &a, const Matrix<T2> &b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("Matrix size mismatch");
        Matrix<R> result(a.rows(), a.cols());
        for (std::size_t i = 0; i < a.data().size(); ++i)
            result(i / a.cols(), i % a.cols()) =
                static_cast<R>(a.data()[i]) + static_cast<R>(b.data()[i]);
        return result;
    }

    // ---- Matrix - Matrix (mixed types) ----
    template <
        typename T1, typename T2,
        typename R = typename std::common_type<T1, T2>::type,
        typename = EnableIfArithmetic<T1>,
        typename = EnableIfArithmetic<T2>>
    Matrix<R> operator-(const Matrix<T1> &a, const Matrix<T2> &b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
            throw std::runtime_error("Matrix size mismatch");
        Matrix<R> result(a.rows(), a.cols());
        for (std::size_t i = 0; i < a.data().size(); ++i)
            result(i / a.cols(), i % a.cols()) =
                static_cast<R>(a.data()[i]) - static_cast<R>(b.data()[i]);
        return result;
    }

    // ---- Matrix + Scalar (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator+(const Matrix<T> &mat, const U &scalar)
    {
        Matrix<R> result(mat.rows(), mat.cols());
        for (std::size_t i = 0; i < mat.data().size(); ++i)
            result(i / mat.cols(), i % mat.cols()) = static_cast<R>(mat.data()[i]) + static_cast<R>(scalar);
        return result;
    }

    // ---- Scalar + Matrix (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator+(const T &scalar, const Matrix<U> &mat)
    {
        return mat + scalar;
    }

    // ---- Matrix - Scalar (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator-(const Matrix<T> &mat, const U &scalar)
    {
        Matrix<R> result(mat.rows(), mat.cols());
        for (std::size_t i = 0; i < mat.data().size(); ++i)
            result(i / mat.cols(), i % mat.cols()) = static_cast<R>(mat.data()[i]) - static_cast<R>(scalar);
        return result;
    }

    // ---- Scalar - Matrix (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator-(const T &scalar, const Matrix<U> &mat)
    {
        Matrix<R> result(mat.rows(), mat.cols());
        for (std::size_t i = 0; i < mat.data().size(); ++i)
            result(i / mat.cols(), i % mat.cols()) = static_cast<R>(scalar) - static_cast<R>(mat.data()[i]);
        return result;
    }

    // ---- Matrix * Scalar (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator*(const Matrix<T> &mat, const U &scalar)
    {
        Matrix<R> result(mat.rows(), mat.cols());
        for (std::size_t i = 0; i < mat.data().size(); ++i)
            result(i / mat.cols(), i % mat.cols()) = static_cast<R>(mat.data()[i]) * static_cast<R>(scalar);
        return result;
    }

    // ---- Scalar * Matrix (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator*(const T &scalar, const Matrix<U> &mat)
    {
        return mat * scalar;
    }

    // ---- Matrix / Scalar (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator/(const Matrix<T> &mat, const U &scalar)
    {
        Matrix<R> result(mat.rows(), mat.cols());
        for (std::size_t i = 0; i < mat.data().size(); ++i)
            result(i / mat.cols(), i % mat.cols()) = static_cast<R>(mat.data()[i]) / static_cast<R>(scalar);
        return result;
    }

    // ---- Scalar / Matrix (mixed types) ----
    template <
        typename T, typename U,
        typename R = typename std::common_type<T, U>::type,
        typename = EnableIfArithmetic<T>,
        typename = EnableIfArithmetic<U>>
    Matrix<R> operator/(const T &scalar, const Matrix<U> &mat)
    {
        Matrix<R> result(mat.rows(), mat.cols());
        for (std::size_t i = 0; i < mat.data().size(); ++i)
            result(i / mat.cols(), i % mat.cols()) = static_cast<R>(scalar) / static_cast<R>(mat.data()[i]);
        return result;
    }

    // ---- Matrix * Matrix (mixed types) ----
    template <
        typename T1, typename T2,
        typename R = typename std::common_type<T1, T2>::type,
        typename = EnableIfArithmetic<T1>,
        typename = EnableIfArithmetic<T2>>
    Matrix<R> operator*(const Matrix<T1> &a, const Matrix<T2> &b)
    {
        if (a.cols() != b.rows())
            throw std::runtime_error("Matrix dimension mismatch");

        Matrix<R> result(a.rows(), b.cols(), R{});
        for (std::size_t i = 0; i < a.rows(); ++i)
        {
            for (std::size_t j = 0; j < b.cols(); ++j)
            {
                R sum = R{};
                for (std::size_t k = 0; k < a.cols(); ++k)
                    sum += static_cast<R>(a(i, k)) * static_cast<R>(b(k, j));
                result(i, j) = sum;
            }
        }
        return result;
    }

    // ---- Stream output ----
    template <typename T, typename = EnableIfArithmetic<T>>
    std::ostream &operator<<(std::ostream &os, const Matrix<T> &mat)
    {
        os << "[";
        for (std::size_t i = 0; i < mat.rows(); ++i)
        {
            os << "[";
            for (std::size_t j = 0; j < mat.cols(); ++j)
            {
                os << mat(i, j);
                if (j + 1 < mat.cols())
                    os << ", ";
            }
            os << "]";
            if (i + 1 < mat.rows())
                os << "\n ";
        }
        os << "]";
        return os;
    }

} // namespace matrixlib
