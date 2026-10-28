// A vector type, a sparse matrix and a Cholesky factor that live in C++,
// bound with nanobind.
//
// The vector is a std::vector<double> and nothing else: no buffer protocol,
// no __array__, no indexing, no arithmetic operators. Python can ask it to
// copy itself, scale itself, add a multiple of another vector, take a dot
// product (plain or weighted by a sparse matrix), fill itself with normal
// draws, and -- for plotting only -- hand its values out as a list. That is
// the whole surface the library needs, and the point of keeping it that
// small is that nothing above it can reach for a component by accident.
//
// The matrix is CSR, built from triplets, applied directly or transposed.
// Its Cholesky factor is banded, which is all a one-dimensional finite
// element matrix needs, and it offers the four solves a Gram matrix's factor
// is asked for: the full solve, the factor and its inverse, and the inverse
// transpose, which is what white noise with respect to the weighted inner
// product takes.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

// --------------------------------------------------------------------- //
//                                 Vector                                 //
// --------------------------------------------------------------------- //

class SparseMatrix;

class Vector {
  public:
    explicit Vector(std::size_t size) : data_(size, 0.0) {}
    explicit Vector(std::vector<double> values) : data_(std::move(values)) {}

    std::size_t size() const { return data_.size(); }

    Vector copy() const { return Vector(data_); }

    // self += a * x
    void axpy(double a, const Vector& x) {
        check_same_size(x);
        for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += a * x.data_[i];
    }

    // self *= a
    void scale(double a) {
        for (double& value : data_) value *= a;
    }

    double dot(const Vector& other) const {
        check_same_size(other);
        double sum = 0.0;
        for (std::size_t i = 0; i < data_.size(); ++i) sum += data_[i] * other.data_[i];
        return sum;
    }

    // self^T (G other)
    double dot_weighted(const Vector& other, const SparseMatrix& gram) const;

    // Independent standard normal entries from this vector's own generator.
    // Box-Muller on the engine's raw 64-bit words rather than
    // std::normal_distribution, whose algorithm the standard leaves to each
    // implementation: the same seed then gives the same draw with libstdc++,
    // libc++ and MSVC alike, which is what a seeded test relies on.
    void fill_random(std::uint64_t seed) {
        std::mt19937_64 engine(seed);
        auto uniform = [&engine]() {
            // 53 random bits, shifted into (0, 1]: never zero, so the
            // logarithm below is finite.
            return (static_cast<double>(engine() >> 11) + 1.0) * 0x1.0p-53;
        };
        constexpr double two_pi = 6.283185307179586;
        std::size_t i = 0;
        while (i < data_.size()) {
            const double radius = std::sqrt(-2.0 * std::log(uniform()));
            const double angle = two_pi * uniform();
            data_[i++] = radius * std::cos(angle);
            if (i < data_.size()) data_[i++] = radius * std::sin(angle);
        }
    }

    // For the plotting boundary only.
    std::vector<double> to_list() const { return data_; }
    static Vector from_list(const std::vector<double>& values) { return Vector(values); }

    const std::vector<double>& values() const { return data_; }
    std::vector<double>& values() { return data_; }

  private:
    std::vector<double> data_;

    void check_same_size(const Vector& other) const {
        if (other.data_.size() != data_.size())
            throw std::invalid_argument("Vectors have different sizes.");
    }
};

// --------------------------------------------------------------------- //
//                              SparseMatrix                             //
// --------------------------------------------------------------------- //

class CholeskyFactor;

class SparseMatrix {
  public:
    // From triplets; duplicate entries are summed, as in every CSR builder.
    SparseMatrix(std::size_t n_rows, std::size_t n_cols, const std::vector<int>& rows,
                 const std::vector<int>& cols, const std::vector<double>& values)
        : n_rows_(n_rows), n_cols_(n_cols) {
        if (rows.size() != cols.size() || rows.size() != values.size())
            throw std::invalid_argument("rows, cols and values must have one length.");
        std::vector<std::size_t> order(rows.size());
        for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return rows[a] != rows[b] ? rows[a] < rows[b] : cols[a] < cols[b];
        });
        row_start_.assign(n_rows + 1, 0);
        for (std::size_t k : order) {
            int i = rows[k], j = cols[k];
            if (i < 0 || j < 0 || std::size_t(i) >= n_rows || std::size_t(j) >= n_cols)
                throw std::invalid_argument("A triplet index is out of range.");
            if (!col_.empty() && row_of_last_ == i && col_.back() == j) {
                val_.back() += values[k];
            } else {
                col_.push_back(j);
                val_.push_back(values[k]);
                row_of_last_ = i;
                row_start_[std::size_t(i) + 1] += 1;
            }
        }
        for (std::size_t i = 0; i < n_rows; ++i) row_start_[i + 1] += row_start_[i];
    }

    std::size_t n_rows() const { return n_rows_; }
    std::size_t n_cols() const { return n_cols_; }
    std::size_t nnz() const { return val_.size(); }

    Vector apply(const Vector& x) const {
        if (x.size() != n_cols_) throw std::invalid_argument("apply: size mismatch.");
        Vector out(n_rows_);
        const auto& xv = x.values();
        auto& ov = out.values();
        for (std::size_t i = 0; i < n_rows_; ++i) {
            double sum = 0.0;
            for (std::size_t k = row_start_[i]; k < row_start_[i + 1]; ++k)
                sum += val_[k] * xv[std::size_t(col_[k])];
            ov[i] = sum;
        }
        return out;
    }

    Vector apply_transpose(const Vector& x) const {
        if (x.size() != n_rows_) throw std::invalid_argument("apply_transpose: size mismatch.");
        Vector out(n_cols_);
        const auto& xv = x.values();
        auto& ov = out.values();
        for (std::size_t i = 0; i < n_rows_; ++i)
            for (std::size_t k = row_start_[i]; k < row_start_[i + 1]; ++k)
                ov[std::size_t(col_[k])] += val_[k] * xv[i];
        return out;
    }

    double entry(std::size_t i, std::size_t j) const {
        for (std::size_t k = row_start_[i]; k < row_start_[i + 1]; ++k)
            if (std::size_t(col_[k]) == j) return val_[k];
        return 0.0;
    }

    // The largest |i - j| over the stored entries.
    std::size_t bandwidth() const {
        std::size_t band = 0;
        for (std::size_t i = 0; i < n_rows_; ++i)
            for (std::size_t k = row_start_[i]; k < row_start_[i + 1]; ++k) {
                std::size_t j = std::size_t(col_[k]);
                band = std::max(band, i > j ? i - j : j - i);
            }
        return band;
    }

    CholeskyFactor cholesky() const;

  private:
    std::size_t n_rows_, n_cols_;
    std::vector<std::size_t> row_start_;
    std::vector<int> col_;
    std::vector<double> val_;
    int row_of_last_ = -1;
};

double Vector::dot_weighted(const Vector& other, const SparseMatrix& gram) const {
    return dot(gram.apply(other));
}

// --------------------------------------------------------------------- //
//                             CholeskyFactor                            //
// --------------------------------------------------------------------- //

// A = L L^T with L lower triangular of bandwidth p, stored by rows: entry
// (i, j) with i - p <= j <= i sits at band_[i * (p + 1) + (j - i + p)].
class CholeskyFactor {
  public:
    explicit CholeskyFactor(const SparseMatrix& matrix)
        : n_(matrix.n_rows()), p_(matrix.bandwidth()), band_(n_ * (p_ + 1), 0.0) {
        if (matrix.n_rows() != matrix.n_cols())
            throw std::invalid_argument("Only a square matrix has a Cholesky factor.");
        for (std::size_t i = 0; i < n_; ++i) {
            std::size_t j0 = i > p_ ? i - p_ : 0;
            for (std::size_t j = j0; j <= i; ++j) {
                double sum = matrix.entry(i, j);
                std::size_t k0 = std::max(j0, j > p_ ? j - p_ : 0);
                for (std::size_t k = k0; k < j; ++k) sum -= at(i, k) * at(j, k);
                if (i == j) {
                    if (sum <= 0.0)
                        throw std::runtime_error("The matrix is not positive definite.");
                    at(i, i) = std::sqrt(sum);
                } else {
                    at(i, j) = sum / at(j, j);
                }
            }
        }
    }

    std::size_t size() const { return n_; }
    std::size_t bandwidth() const { return p_; }

    // L^-1 b, by forward substitution.
    Vector solve_factor(const Vector& b) const {
        check(b);
        Vector y = b.copy();
        auto& yv = y.values();
        for (std::size_t i = 0; i < n_; ++i) {
            std::size_t j0 = i > p_ ? i - p_ : 0;
            for (std::size_t j = j0; j < i; ++j) yv[i] -= at(i, j) * yv[j];
            yv[i] /= at(i, i);
        }
        return y;
    }

    // L^-T y, by back substitution.
    Vector solve_factor_transpose(const Vector& y) const {
        check(y);
        Vector x = y.copy();
        auto& xv = x.values();
        for (std::size_t i = n_; i-- > 0;) {
            std::size_t j1 = std::min(n_ - 1, i + p_);
            for (std::size_t j = i + 1; j <= j1; ++j) xv[i] -= at(j, i) * xv[j];
            xv[i] /= at(i, i);
        }
        return x;
    }

    // (L L^T)^-1 b
    Vector solve(const Vector& b) const { return solve_factor_transpose(solve_factor(b)); }

    // L z
    Vector apply_factor(const Vector& z) const {
        check(z);
        Vector out(n_);
        const auto& zv = z.values();
        auto& ov = out.values();
        for (std::size_t i = 0; i < n_; ++i) {
            std::size_t j0 = i > p_ ? i - p_ : 0;
            double sum = 0.0;
            for (std::size_t j = j0; j <= i; ++j) sum += at(i, j) * zv[j];
            ov[i] = sum;
        }
        return out;
    }

    // log det A = 2 sum_i log L_ii
    double log_determinant() const {
        double sum = 0.0;
        for (std::size_t i = 0; i < n_; ++i) sum += std::log(at(i, i));
        return 2.0 * sum;
    }

  private:
    std::size_t n_, p_;
    std::vector<double> band_;

    double& at(std::size_t i, std::size_t j) { return band_[i * (p_ + 1) + (j + p_ - i)]; }
    double at(std::size_t i, std::size_t j) const { return band_[i * (p_ + 1) + (j + p_ - i)]; }

    void check(const Vector& v) const {
        if (v.size() != n_) throw std::invalid_argument("Cholesky solve: size mismatch.");
    }
};

CholeskyFactor SparseMatrix::cholesky() const { return CholeskyFactor(*this); }

// --------------------------------------------------------------------- //
//                              The module                               //
// --------------------------------------------------------------------- //

NB_MODULE(cppvec, m) {
    m.doc() = "A vector, a sparse matrix and a banded Cholesky factor, all in C++.";

    nb::class_<Vector>(m, "Vector")
        .def(nb::init<std::size_t>(), nb::arg("size"))
        .def("size", &Vector::size)
        .def("copy", &Vector::copy)
        .def("axpy", &Vector::axpy, nb::arg("a"), nb::arg("x"), "self += a * x, in place.")
        .def("scale", &Vector::scale, nb::arg("a"), "self *= a, in place.")
        .def("dot", &Vector::dot, nb::arg("other"))
        .def("dot_weighted", &Vector::dot_weighted, nb::arg("other"), nb::arg("gram"),
             "self^T (gram other).")
        .def("fill_random", &Vector::fill_random, nb::arg("seed"),
             "Independent standard normal entries from the vector's own generator.")
        .def("to_list", &Vector::to_list, "The entries as a list; for plotting only.")
        .def_static("from_list", &Vector::from_list, nb::arg("values"))
        .def("__repr__", [](const Vector& v) {
            return "<cppvec.Vector of size " + std::to_string(v.size()) + ">";
        });

    nb::class_<SparseMatrix>(m, "SparseMatrix")
        .def(nb::init<std::size_t, std::size_t, const std::vector<int>&, const std::vector<int>&,
                      const std::vector<double>&>(),
             nb::arg("n_rows"), nb::arg("n_cols"), nb::arg("rows"), nb::arg("cols"),
             nb::arg("values"), "CSR from triplets; duplicates are summed.")
        .def_prop_ro("n_rows", &SparseMatrix::n_rows)
        .def_prop_ro("n_cols", &SparseMatrix::n_cols)
        .def_prop_ro("nnz", &SparseMatrix::nnz)
        .def("apply", &SparseMatrix::apply, nb::arg("x"))
        .def("apply_transpose", &SparseMatrix::apply_transpose, nb::arg("x"))
        .def("bandwidth", &SparseMatrix::bandwidth)
        .def("cholesky", &SparseMatrix::cholesky, "The banded Cholesky factor.");

    nb::class_<CholeskyFactor>(m, "CholeskyFactor")
        .def("size", &CholeskyFactor::size)
        .def("bandwidth", &CholeskyFactor::bandwidth)
        .def("solve", &CholeskyFactor::solve, nb::arg("b"), "(L L^T)^-1 b")
        .def("apply_factor", &CholeskyFactor::apply_factor, nb::arg("z"), "L z")
        .def("solve_factor", &CholeskyFactor::solve_factor, nb::arg("b"), "L^-1 b")
        .def("solve_factor_transpose", &CholeskyFactor::solve_factor_transpose, nb::arg("y"),
             "L^-T y")
        .def("log_determinant", &CholeskyFactor::log_determinant);
}
