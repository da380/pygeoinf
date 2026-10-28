// A two-dimensional acoustic wave solver, its linearization and its adjoints,
// bound with nanobind, serial or split across MPI ranks.
//
// The unknown is m = 1/c^2, the slowness squared, on an nx-by-nz grid of
// spacing h. Axis 0 is x, axis 1 is depth z, and z runs fastest in memory, so
// a model is a C-ordered (nx, nz) array. The scheme is second order in time and
// fourth order in space: from rest, u^0 = u^{-1} = 0, for n = 0 .. N-1
//
//     u^{n+1} = tau (2 u^n - tau u^{n-1} + (dt^2 / m) (L u^n + s^n))
//
// with L the fourth-order five-point Laplacian along each axis, s^n the source
// at step n and tau the Cerjan taper: one inside, exp(-(rate (width - d))^2)
// at d cells from the left, right and bottom edges. The taper damps both time
// levels, as Cerjan's sponge does: the plain leapfrog step with only its new
// level tapered damps the field but not its rate of change, and on this grid
// that sponge sends back about 40% of the data's rms where this one sends
// back 5%. The top row is a free surface: u is held at zero there (the
// Laplacian's row 0 is zero) and the stencil reaches above it through the
// image u(-z) = -u(z). Beyond the other three edges u is taken as zero, which
// the taper makes harmless.
//
// Written before dividing by m, step n is the equation
//
//     (m / dt^2) (u^{n+1} / tau - 2 u^n + tau u^{n-1}) - L u^n = s^n,
//
// so the whole run is a linear system A(m) U = s, lower block triangular in
// time and affine in m. Its derivative in m along dm is the field whose
// step-n block is dm * a^n, with a^n = (u^{n+1} / tau - 2 u^n + tau u^{n-1}) / dt^2
// the acceleration of the field at step n, and the second derivative of A is
// zero. Everything an adjoint method needs is then four recursions:
//
//   forward      A U = s for a point source;
//   born         A dU = -dm * a(U), the linearized field;
//   adjoint      A^T L = S^T r, backward in time, the residual r injected at
//                the receivers, with the transposed Laplacian:
//                l^{k-1} = tau (2 l^k - tau l^{k+1} + (dt^2 / m) (L^T l^k + q^k));
//   adjoint_born A^T M = (dA[dm])^T L, the second adjoint, whose source is
//                dm times the transposed acceleration of L,
//                (l^{k-1} / tau - 2 l^k + tau l^{k+1}) / dt^2: each taper
//                factor sits on the other side of the step, where the
//                transpose puts it;
//
// and one correlation, correlate(A, B) = -sum_n a(A)^n * B^n. The gradient
// and the curvature term of the Hessian are those correlations of these
// fields; `gradient` and `curvature` compute them in one backward pass each,
// without storing the adjoint fields, and `second_born` gives F''(m)[dm, dm2]
// by running the two linearized fields and the second-order one in lockstep.
//
// The Laplacian is not symmetric: its row 0 is zero, while rows 1 and 2
// reach row 0 of the field. Its transpose is therefore written out, twice:
// as the gather the steps use, which differs from L's on rows 0, 1 and 2,
// and as the literal scatter of L's terms, which `laplacian(u, True,
// reference=True)` applies so the two can be compared.
//
// Fields. A field is all N steps of the grid, U[n] = u^{n+1}, and it stays in
// C++: a run returns a handle, an integer, which the other kernels take and
// `release` frees. The receiver data are (N, n_receivers), D[n] = S U[n].
//
// Ranks. Built with WAVEFORM_MPI, the grid's columns are split into one
// contiguous block per rank, each block's fields live on its rank, and every
// step exchanges two columns of halo with the neighbours, which is what the
// stencil reaches. Rank 0 is the one Python drives: each call broadcasts a
// command to the other ranks, which wait in `serve()`, scatters the model by
// columns, broadcasts the traces, and gathers the data and the gradients
// back. Every kernel is written once, for the block a rank owns; the serial
// build is the same code with one block. The kernels release the
// interpreter lock while they run; serially the shots of a survey can then be
// run on threads, while across ranks the calls are serialized.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#ifdef WAVEFORM_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

using Array = nb::ndarray<nb::numpy, double>;
using Grid = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

namespace {

// A heap buffer handed to NumPy, which frees it.
Array owned(double* data, std::initializer_list<std::size_t> shape) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return Array(data, shape, owner);
}

std::string shape_of(const Grid& a) {
    std::string out = "(";
    for (std::size_t i = 0; i < a.ndim(); ++i) {
        if (i) out += ", ";
        out += std::to_string(a.shape(i));
    }
    return out + (a.ndim() == 1 ? ",)" : ")");
}

}  // namespace

// The two builds name their classes apart, so that both modules can be
// imported into one process: nanobind registers a C++ type once.
#ifdef WAVEFORM_MPI
#define WAVEFORM_NAMESPACE waveform_mpi
#else
#define WAVEFORM_NAMESPACE waveform_serial
#endif

namespace WAVEFORM_NAMESPACE {

// --------------------------------------------------------------------- //
//                              The ranks                                //
// --------------------------------------------------------------------- //

// The split of the columns over the ranks, and the few collectives the
// kernels need. With one rank every operation is a copy or nothing.
class Ranks {
  public:
    Ranks(int nx, int nz) : nx_(nx), nz_(nz) {
#ifdef WAVEFORM_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
#endif
        start_.resize(std::size_t(size_) + 1);
        for (int r = 0; r <= size_; ++r) start_[r] = int((long(nx) * r) / size_);
        for (int r = 0; r < size_; ++r)
            if (start_[r + 1] - start_[r] < 2)
                throw std::invalid_argument("Each rank needs at least two columns.");
    }

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool root() const { return rank_ == 0; }
    int x0() const { return start_[rank_]; }
    int columns() const { return start_[rank_ + 1] - start_[rank_]; }
    std::size_t cells() const { return std::size_t(columns()) * nz_; }
    const std::vector<int>& starts() const { return start_; }

    // Fill the two halo columns on each side of a buffer of columns() + 4
    // columns from the neighbours' owned columns. The halos beyond the grid's
    // edges are never written and stay zero.
    void exchange(double* buffer) const {
#ifdef WAVEFORM_MPI
        if (size_ == 1) return;
        const int count = 2 * nz_;
        const int n = columns();
        const int left = rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL;
        const int right = rank_ + 1 < size_ ? rank_ + 1 : MPI_PROC_NULL;
        // My last two owned columns to the right; the left's into my left halo.
        MPI_Sendrecv(buffer + std::size_t(n) * nz_, count, MPI_DOUBLE, right, 0, buffer, count,
                     MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // My first two owned columns to the left; the right's into my right halo.
        MPI_Sendrecv(buffer + 2 * std::size_t(nz_), count, MPI_DOUBLE, left, 1,
                     buffer + std::size_t(n + 2) * nz_, count, MPI_DOUBLE, right, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
        (void)buffer;
#endif
    }

    // Each rank's block of columns of a grid that rank 0 holds whole.
    void scatter(const double* global, double* local) const {
#ifdef WAVEFORM_MPI
        if (size_ > 1) {
            std::vector<int> counts, offsets;
            layout(counts, offsets);
            MPI_Scatterv(global, counts.data(), offsets.data(), MPI_DOUBLE, local, int(cells()),
                         MPI_DOUBLE, 0, MPI_COMM_WORLD);
            return;
        }
#endif
        std::copy(global, global + cells(), local);
    }

    // The whole grid on rank 0, from each rank's block.
    void gather(const double* local, double* global) const {
#ifdef WAVEFORM_MPI
        if (size_ > 1) {
            std::vector<int> counts, offsets;
            layout(counts, offsets);
            MPI_Gatherv(local, int(cells()), MPI_DOUBLE, global, counts.data(), offsets.data(),
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            return;
        }
#endif
        std::copy(local, local + cells(), global);
    }

    // The sum over the ranks, on rank 0: each receiver belongs to one rank.
    void sum(const double* local, double* global, std::size_t n) const {
#ifdef WAVEFORM_MPI
        if (size_ > 1) {
            MPI_Reduce(local, global, int(n), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            return;
        }
#endif
        std::copy(local, local + n, global);
    }

    void broadcast(double* data, std::size_t n) const {
#ifdef WAVEFORM_MPI
        if (size_ > 1) MPI_Bcast(data, int(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
        (void)data;
        (void)n;
#endif
    }

    void broadcast(std::int64_t* data, std::size_t n) const {
#ifdef WAVEFORM_MPI
        if (size_ > 1) MPI_Bcast(data, int(n), MPI_INT64_T, 0, MPI_COMM_WORLD);
#else
        (void)data;
        (void)n;
#endif
    }

  private:
    int nx_, nz_;
    int rank_ = 0, size_ = 1;
    std::vector<int> start_;

    void layout(std::vector<int>& counts, std::vector<int>& offsets) const {
        counts.resize(std::size_t(size_));
        offsets.resize(std::size_t(size_));
        for (int r = 0; r < size_; ++r) {
            counts[r] = (start_[r + 1] - start_[r]) * nz_;
            offsets[r] = start_[r] * nz_;
        }
    }
};

// --------------------------------------------------------------------- //
//                              The model                                //
// --------------------------------------------------------------------- //

class Model {
  public:
    Model(int nx, int nz, double h, double dt, int steps, int width, double rate,
          const std::vector<std::int64_t>& receiver_x,
          const std::vector<std::int64_t>& receiver_z)
        : nx_(nx), nz_(nz), h_(h), dt_(dt), steps_(steps), width_(width), rate_(rate),
          ranks_(check_grid_size(nx, nz), nz) {
        if (h <= 0.0 || dt <= 0.0) throw std::invalid_argument("h and dt must be positive.");
        if (steps < 1) throw std::invalid_argument("At least one time step is needed.");
        if (width < 0 || 2 * width >= nx || width >= nz)
            throw std::invalid_argument("The sponge is wider than the grid.");
        if (receiver_x.size() != receiver_z.size())
            throw std::invalid_argument("Receiver x and z have different lengths.");
        cells_ = ranks_.cells();
        columns_ = ranks_.columns();
        x0_ = ranks_.x0();
        n_receivers_ = receiver_x.size();
        receivers_ = local_points(receiver_x, receiver_z, "receiver");

        // The Cerjan taper: exp(-(rate (width - d))^2) at d cells from an edge.
        auto edge = [&](int d) {
            if (d >= width_) return 1.0;
            const double x = rate_ * double(width_ - d);
            return std::exp(-x * x);
        };
        taper_.resize(cells_);
        inverse_taper_.resize(cells_);
        for (int j = 0; j < columns_; ++j)
            for (int iz = 0; iz < nz_; ++iz) {
                const int ix = x0_ + j;
                const std::size_t i = std::size_t(j) * nz_ + iz;
                taper_[i] = edge(ix) * edge(nx_ - 1 - ix) * edge(nz_ - 1 - iz);
                inverse_taper_[i] = 1.0 / taper_[i];
            }
        zero_.assign(cells_, 0.0);

        const double h2 = h_ * h_;
        w0_ = -2.5 / h2;
        w1_ = 4.0 / 3.0 / h2;
        w2_ = -1.0 / 12.0 / h2;
    }

    int nx() const { return nx_; }
    int nz() const { return nz_; }
    double h() const { return h_; }
    double dt() const { return dt_; }
    int steps() const { return steps_; }
    std::size_t n_receivers() const { return n_receivers_; }
    int rank() const { return ranks_.rank(); }
    int size() const { return ranks_.size(); }
    std::vector<int> column_starts() const { return ranks_.starts(); }

    // The fields this rank holds, and their size.
    std::size_t held() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        return fields_.size();
    }
    double held_megabytes() const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        double total = 0.0;
        for (const auto& entry : fields_) total += double(entry.second.size()) * 8.0 / 1e6;
        return total;
    }

    // ------------------------------------------------------------------ //
    //                      The calls Python makes                        //
    // ------------------------------------------------------------------ //
    //
    // On rank 0. Each validates its arguments, tells the other ranks what to
    // do, and runs its kernel on its own block, the others running theirs
    // in `serve()`.

    Array taper() const {
        Command command(*this, Op::Taper);
        double* out = root_grid();
        command.run([&] { ranks_.gather(taper_.data(), out); });
        return owned(out, {std::size_t(nx_), std::size_t(nz_)});
    }

    Array laplacian(const Grid& u, bool transpose, bool reference) const {
        check_grid(u, "u");
        if (reference && ranks_.size() > 1)
            throw std::invalid_argument("The literal Laplacian is computed on one rank.");
        double* out = root_grid();
        if (reference) {
            nb::gil_scoped_release release;
            reference_laplacian(u.data(), out, transpose);
        } else {
            Command command(*this, Op::Laplacian, {transpose ? 1 : 0});
            command.run([&] { body_laplacian(u.data(), out, transpose); });
        }
        return owned(out, {std::size_t(nx_), std::size_t(nz_)});
    }

    // A point-source run: (handle, data), or (None, data) without keep.
    // traces is (N, n_sources), injected as f / h^2 at the source cells.
    nb::tuple forward(const Grid& m, const std::vector<std::int64_t>& source_x,
                      const std::vector<std::int64_t>& source_z, const Grid& traces,
                      bool keep) const {
        check_model(m, "m");
        if (source_x.size() != source_z.size())
            throw std::invalid_argument("Source x and z have different lengths.");
        for (std::size_t s = 0; s < source_x.size(); ++s) check_point(source_x[s], source_z[s], "source");
        check_traces(traces, source_x.size(), "traces");
        std::vector<std::int64_t> points(source_x);
        points.insert(points.end(), source_z.begin(), source_z.end());
        double* data = root_data();
        std::int64_t handle = -1;
        Command command(*this, Op::Forward,
                        {std::int64_t(source_x.size()), keep ? 1 : 0});
        command.run([&] { handle = body_forward(m.data(), points, traces.data(), keep, data); });
        return nb::make_tuple(handle_or_none(handle), data_array(data));
    }

    // The kept steps of a point-source run, every stride-th, gathered: for
    // snapshots. (fields, data).
    nb::tuple snapshots(const Grid& m, const std::vector<std::int64_t>& source_x,
                        const std::vector<std::int64_t>& source_z, const Grid& traces,
                        int stride) const {
        check_model(m, "m");
        if (source_x.size() != source_z.size())
            throw std::invalid_argument("Source x and z have different lengths.");
        for (std::size_t s = 0; s < source_x.size(); ++s) check_point(source_x[s], source_z[s], "source");
        check_traces(traces, source_x.size(), "traces");
        if (stride < 1) throw std::invalid_argument("stride must be positive.");
        std::vector<std::int64_t> points(source_x);
        points.insert(points.end(), source_z.begin(), source_z.end());
        const std::size_t kept = (std::size_t(steps_) + stride - 1) / stride;
        double* out = ranks_.root() ? new double[kept * std::size_t(nx_) * nz_] : nullptr;
        double* data = root_data();
        Command command(*this, Op::Snapshots, {std::int64_t(source_x.size()), stride});
        command.run([&] { body_snapshots(m.data(), points, traces.data(), stride, out, data); });
        return nb::make_tuple(owned(out, {kept, std::size_t(nx_), std::size_t(nz_)}),
                              data_array(data));
    }

    // The linearized run along dm from a forward field: (handle or None, data).
    nb::tuple born(const Grid& m, std::int64_t field, const Grid& dm, bool keep) const {
        check_model(m, "m");
        check_grid(dm, "dm");
        check_handle(field);
        double* data = root_data();
        std::int64_t handle = -1;
        Command command(*this, Op::Born, {field, keep ? 1 : 0});
        command.run([&] { handle = body_born(m.data(), field, dm.data(), keep, data); });
        return nb::make_tuple(handle_or_none(handle), data_array(data));
    }

    // The data of F''(m)[dm, dm2], the three runs in lockstep.
    Array second_born(const Grid& m, std::int64_t field, const Grid& dm, const Grid& dm2) const {
        check_model(m, "m");
        check_grid(dm, "dm");
        check_grid(dm2, "dm2");
        check_handle(field);
        double* data = root_data();
        Command command(*this, Op::SecondBorn, {field});
        command.run([&] { body_second_born(m.data(), field, dm.data(), dm2.data(), data); });
        return data_array(data);
    }

    // The adjoint field of residual traces (N, n_receivers): a handle.
    std::int64_t adjoint(const Grid& m, const Grid& traces) const {
        check_model(m, "m");
        check_traces(traces, n_receivers_, "traces");
        std::int64_t handle = -1;
        Command command(*this, Op::Adjoint);
        command.run([&] { handle = body_adjoint(m.data(), traces.data()); });
        return handle;
    }

    // The second adjoint field, sourced by dm times the adjoint's acceleration.
    std::int64_t adjoint_born(const Grid& m, std::int64_t lam, const Grid& dm) const {
        check_model(m, "m");
        check_grid(dm, "dm");
        check_handle(lam);
        std::int64_t handle = -1;
        Command command(*this, Op::AdjointBorn, {lam});
        command.run([&] { handle = body_adjoint_born(m.data(), lam, dm.data()); });
        return handle;
    }

    // -sum_n a(A)^n * B^n, a grid.
    Array correlate(std::int64_t a, std::int64_t b) const {
        check_handle(a);
        check_handle(b);
        double* out = root_grid();
        Command command(*this, Op::Correlate, {a, b});
        command.run([&] { body_correlate(a, b, out); });
        return owned(out, {std::size_t(nx_), std::size_t(nz_)});
    }

    // correlate(field, adjoint(m, traces)) in one backward pass.
    Array gradient(const Grid& m, std::int64_t field, const Grid& traces) const {
        check_model(m, "m");
        check_handle(field);
        check_traces(traces, n_receivers_, "traces");
        double* out = root_grid();
        Command command(*this, Op::Gradient, {field});
        command.run([&] { body_gradient(m.data(), field, traces.data(), out); });
        return owned(out, {std::size_t(nx_), std::size_t(nz_)});
    }

    // correlate(dfield, L) - correlate(field, adjoint_born(m, L, dm)), with
    // L = adjoint(m, traces), in one backward pass.
    Array curvature(const Grid& m, std::int64_t field, std::int64_t dfield, const Grid& traces,
                    const Grid& dm) const {
        check_model(m, "m");
        check_grid(dm, "dm");
        check_handle(field);
        check_handle(dfield);
        check_traces(traces, n_receivers_, "traces");
        double* out = root_grid();
        Command command(*this, Op::Curvature, {field, dfield});
        command.run([&] { body_curvature(m.data(), field, dfield, traces.data(), dm.data(), out); });
        return owned(out, {std::size_t(nx_), std::size_t(nz_)});
    }

    // A field, gathered whole: (N, nx, nz).
    Array field(std::int64_t handle) const {
        check_handle(handle);
        double* out =
            ranks_.root() ? new double[std::size_t(steps_) * std::size_t(nx_) * nz_] : nullptr;
        Command command(*this, Op::Field, {handle});
        command.run([&] { body_field(handle, out); });
        return owned(out, {std::size_t(steps_), std::size_t(nx_), std::size_t(nz_)});
    }

    void release(std::int64_t handle) const {
        check_handle(handle);
        Command command(*this, Op::Release, {handle});
        command.run([&] { drop(handle); });
    }

    // The other ranks: run the commands rank 0 sends until it says stop.
    void serve() const {
#ifdef WAVEFORM_MPI
        if (ranks_.root()) throw std::logic_error("Rank 0 drives; the other ranks serve.");
        nb::gil_scoped_release release;
        for (;;) {
            std::int64_t header[4] = {0, 0, 0, 0};
            ranks_.broadcast(header, 4);
            const Op op = Op(header[0]);
            try {
                if (op == Op::Stop) return;
                dispatch(op, header + 1);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "rank %d: %s\n", ranks_.rank(), error.what());
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
#endif
    }

    // Rank 0: release the other ranks from `serve()`.
    void stop() const {
#ifdef WAVEFORM_MPI
        if (ranks_.size() > 1 && ranks_.root()) {
            std::lock_guard<std::mutex> lock(command_mutex_);
            std::int64_t header[4] = {std::int64_t(Op::Stop), 0, 0, 0};
            ranks_.broadcast(header, 4);
        }
#endif
    }

  private:
    enum class Op : std::int64_t {
        Stop,
        Taper,
        Laplacian,
        Forward,
        Snapshots,
        Born,
        SecondBorn,
        Adjoint,
        AdjointBorn,
        Correlate,
        Gradient,
        Curvature,
        Field,
        Release,
    };

    // One call from rank 0: the header to the other ranks, then the kernel
    // here with the interpreter lock released. Across ranks the calls go one
    // at a time; on one rank they may run on as many threads as call them.
    class Command {
      public:
        Command(const Model& model, Op op, std::initializer_list<std::int64_t> args = {})
            : model_(model) {
            header_[0] = std::int64_t(op);
            std::size_t k = 1;
            for (std::int64_t value : args) header_[k++] = value;
        }
        template <class Body>
        void run(Body body) {
            nb::gil_scoped_release release;
            if (model_.ranks_.size() == 1) {
                body();
                return;
            }
            std::lock_guard<std::mutex> lock(model_.command_mutex_);
            model_.ranks_.broadcast(header_, 4);
            body();
        }

      private:
        const Model& model_;
        std::int64_t header_[4] = {0, 0, 0, 0};
    };

    // The other ranks' half of each call: the same kernels, with no global
    // arrays of their own.
    void dispatch(Op op, const std::int64_t* args) const {
        switch (op) {
            case Op::Taper:
                ranks_.gather(taper_.data(), nullptr);
                break;
            case Op::Laplacian:
                body_laplacian(nullptr, nullptr, args[0] != 0);
                break;
            case Op::Forward: {
                std::vector<std::int64_t> points(2 * std::size_t(args[0]));
                body_forward(nullptr, points, nullptr, args[1] != 0, nullptr);
                break;
            }
            case Op::Snapshots: {
                std::vector<std::int64_t> points(2 * std::size_t(args[0]));
                body_snapshots(nullptr, points, nullptr, int(args[1]), nullptr, nullptr);
                break;
            }
            case Op::Born:
                body_born(nullptr, args[0], nullptr, args[1] != 0, nullptr);
                break;
            case Op::SecondBorn:
                body_second_born(nullptr, args[0], nullptr, nullptr, nullptr);
                break;
            case Op::Adjoint:
                body_adjoint(nullptr, nullptr);
                break;
            case Op::AdjointBorn:
                body_adjoint_born(nullptr, args[0], nullptr);
                break;
            case Op::Correlate:
                body_correlate(args[0], args[1], nullptr);
                break;
            case Op::Gradient:
                body_gradient(nullptr, args[0], nullptr, nullptr);
                break;
            case Op::Curvature:
                body_curvature(nullptr, args[0], args[1], nullptr, nullptr, nullptr);
                break;
            case Op::Field:
                body_field(args[0], nullptr);
                break;
            case Op::Release:
                drop(args[0]);
                break;
            default:
                throw std::runtime_error("Unknown command.");
        }
    }

    // ------------------------------------------------------------------ //
    //                     The kernels, on one block                       //
    // ------------------------------------------------------------------ //
    //
    // Every argument named *_global lives on rank 0 only (null elsewhere);
    // each kernel scatters, broadcasts and gathers what it needs. Owned
    // arrays hold this rank's columns, columns_ * nz_ values; working arrays
    // hold two more columns of halo on each side, and own() points at their
    // owned part, so that pointwise loops index both alike.

    using Buffer = std::vector<double>;

    Buffer halo_buffer() const { return Buffer(std::size_t(columns_ + 4) * nz_, 0.0); }
    double* own(Buffer& b) const { return b.data() + 2 * std::size_t(nz_); }
    const double* own(const Buffer& b) const { return b.data() + 2 * std::size_t(nz_); }

    // dt^2 / m on this block, from the model rank 0 holds.
    Buffer coefficient(const double* m_global) const {
        Buffer m(cells_);
        ranks_.scatter(m_global, m.data());
        for (std::size_t i = 0; i < cells_; ++i) m[i] = dt_ * dt_ / m[i];
        return m;
    }

    Buffer local_grid(const double* global) const {
        Buffer out(cells_);
        ranks_.scatter(global, out.data());
        return out;
    }

    Buffer broadcast_traces(const double* traces_global, std::size_t columns) const {
        Buffer traces(std::size_t(steps_) * columns);
        if (ranks_.root()) std::copy(traces_global, traces_global + traces.size(), traces.begin());
        ranks_.broadcast(traces.data(), traces.size());
        return traces;
    }

    void body_laplacian(const double* u_global, double* out_global, bool transpose) const {
        Buffer u = halo_buffer();
        ranks_.scatter(u_global, own(u));
        ranks_.exchange(u.data());
        Buffer out(cells_), column(nz_);
        for (int j = 0; j < columns_; ++j) {
            const double* c = u.data() + std::size_t(j + 2) * nz_;
            if (transpose)
                column_stencil<true>(c, out.data() + std::size_t(j) * nz_);
            else
                column_stencil<false>(c, out.data() + std::size_t(j) * nz_);
        }
        ranks_.gather(out.data(), out_global);
    }

    std::int64_t body_forward(const double* m_global, std::vector<std::int64_t>& points,
                              const double* traces_global, bool keep,
                              double* data_global) const {
        const Buffer c = coefficient(m_global);
        ranks_.broadcast(points.data(), points.size());
        const std::size_t ns = points.size() / 2;
        const Buffer traces = broadcast_traces(traces_global, ns);
        const Sources sources = local_sources(points);
        std::int64_t handle = keep ? store(Buffer(std::size_t(steps_) * cells_)) : -1;
        double* field = keep ? data_of(handle) : nullptr;
        Buffer prev = halo_buffer(), cur = halo_buffer(), next = halo_buffer(), column(nz_);
        Buffer data(std::size_t(steps_) * n_receivers_, 0.0), values(ns);
        const double scale = 1.0 / (h_ * h_);
        for (int n = 0; n < steps_; ++n) {
            step<false>(c, prev, cur, next, column, nullptr);
            for (std::size_t s = 0; s < ns; ++s) values[s] = scale * traces[std::size_t(n) * ns + s];
            inject(c, sources, values.data(), own(next));
            ranks_.exchange(next.data());
            sample(own(next), data.data() + std::size_t(n) * n_receivers_);
            if (keep) std::copy(own(next), own(next) + cells_, field + std::size_t(n) * cells_);
            std::swap(prev, cur);
            std::swap(cur, next);
        }
        ranks_.sum(data.data(), data_global, data.size());
        return handle;
    }

    void body_snapshots(const double* m_global, std::vector<std::int64_t>& points,
                        const double* traces_global, int stride, double* out_global,
                        double* data_global) const {
        const Buffer c = coefficient(m_global);
        ranks_.broadcast(points.data(), points.size());
        const std::size_t ns = points.size() / 2;
        const Buffer traces = broadcast_traces(traces_global, ns);
        const Sources sources = local_sources(points);
        Buffer prev = halo_buffer(), cur = halo_buffer(), next = halo_buffer(), column(nz_);
        Buffer data(std::size_t(steps_) * n_receivers_, 0.0), values(ns);
        const double scale = 1.0 / (h_ * h_);
        const std::size_t grid = std::size_t(nx_) * nz_;
        for (int n = 0; n < steps_; ++n) {
            step<false>(c, prev, cur, next, column, nullptr);
            for (std::size_t s = 0; s < ns; ++s) values[s] = scale * traces[std::size_t(n) * ns + s];
            inject(c, sources, values.data(), own(next));
            ranks_.exchange(next.data());
            sample(own(next), data.data() + std::size_t(n) * n_receivers_);
            if (n % stride == 0)
                ranks_.gather(own(next), out_global ? out_global + std::size_t(n / stride) * grid
                                                    : nullptr);
            std::swap(prev, cur);
            std::swap(cur, next);
        }
        ranks_.sum(data.data(), data_global, data.size());
    }

    std::int64_t body_born(const double* m_global, std::int64_t field_handle,
                           const double* dm_global, bool keep, double* data_global) const {
        const Buffer c = coefficient(m_global);
        const Buffer dm = local_grid(dm_global);
        const double* field = data_of(field_handle);
        std::int64_t handle = keep ? store(Buffer(std::size_t(steps_) * cells_)) : -1;
        double* out = keep ? data_of(handle) : nullptr;
        Buffer prev = halo_buffer(), cur = halo_buffer(), next = halo_buffer(), column(nz_);
        Buffer source(cells_), data(std::size_t(steps_) * n_receivers_, 0.0);
        for (int n = 0; n < steps_; ++n) {
            scaled(source.data(), -1.0, dm.data(), acceleration(field, n));
            step<false>(c, prev, cur, next, column, source.data());
            ranks_.exchange(next.data());
            sample(own(next), data.data() + std::size_t(n) * n_receivers_);
            if (keep) std::copy(own(next), own(next) + cells_, out + std::size_t(n) * cells_);
            std::swap(prev, cur);
            std::swap(cur, next);
        }
        ranks_.sum(data.data(), data_global, data.size());
        return handle;
    }

    void body_second_born(const double* m_global, std::int64_t field_handle,
                          const double* dm_global, const double* dm2_global,
                          double* data_global) const {
        const Buffer c = coefficient(m_global);
        const Buffer dm = local_grid(dm_global);
        const Buffer dm2 = local_grid(dm2_global);
        const double* field = data_of(field_handle);
        const double inv_dt2 = 1.0 / (dt_ * dt_);
        Buffer ap = halo_buffer(), a0 = halo_buffer(), an = halo_buffer();
        Buffer bp = halo_buffer(), b0 = halo_buffer(), bn = halo_buffer();
        Buffer sp = halo_buffer(), s0 = halo_buffer(), sn = halo_buffer();
        Buffer column(nz_), source(cells_), other(cells_);
        Buffer data(std::size_t(steps_) * n_receivers_, 0.0);
        for (int n = 0; n < steps_; ++n) {
            const Acceleration a = acceleration(field, n);
            scaled(source.data(), -1.0, dm.data(), a);
            step<false>(c, ap, a0, an, column, source.data());
            ranks_.exchange(an.data());
            scaled(source.data(), -1.0, dm2.data(), a);
            step<false>(c, bp, b0, bn, column, source.data());
            ranks_.exchange(bn.data());
            // -(dm * a(dU2) + dm2 * a(dU)) at step n, from the steps just taken.
            scaled(source.data(), -1.0, dm.data(),
                   {own(bn), own(b0), own(bp), inverse_taper_.data(), taper_.data(), inv_dt2});
            scaled(other.data(), -1.0, dm2.data(),
                   {own(an), own(a0), own(ap), inverse_taper_.data(), taper_.data(), inv_dt2});
            for (std::size_t i = 0; i < cells_; ++i) source[i] += other[i];
            step<false>(c, sp, s0, sn, column, source.data());
            ranks_.exchange(sn.data());
            sample(own(sn), data.data() + std::size_t(n) * n_receivers_);
            std::swap(ap, a0);
            std::swap(a0, an);
            std::swap(bp, b0);
            std::swap(b0, bn);
            std::swap(sp, s0);
            std::swap(s0, sn);
        }
        ranks_.sum(data.data(), data_global, data.size());
    }

    // For k = N .. 1: l^{k-1} = tau (2 l^k - tau l^{k+1} + c (L^T l^k + S^T r^k)),
    // from l^N = l^{N+1} = 0, with r^k the residual of the data of U[k-1].
    // L[n] = l^n, n = 0 .. N-1.
    std::int64_t body_adjoint(const double* m_global, const double* traces_global) const {
        const Buffer c = coefficient(m_global);
        const Buffer traces = broadcast_traces(traces_global, n_receivers_);
        std::int64_t handle = store(Buffer(std::size_t(steps_) * cells_));
        double* out = data_of(handle);
        Buffer b = halo_buffer(), a = halo_buffer(), l = halo_buffer(), column(nz_);
        for (int k = steps_; k >= 1; --k) {
            step<true>(c, b, a, l, column, nullptr);
            inject(c, receivers_, traces.data() + std::size_t(k - 1) * n_receivers_, own(l));
            ranks_.exchange(l.data());
            std::copy(own(l), own(l) + cells_, out + std::size_t(k - 1) * cells_);
            std::swap(b, a);
            std::swap(a, l);
        }
        return handle;
    }

    // A^T M = (dA[dm])^T L: the source at k is dm (l^{k-1} / tau - 2 l^k + tau l^{k+1}) / dt^2.
    std::int64_t body_adjoint_born(const double* m_global, std::int64_t lam_handle,
                                   const double* dm_global) const {
        const Buffer c = coefficient(m_global);
        const Buffer dm = local_grid(dm_global);
        const double* lam = data_of(lam_handle);
        std::int64_t handle = store(Buffer(std::size_t(steps_) * cells_));
        double* out = data_of(handle);
        Buffer b = halo_buffer(), a = halo_buffer(), l = halo_buffer(), column(nz_), source(cells_);
        for (int k = steps_; k >= 1; --k) {
            scaled(source.data(), 1.0, dm.data(), transposed_acceleration(lam, k));
            step<true>(c, b, a, l, column, source.data());
            ranks_.exchange(l.data());
            std::copy(own(l), own(l) + cells_, out + std::size_t(k - 1) * cells_);
            std::swap(b, a);
            std::swap(a, l);
        }
        return handle;
    }

    void body_correlate(std::int64_t a_handle, std::int64_t b_handle, double* out_global) const {
        const double* a = data_of(a_handle);
        const double* b = data_of(b_handle);
        Buffer out(cells_, 0.0);
        for (int n = 0; n < steps_; ++n)
            accumulate(out.data(), -1.0, acceleration(a, n), b + std::size_t(n) * cells_);
        ranks_.gather(out.data(), out_global);
    }

    void body_gradient(const double* m_global, std::int64_t field_handle,
                       const double* traces_global, double* out_global) const {
        const Buffer c = coefficient(m_global);
        const Buffer traces = broadcast_traces(traces_global, n_receivers_);
        const double* field = data_of(field_handle);
        Buffer out(cells_, 0.0);
        Buffer b = halo_buffer(), a = halo_buffer(), l = halo_buffer(), column(nz_);
        for (int k = steps_; k >= 1; --k) {
            step<true>(c, b, a, l, column, nullptr);
            inject(c, receivers_, traces.data() + std::size_t(k - 1) * n_receivers_, own(l));
            ranks_.exchange(l.data());
            accumulate(out.data(), -1.0, acceleration(field, k - 1), own(l));
            std::swap(b, a);
            std::swap(a, l);
        }
        ranks_.gather(out.data(), out_global);
    }

    void body_curvature(const double* m_global, std::int64_t field_handle,
                        std::int64_t dfield_handle, const double* traces_global,
                        const double* dm_global, double* out_global) const {
        const Buffer c = coefficient(m_global);
        const Buffer traces = broadcast_traces(traces_global, n_receivers_);
        const Buffer dm = local_grid(dm_global);
        const double* field = data_of(field_handle);
        const double* dfield = data_of(dfield_handle);
        const double inv_dt2 = 1.0 / (dt_ * dt_);
        Buffer out(cells_, 0.0);
        Buffer lb = halo_buffer(), la = halo_buffer(), l = halo_buffer();
        Buffer mb = halo_buffer(), ma = halo_buffer(), mu = halo_buffer();
        Buffer column(nz_), source(cells_);
        for (int k = steps_; k >= 1; --k) {
            // l^{k-1} from l^k = la and l^{k+1} = lb.
            step<true>(c, lb, la, l, column, nullptr);
            inject(c, receivers_, traces.data() + std::size_t(k - 1) * n_receivers_, own(l));
            ranks_.exchange(l.data());
            // mu^{k-1}, sourced by dm times the transposed acceleration of l at k.
            scaled(source.data(), 1.0, dm.data(),
                   {own(l), own(la), own(lb), inverse_taper_.data(), taper_.data(), inv_dt2});
            step<true>(c, mb, ma, mu, column, source.data());
            ranks_.exchange(mu.data());
            accumulate(out.data(), -1.0, acceleration(dfield, k - 1), own(l));
            accumulate(out.data(), 1.0, acceleration(field, k - 1), own(mu));
            std::swap(lb, la);
            std::swap(la, l);
            std::swap(mb, ma);
            std::swap(ma, mu);
        }
        ranks_.gather(out.data(), out_global);
    }

    void body_field(std::int64_t handle, double* out_global) const {
        const double* field = data_of(handle);
        const std::size_t grid = std::size_t(nx_) * nz_;
        for (int n = 0; n < steps_; ++n)
            ranks_.gather(field + std::size_t(n) * cells_,
                          out_global ? out_global + std::size_t(n) * grid : nullptr);
    }

    // ------------------------------------------------------------------ //
    //                           The stencil                              //
    // ------------------------------------------------------------------ //

    // One column of L u, or of L^T u, into out[0 .. nz): c points at the
    // column in a halo buffer, so its neighbours are c -+ nz and c -+ 2 nz.
    //
    // L, rows iz >= 1: the five-point stencil along x and along z, the
    // z-stencil reaching row 0 of u where it exists and the image -u(-j)
    // above it (only row 1 reaches above, to -u(1)). Row 0: zero.
    //
    // L^T, the same terms read the other way: rows iz >= 1 are L's rows
    // without the reach into row 0, because no row of L is reached *from*
    // row 0 (L's row 0 being zero), and row 0 is what rows 1 and 2 of L take
    // from it, w1 u(1) + w2 u(2). The image is a diagonal term, its own
    // transpose. Every other term is symmetric: zero beyond the bottom and
    // sides, and the same weights both ways.
    template <bool Transpose>
    void column_stencil(const double* c, double* __restrict__ out) const {
        const std::size_t n = std::size_t(nz_);
        const double* xm2 = c - 2 * n;
        const double* xm1 = c - n;
        const double* xp1 = c + n;
        const double* xp2 = c + 2 * n;
        // The x-terms and both center terms, the same on every row.
        auto across = [&](int iz) {
            return 2.0 * w0_ * c[iz] + w1_ * (xm1[iz] + xp1[iz]) + w2_ * (xm2[iz] + xp2[iz]);
        };
        // Row 0 of u is reached from rows 1 and 2 of L, and not at all by L^T.
        const double top = Transpose ? 0.0 : c[0];
        const int n1 = nz_ - 1, n2 = nz_ - 2;
        out[0] = Transpose ? w1_ * c[1] + w2_ * c[2] : 0.0;
        // Row 1: its z-stencil reaches row 0 and, above it, the image -u(1).
        out[1] = across(1) + w2_ * (c[3] - c[1]) + w1_ * (top + c[2]);
        out[2] = across(2) + w2_ * (top + c[4]) + w1_ * (c[1] + c[3]);
        for (int iz = 3; iz < n2; ++iz)
            out[iz] = 2.0 * w0_ * c[iz] + w1_ * (xm1[iz] + xp1[iz] + c[iz - 1] + c[iz + 1]) +
                      w2_ * (xm2[iz] + xp2[iz] + c[iz - 2] + c[iz + 2]);
        // The bottom two rows: zero below the grid.
        out[n2] = across(n2) + w2_ * c[n2 - 2] + w1_ * (c[n2 - 1] + c[n1]);
        out[n1] = across(n1) + w2_ * c[n1 - 2] + w1_ * c[n1 - 1];
    }

    // L u as the literal sum of its terms, and L^T v as the literal scatter
    // of the same terms, on the whole grid: the reference the column stencils
    // are checked on.
    void reference_laplacian(const double* u, double* out, bool transpose) const {
        const std::size_t total = std::size_t(nx_) * nz_;
        std::fill(out, out + total, 0.0);
        const double weight[3] = {w0_, w1_, w2_};
        auto idx = [&](int ix, int iz) { return std::size_t(ix) * nz_ + iz; };
        // term(i, j, w): (L u)_i += w u_j.
        auto term = [&](std::size_t i, std::size_t j, double w) {
            if (transpose)
                out[j] += w * u[i];
            else
                out[i] += w * u[j];
        };
        for (int ix = 0; ix < nx_; ++ix)
            for (int iz = 1; iz < nz_; ++iz) {
                const std::size_t i = idx(ix, iz);
                for (int k = -2; k <= 2; ++k) {
                    const double w = weight[std::abs(k)];
                    if (ix + k >= 0 && ix + k < nx_) term(i, idx(ix + k, iz), w);
                    const int j = iz + k;
                    if (j >= nz_) continue;
                    if (j >= 0)
                        term(i, idx(ix, j), w);
                    else
                        term(i, idx(ix, -j), -w);
                }
            }
    }

    // ------------------------------------------------------------------ //
    //                       Steps and pointwise work                     //
    // ------------------------------------------------------------------ //

    // The acceleration of a field at one step, cell by cell: for a forward
    // field U[n] = u^{n+1}, (u^{n+1} / tau - 2 u^n + tau u^{n-1}) / dt^2; for
    // an adjoint field L[n] = l^n at k, (l^{k-1} / tau - 2 l^k + tau l^{k+1}) / dt^2.
    // The steps outside the record are zero.
    struct Acceleration {
        const double* first;
        const double* middle;
        const double* last;
        const double* inverse_taper;
        const double* taper;
        double inv_dt2;
    };

    Acceleration acceleration(const double* field, int n) const {
        const double* zero = zero_.data();
        return {field + std::size_t(n) * cells_,
                n >= 1 ? field + std::size_t(n - 1) * cells_ : zero,
                n >= 2 ? field + std::size_t(n - 2) * cells_ : zero, inverse_taper_.data(),
                taper_.data(), 1.0 / (dt_ * dt_)};
    }

    Acceleration transposed_acceleration(const double* lam, int k) const {
        const double* zero = zero_.data();
        return {lam + std::size_t(k - 1) * cells_,
                k < steps_ ? lam + std::size_t(k) * cells_ : zero,
                k + 1 < steps_ ? lam + std::size_t(k + 1) * cells_ : zero, inverse_taper_.data(),
                taper_.data(), 1.0 / (dt_ * dt_)};
    }

    // next = tau (2 cur - tau prev + c (L cur + rhs)) on the owned columns,
    // L or L^T; cur's halos must be current; rhs is owned, or null for none.
    template <bool Transpose>
    void step(const Buffer& c, const Buffer& prev, const Buffer& cur, Buffer& next,
              Buffer& column, const double* rhs) const {
        const double* __restrict__ taper = taper_.data();
        const double* __restrict__ coef = c.data();
        const double* __restrict__ source = rhs ? rhs : zero_.data();
        const double* __restrict__ p = own(prev);
        const double* __restrict__ u = own(cur);
        double* __restrict__ v = own(next);
        double* __restrict__ col = column.data();
        for (int j = 0; j < columns_; ++j) {
            column_stencil<Transpose>(cur.data() + std::size_t(j + 2) * nz_, col);
            const std::size_t base = std::size_t(j) * nz_;
            for (int iz = 0; iz < nz_; ++iz) {
                const std::size_t i = base + iz;
                v[i] = taper[i] * (2.0 * u[i] - taper[i] * p[i] + coef[i] * (col[iz] + source[i]));
            }
        }
    }

    // out = sign dm * a, on the owned cells.
    void scaled(double* __restrict__ out, double sign, const double* __restrict__ dm,
                const Acceleration& a) const {
        const double* __restrict__ f = a.first;
        const double* __restrict__ g = a.middle;
        const double* __restrict__ h = a.last;
        const double* __restrict__ t = a.inverse_taper;
        const double* __restrict__ u = a.taper;
        const double scale = sign * a.inv_dt2;
        for (std::size_t i = 0; i < cells_; ++i)
            out[i] = scale * dm[i] * (f[i] * t[i] - 2.0 * g[i] + u[i] * h[i]);
    }

    // out += sign a * b, on the owned cells.
    void accumulate(double* __restrict__ out, double sign, const Acceleration& a,
                    const double* __restrict__ b) const {
        const double* __restrict__ f = a.first;
        const double* __restrict__ g = a.middle;
        const double* __restrict__ h = a.last;
        const double* __restrict__ t = a.inverse_taper;
        const double* __restrict__ u = a.taper;
        const double scale = sign * a.inv_dt2;
        for (std::size_t i = 0; i < cells_; ++i)
            out[i] += scale * (f[i] * t[i] - 2.0 * g[i] + u[i] * h[i]) * b[i];
    }

    // The point terms of the right-hand side, added after a step: the step is
    // linear in its right-hand side, so tau c value at each cell is exact.
    struct Point {
        std::size_t cell;   // owned index on this rank
        std::size_t index;  // which source or receiver
    };
    using Sources = std::vector<Point>;

    void inject(const Buffer& c, const Sources& points, const double* values, double* next) const {
        for (const Point& p : points) next[p.cell] += taper_[p.cell] * c[p.cell] * values[p.index];
    }

    void sample(const double* u, double* out) const {
        for (const Point& p : receivers_) out[p.index] = u[p.cell];
    }

    // The points in this rank's columns, from global (x, z), x's then z's.
    Sources local_sources(const std::vector<std::int64_t>& points) const {
        const std::size_t count = points.size() / 2;
        std::vector<std::int64_t> x(points.begin(), points.begin() + count);
        std::vector<std::int64_t> z(points.begin() + count, points.end());
        return local_points(x, z, "source");
    }

    Sources local_points(const std::vector<std::int64_t>& x, const std::vector<std::int64_t>& z,
                         const char* what) const {
        Sources out;
        for (std::size_t i = 0; i < x.size(); ++i) {
            check_point(x[i], z[i], what);
            if (x[i] >= x0_ && x[i] < x0_ + columns_)
                out.push_back({std::size_t(x[i] - x0_) * nz_ + std::size_t(z[i]), i});
        }
        return out;
    }

    // ------------------------------------------------------------------ //
    //                          Fields and checks                         //
    // ------------------------------------------------------------------ //

    std::int64_t store(Buffer field) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        const std::int64_t handle = next_handle_++;
        fields_.emplace(handle, std::move(field));
        return handle;
    }

    double* data_of(std::int64_t handle) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        auto found = fields_.find(handle);
        if (found == fields_.end())
            throw std::invalid_argument("No field has the handle " + std::to_string(handle) + ".");
        return found->second.data();
    }

    void drop(std::int64_t handle) const {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        fields_.erase(handle);
    }

    void check_handle(std::int64_t handle) const { data_of(handle); }

    nb::object handle_or_none(std::int64_t handle) const {
        return handle < 0 ? nb::none() : nb::cast(handle);
    }

    double* root_grid() const {
        return ranks_.root() ? new double[std::size_t(nx_) * nz_] : nullptr;
    }

    double* root_data() const {
        return ranks_.root() ? new double[std::size_t(steps_) * n_receivers_] : nullptr;
    }

    Array data_array(double* data) const {
        return owned(data, {std::size_t(steps_), n_receivers_});
    }

    static int check_grid_size(int nx, int nz) {
        if (nx < 5 || nz < 7) throw std::invalid_argument("The grid needs 5 columns and 7 rows.");
        return nx;
    }

    void check_point(std::int64_t x, std::int64_t z, const char* what) const {
        if (x < 0 || x >= nx_ || z < 1 || z >= nz_)
            throw std::invalid_argument(std::string("A ") + what +
                                        " lies outside the grid or on the free surface.");
    }

    void check_grid(const Grid& a, const char* name) const {
        if (a.ndim() != 2 || a.shape(0) != std::size_t(nx_) || a.shape(1) != std::size_t(nz_))
            throw std::invalid_argument(std::string(name) + " has shape " + shape_of(a) +
                                        ", not the grid's (" + std::to_string(nx_) + ", " +
                                        std::to_string(nz_) + ").");
    }

    // A model must be positive everywhere; checked here, on rank 0, before
    // any rank is asked to divide by it.
    void check_model(const Grid& m, const char* name) const {
        check_grid(m, name);
        const double* v = m.data();
        for (std::size_t i = 0; i < std::size_t(nx_) * nz_; ++i)
            if (!(v[i] > 0.0)) throw std::invalid_argument("m must be positive everywhere.");
    }

    void check_traces(const Grid& a, std::size_t columns, const char* name) const {
        if (a.ndim() != 2 || a.shape(0) != std::size_t(steps_) || a.shape(1) != columns)
            throw std::invalid_argument(std::string(name) + " has shape " + shape_of(a) +
                                        ", not (" + std::to_string(steps_) + ", " +
                                        std::to_string(columns) + ").");
    }

    int nx_, nz_;
    double h_, dt_;
    int steps_, width_;
    double rate_;
    Ranks ranks_;
    std::size_t cells_ = 0;
    int columns_ = 0, x0_ = 0;
    std::size_t n_receivers_ = 0;
    Sources receivers_;
    Buffer taper_, inverse_taper_, zero_;
    double w0_ = 0.0, w1_ = 0.0, w2_ = 0.0;
    mutable std::map<std::int64_t, Buffer> fields_;
    mutable std::int64_t next_handle_ = 0;
    mutable std::mutex registry_mutex_, command_mutex_;
};

}  // namespace WAVEFORM_NAMESPACE

using WAVEFORM_NAMESPACE::Model;

#ifdef WAVEFORM_MPI
#define MODULE_NAME acoustic_mpi
#else
#define MODULE_NAME acoustic
#endif

NB_MODULE(MODULE_NAME, mod) {
#ifdef WAVEFORM_MPI
    mod.doc() = "The 2-D acoustic solver with its grid split across MPI ranks.";
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided);
    }
    mod.def("finalize", [] {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Finalize();
    });
    mod.def("abort", [](int code) { MPI_Abort(MPI_COMM_WORLD, code); }, nb::arg("code") = 1);
    mod.def("rank", [] {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    });
    mod.def("size", [] {
        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    });
#else
    mod.doc() = "A 2-D acoustic finite-difference solver with its linearization and adjoints.";
#endif

    nb::class_<Model>(mod, "Model")
        .def(nb::init<int, int, double, double, int, int, double, const std::vector<std::int64_t>&,
                      const std::vector<std::int64_t>&>(),
             nb::arg("nx"), nb::arg("nz"), nb::arg("h"), nb::arg("dt"), nb::arg("steps"),
             nb::arg("width"), nb::arg("rate"), nb::arg("receiver_x"), nb::arg("receiver_z"))
        .def_prop_ro("nx", &Model::nx)
        .def_prop_ro("nz", &Model::nz)
        .def_prop_ro("h", &Model::h)
        .def_prop_ro("dt", &Model::dt)
        .def_prop_ro("steps", &Model::steps)
        .def_prop_ro("n_receivers", &Model::n_receivers)
        .def_prop_ro("rank", &Model::rank, "This process's rank; 0 serially.")
        .def_prop_ro("size", &Model::size, "The number of ranks; 1 serially.")
        .def_prop_ro("column_starts", &Model::column_starts,
                     "The first column of each rank's block, and nx.")
        .def_prop_ro("held", &Model::held, "How many fields this rank holds.")
        .def_prop_ro("held_megabytes", &Model::held_megabytes,
                     "The memory of the fields this rank holds.")
        .def("taper", &Model::taper, "The sponge taper, an (nx, nz) array.")
        .def("laplacian", &Model::laplacian, nb::arg("u"), nb::arg("transpose") = false,
             nb::arg("reference") = false,
             "L u or L^T u by the columns the steps use, or, with reference=True, as the "
             "literal sum or scatter of L's terms.")
        .def("forward", &Model::forward, nb::arg("m"), nb::arg("source_x"), nb::arg("source_z"),
             nb::arg("traces"), nb::arg("keep") = true,
             "(handle or None, data): a point-source run.")
        .def("snapshots", &Model::snapshots, nb::arg("m"), nb::arg("source_x"),
             nb::arg("source_z"), nb::arg("traces"), nb::arg("stride"),
             "(every stride-th step of the field, data): a point-source run for figures.")
        .def("born", &Model::born, nb::arg("m"), nb::arg("field"), nb::arg("dm"),
             nb::arg("keep") = true, "(handle or None, data): the linearized run.")
        .def("second_born", &Model::second_born, nb::arg("m"), nb::arg("field"), nb::arg("dm"),
             nb::arg("dm2"), "The data of F''(m)[dm, dm2], the three runs in lockstep.")
        .def("adjoint", &Model::adjoint, nb::arg("m"), nb::arg("traces"),
             "The adjoint field of residual traces injected at the receivers: a handle.")
        .def("adjoint_born", &Model::adjoint_born, nb::arg("m"), nb::arg("lam"), nb::arg("dm"),
             "The second adjoint field, sourced by dm times the adjoint's acceleration.")
        .def("correlate", &Model::correlate, nb::arg("a"), nb::arg("b"),
             "-sum_n a(A)^n * B^n, a grid.")
        .def("gradient", &Model::gradient, nb::arg("m"), nb::arg("field"), nb::arg("traces"),
             "correlate(field, adjoint(m, traces)) in one backward pass.")
        .def("curvature", &Model::curvature, nb::arg("m"), nb::arg("field"), nb::arg("dfield"),
             nb::arg("traces"), nb::arg("dm"),
             "correlate(dfield, L) - correlate(field, adjoint_born(m, L, dm)), L = adjoint(m, traces).")
        .def("field", &Model::field, nb::arg("handle"), "A field gathered whole, (N, nx, nz).")
        .def("release", &Model::release, nb::arg("handle"), "Free a field on every rank.")
        .def("serve", &Model::serve, "On ranks other than 0: run rank 0's calls until stop().")
        .def("stop", &Model::stop, "On rank 0: release the other ranks from serve().");
}
