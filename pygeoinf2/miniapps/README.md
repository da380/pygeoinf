# Mini-apps

Mini-apps are small, complete applications of `pygeoinf2`, in the spirit of
MFEM's: each is a self-contained directory with a README stating the problem,
real or realistic data, several features of the library composed into one
inverse problem, and a run of at most a few minutes that prints its numbers
and saves its figure. They live under the package directory but are not imported
by it; they are read, built where they need building, and run from their own
directory. The test suite runs each one that can be run on the machine at
hand and skips the rest with a reason.

| directory | problem |
|---|---|
| `cpp_backend/` | a C++ vector type with its own inner product, bound with nanobind and exposed as a `HilbertSpace` with no array underneath; an inverse problem run on it end to end |
| `mfem/` | MFEM two ways: a finite element space as a `HilbertSpace`, through its matrices and with no coordinates at all, with an inverse problem on each; and an MFEM solve as an operator on the library's own periodic box, its adjoint proved by the check ladder, with the point, set and measure answers on it |
| `planet_density/` | the Earth's radial density from its mass and moment of inertia on a three-layer space: what two numbers say under a loose and a tight prior, as posteriors, a minimum-norm point and feasible brackets; a mixture of two references weighed by their evidence; and the boundary radii as nonlinear unknowns, by the mode, the Laplace Gaussian and Occam |
| `point_data/` | a field on the sphere observed at tens of thousands of ocean points under a Sobolev prior: the one solve behind every answer at scale (plain against surrogate-preconditioned conjugate gradients), then the Bayesian, point and set answers on the same regional averages, and a non-stationary prior as an extension |
| `tomography/` | travel times through a radially layered planet from a user's own ray tracer, wrapped as a path-integral operator on the layered disk through point evaluation at the ray nodes: the Bayesian, point and set answers on the mean slowness under good coverage, poor coverage and a core no ray enters; then the same in a ball, and with the rays re-traced in the model by shooting, its derivative by Fermat's principle and its mode by Gauss-Newton |
| `waveform/` | two-dimensional acoustic waveform inversion on a finite-difference solver written in C++ and bound with nanobind, with its linearization and its first and second adjoints proved by the check ladder: the MAP model by Newton-CG on the Gauss-Newton and the exact Hessian, the Laplace Gaussian on box means through a randomized-eigendecomposition preconditioner, and Occam's inversion; and the same solver with its grid split across MPI ranks by halo exchange, which gives the serial answers bit for bit |
