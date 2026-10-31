# Mini-apps

Mini-apps are small, complete applications of `pygeoinf`: each is a self-contained directory with a README stating the problem,
real or realistic data, several features of the library composed into one
inverse problem, and a run of at most a few minutes that prints its numbers
and saves its figure. They live beside the package, not in it, and are not
imported by it; they are read, built where they need building, and run as scripts,
from their own directory or by path from any other. The test suite runs each
one that can be run on the machine at hand and skips the rest with a reason.

Every app is laid out the same way: its modules in a package `src/`, its
part scripts, `run_all.py`, `README.md` and `IMPLEMENTATION.md` at the top,
the sources and build of a C++ extension in `cpp/`, and whatever a run
writes in `results/` (`results/quick/` at the quick scale), which git
ignores. Every script takes `--quick` or `--full` (the default) to pick a
preset scale, then every field of the app's scale as an option of its own
name (`--noise-std`, `--length-scale`, ...), so that the resolution, the
counts, the noise and the prior can be changed from the command line;
`--help` lists them with the preset's values.

Forward code that more than one app runs lives beside them, outside every app
and outside the library. `raytracer/` is the ray tracer of the tomography app
and of any app to come that traces rays, written without the library: rays in
a planet whose velocity depends on radius alone, integrated piece by piece
between its discontinuities, between a source and a receiver at any radius,
with a node wherever a ray crosses a discontinuity, so that its path integrals
stay second order across them, with the closed form of a medium whose rays are
circles to check it against, and a shooting tracer for a medium that is not
radial. An app loads it as it would load an external code: its `src/rays.py`
puts the directory on the path for the import, takes it off again and
re-exports the tracer, so the app's scripts and tests import `src.rays` and
nothing else. The tracer's own tests need only NumPy and SciPy and run with
the fast suite, in `tests/test_miniapp_raytracer.py`.

| directory | problem |
|---|---|
| `cpp_backend/` | a C++ vector type, bound with nanobind and exposed as `H^1` on an interval, a `HilbertSpace` with no array underneath whose inner product and Riesz map are computed in C++: the check ladder on the space, its operators and a Matérn-type prior on it, then the point, measure and set answers on two local averages from a dozen point values, with a tally of every call the library made of the backend |
| `mfem/` | MFEM two ways: a finite element space as a `HilbertSpace`, through its matrices and with no coordinates at all, with an inverse problem on each; and an MFEM solve as an operator on the library's own periodic box, its adjoint proved by the check ladder, with the point, measure and set answers on it |
| `planet_density/` | the Earth's radial density from its mass and moment of inertia on a space of PREM's own layers, the truth PREM plus a draw from a tight prior about it: the point, measure and set answers under that prior and under a deliberately wide bound, each marked against the truth; and the inner-core and core-mantle boundary radii as nonlinear unknowns, by the MAP model, the Laplace Gaussian and Occam |
| `point_data/` | a field on the sphere observed at tens of thousands of ocean points under a Sobolev prior: the one solve behind every answer at scale (plain against surrogate-preconditioned conjugate gradients), then the point, Bayesian and set answers on the same regional averages, and a non-stationary prior as an extension |
| `tomography/` | travel times through a radially layered planet from a user's own ray tracer, wrapped as a path-integral operator on the layered disk through point evaluation at the ray nodes: the point, Bayesian and set answers on the mean slowness under good coverage, poor coverage and a core no ray enters; a plume at one of five candidate places as a mixture prior, whose posterior weights are the candidates' evidence ratios, across a sweep of the noise; the three answers again in a ball with its stations spread over the sphere, seen in sections by three planes and a shell; and with the rays re-traced in the model by shooting, the derivative by Fermat's principle and the MAP model by Gauss-Newton |
| `waveform/` | two-dimensional acoustic waveform inversion on a finite-difference solver written in C++ and bound with nanobind, with its linearization and its first and second adjoints proved by the check ladder: the MAP model by Newton-CG on the Gauss-Newton and the exact Hessian, Occam's inversion, and the Laplace Gaussian on box means through a randomized-eigendecomposition preconditioner; and the same solver with its grid split across MPI ranks by halo exchange, which gives the serial answers bit for bit |
