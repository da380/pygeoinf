# Finite elements through MFEM: how it uses the library

The [README](README.md) says what the two ways of wrapping MFEM are and what
each measured. This document walks through how they are built on
`pygeoinf2`: a foreign library's function space as a Hilbert space, twice,
and a foreign solve as an operator between the library's own spaces, with
the checks that prove each and the inverse problems run on them. It assumes
the tutorials under `pygeoinf2/tutorials/`, cited by number; tutorial 12 is
the contract and the ladder of checks this app applies to MFEM, and its
closing section points here. Code is quoted from the app's own files.

## 1. The spaces

Each problem is posed on a space appropriate to its forward model and its
answers, and the choice is a modeling decision, stated here as one. Parts A
and C pose theirs on the finite element space under its mass matrix, the
discrete `L^2`: the sensors average the solution of a Poisson equation over
Gaussian windows, so the forward operator is bounded there and no value at a
point is needed. Part E poses its own on `H^2` on the periodic box, because
it observes the solution at points, which needs order above one in two
dimensions; boundedness narrows each choice without making it. Each prior
is a measure of extra regularity on its problem's space: the Matérn
covariance of A and C and part E's Sobolev measure of order two make their
draws smoother without moving the problem to a smoother space, and part E
does nothing relative to `L^2`.

Three spaces, one per arrangement.

**`MfemSpace`** (`src/matrix_space.py`, parts B and C) is a `CoordinateSpace`
whose vectors are `mfem.Vector` objects and whose components are their
degree-of-freedom values. It supplies the contract of tutorial 12: `dim`,
`_key`, the vector algebra on MFEM's own operations (`y.Add(a, x)`,
`x *= a`), `to_components` and `from_components`, and the metric. The metric
is the point: in a finite element space `(u, v) == u^T M v` with `M` the
mass matrix, so the mass matrix is the Gram matrix. It is read out of MFEM
into SciPy once, and `solve_gram` is a sparse factorization of it made once:

```python
def apply_gram(self, c: np.ndarray) -> np.ndarray:
    return self._scipy_mass @ c

def solve_gram(self, c: np.ndarray) -> np.ndarray:
    return self._mass_factorization(np.asarray(c, dtype=float))
```

A homogeneous Dirichlet condition is a subspace, not a special case:
`essential_dofs=` keeps the free degrees of freedom as the components, the
Gram matrix is the mass matrix's free block, and a vector stays the length
MFEM expects, zero on the boundary. `to_components` copies, because a view
into MFEM's memory does not keep its owner alive.

**`MfemHilbertSpace`** (`src/hilbert_space.py`, part A) has no coordinates. It
is the library's `MassWeightedSpace` over `MfemDofSpace`, a bare
`HilbertSpace` of true-dof vectors under the dot product, reweighted by a
mass operator that MFEM applies and inverts:

```python
mass = LinearOperator.self_adjoint(dof_space, mass_apply, traits=Traits.POSITIVE_DEFINITE)
inverse = LinearOperator.self_adjoint(dof_space, mass_solve, traits=Traits.POSITIVE_DEFINITE)
super().__init__(dof_space, mass, mass_solver=inverse)
```

`mass_apply` is MFEM's `Mult` on the mass form, fully or partially
assembled, and `mass_solve` MFEM's conjugate gradients on it. Nothing here
forms a matrix: `LinearOperator.identity(V).matrix()` refuses, every solver
that runs on the space is a Krylov one, and the library takes its
coordinate-free paths throughout (tutorial 2: coordinates are a capability).
`MfemDofSpace` takes a reducing `dot` and a global `dimension`, the two
things a distributed space would supply; that path is not run.

Both say `is_function_space`, so density evaluations refuse on them as on any
discretized function space (tutorial 5).

**The box** (parts D and E) is the library's own space, `box_space` in
`src/common.py`:

```python
Sobolev(
    (scale.box_grid, scale.box_grid),
    scale.box_order,
    scale.length_scale,
    lengths=(1.0, 1.0),
)
```

from `pygeoinf2.symmetric_space.torus`, `H^2` on the unit periodic square,
vectors grid arrays. MFEM is not a space here at all. The order is two
because part E observes `u` at points, which needs order above one in two
dimensions; and a Gram matrix whose weights run from 1 to `4.1e4` is what
makes the adjoint checks of part D able to fail.

The data spaces are `EuclideanSpace`s, a dimension per sensor; the
properties are one total source (A, C) and two disc averages (E).

## 2. The forward code wrapped as an operator

**Forms on the matrix form.** An assembled bilinear form is the Galerkin
matrix of its operator, `a(u, v) == u^T K v` meaning `K == M A_c`, so it goes
straight into `from_matrix` (tutorial 3):

```python
return LinearOperator.from_matrix(
    space, space, space.restrict(form.SpMat()), traits=traits, form="galerkin"
)
```

and the adjoint, `M^-1 K^T` in components, is the library's. An assembled
linear form is a load vector, `b_i == l(phi_i)`: derivative components, so
`LinearFunctional.from_derivative_components(space, load)`, whose
representer is the mass solve `M^-1 b` (tutorial 2). Sensors are linear
forms, stacked into `from_matrix(..., form="galerkin")` by
`operator_from_linear_forms`.

**A PDE solve on the matrix form** (`solver_from_bilinear_form`) is MFEM's
conjugate gradients on MFEM's system, wrapped by `from_callables`. Its
action takes a function's components to a load vector by the mass multiply,
solves, and reads the solution back:

```python
load = space.apply_gram(space.to_components(x))
```

The inverse of `M^-1 K` is `K^-1 M`, and leaving out that multiply gives an
answer smooth, plausible and wrong by a mass matrix. The adjoint is the
solve itself, given only when the caller claims `SELF_ADJOINT` for a
symmetric form; the claim is the caller's and `check_traits` verifies it.
The forward operator of parts A and C is `sensors @ solve`.

**Forms on the Hilbert form.** The operator of a bilinear form is
`M^-1 K`: MFEM's `Mult`, then a mass solve. Its adjoint is `M^-1 K^T`, from
MFEM's `MultTranspose`, so a non-symmetric form gets its true adjoint, which
is neither `K` nor `K^T`. A linear form becomes
`LinearFunctional.from_callables(space, value, representer=representer)`,
the representer a mass solve paid only when asked; the sensor operator's
adjoint is one combination of load vectors and one mass solve, however many
sensors. A bilinear form is taken through MFEM's own `FormSystemMatrix`,
which consumes it, and `_System` keeps the form, the handle and the
operator together, because dropping one while another is in use crashes
rather than raises. Every closure that holds an MFEM solver also holds what
the solver points to (tutorial 12's lessons from MFEM).

**A solve between the library's spaces** (`src/box_operator.py`). The screened
Poisson solve `(1 - s² Laplacian) u = f` on the periodic box is three
operators, each between the box and `EuclideanSpace(dofs)`:

```python
return self.evaluation @ self.solve_at(rtol) @ self.sampling
```

- `sampling`, the field integrated against MFEM's shape functions at MFEM's
  quadrature points, `Q B c`. Its adjoint is the box's type-1 transform of
  `Q^T b`, which gives derivative components, and the inverse Gram factor
  that turns them into a vector of the box:
  `space.from_components(space.solve_gram(derivative))`.
- `solve`, `K^-1` by MFEM, `LinearOperator.self_adjoint(...)`: symmetric in
  the dot product, which on the coefficients is the pairing of a load with a
  solution, `b . u == int f u`.
- `evaluation`, the finite element solution at the box's grid, `P u`. Its
  adjoint applies the box's Gram factor to the vector first, then `P^T`
  with the cell volume.

Each piece is `from_callables` with its hand-written adjoint, and the
whole's adjoint is the composition's. Part E's forward operator is
`X.point_evaluation_operator(points) @ box.operator`.

**What was checked.** Part A: `check_space` on the Hilbert form, `1.0e-15`,
every inner product an MFEM `Mult`; `check_operator` on the convection
form's operator, `1.3e-14`; `check_traits` on the Laplacian, positive
definite on the constrained space. Part B: `check_space` and
`check_coordinates`, `2.5e-15`, and `check_operator` on a form's operator,
`1.3e-14`; the test adds `check_white_noise`. Part D runs the ladder on the
box operator, piece by piece and whole, the probes on the box drawn from
part E's prior with `measure=prior`:

```python
"sampling": check_operator(box.sampling, rng=rng, measure=prior),
"solve": check_operator(box.solve, rng=rng),
"solve traits": check_traits(symmetric, rng=rng),
"evaluation": check_operator(box.evaluation, rng=rng),
"operator": check_operator(box.operator, rng=rng, measure=prior),
```

The transfers hold to rounding, the solve and the whole to MFEM's tolerance
(`8.8e-12`). What failures would have meant: `check_coordinates`, a mass
solve that does not invert the mass multiply, the lumped mass of tutorial
12; `check_operator` on a form, an adjoint taken as `K^T` without the mass
solve; on the box operator, a Gram factor in the wrong place, which the app
shows by building the plain transpose and watching it refused with a defect
of 0.63. The solve's traits are claimed as symmetry alone: a
positive-definite claim implies invertibility, which `check_traits`
verifies on a coordinate space by forming the whole matrix, one MFEM solve
per degree of freedom.

Part D then reads defects rather than judging them,
`check_operator(operator, rtol=np.inf, ...)`, for two studies. The true
adjoint's defect follows MFEM's tolerance down, `8.5e-4` at `1e-6` to
`3.2e-11` at `1e-14`: an operator that is a solve is adjoint only to the
solve's tolerance, which is also the floor below which an outer solver asks
for noise (tutorial 12). And the adjoint the continuum promises, the
operator as its own adjoint, holds only as well as the discretization does,
except on a mesh aligned with the grid; the shortcut would pass a check
there and be wrong everywhere else.

## 3. The priors and the error measures

**A Matérn prior from MFEM's pieces** (parts A and C), `matern_measure`,
its covariance built by the SPDE method: with `A` the operator of the
assembled form of `I - div(Theta grad)` and `S` its MFEM solve, the measure
is given its covariance, its factor and its precision as compositions,
nothing formed:

```python
return GaussianMeasure(
    space,
    covariance=covariance.with_traits(definite),
    covariance_factor=scale * factor,
    precision=((1.0 / scale**2) * (powered @ powered)).with_traits(definite),
)
```

The SPDE is a way of building the covariance out of MFEM solves and
nothing more: the problem stays on the finite element space of section 1,
and the measure is one on that space.

Sampling is the factor applied to white noise, and white noise on a finite
element space has covariance `M^-1` in components, not the identity: MFEM's
white-noise integrator assembles a load with covariance `M`, and one mass
solve turns it into a draw (`white_noise_components` on the matrix form,
`white_noise` on the Hilbert form). Only integer exponents are offered, since
the fractional case needs a rational approximation PyMFEM does not expose;
the field is not stationary within a correlation length of the boundary.

**The box's prior** (parts D and E), `box_prior` in `src/common.py`, is
`space.sobolev_measure(scale.box_prior_order, space.length_scale,
pointwise_std=scale.box_prior_std)` with the scale's order 2 and standard
deviation 1: a covariance `(1 + L^2 lambda)^-2` in `H^2`'s own metric, trace
class there because the order exceeds half the box's dimension, one, so
that its draws are functions of the space with nearly one derivative more
than it asks. The library refuses an order at or below one, and the
`--box-prior-order` option says so. In part D it is the source of probe
fields: white noise on a box is rough in every mode and not what the
operator will meet.

The noise is `GaussianMeasure.from_standard_deviation(codomain, sigma)`, on
the Euclidean data space where an isotropic measure is what it should be.
Part E hardens the noise to its 95% ball with `ambient_ball`, the credible
ellipsoid of an isotropic error, and the prior to the ellipsoid of its own
shape, `prior.weakened_ellipsoid(level=0.95, power=scale.box_set_power)`:
`{f : (Q^-p f, f) <= r^2}` with `r` the radius carrying 95% of the prior.
At `p = 1` it would be the Mahalanobis ellipsoid, `prior.credible_set`,
which the library refuses on a function space because its radius is a
chi-squared quantile on as many degrees of freedom as the grid has and
grows with it; at `p = 0` it is the ball in `H^2`, which forgets the
covariance's shape. The radius has a limit for `p < 1 - d / 2s`, one half
here, and the scale's power, one quarter, is half way. The covariance is
diagonal in the box's Fourier basis, so the ellipsoid's operators are exact
and free. It is still a lossy step: it admits a field spending its whole
budget on one disc.

## 4. The solvers

Parts A and C invert the data-space normal operator on sixteen sensors with
the default conjugate gradients; each application runs MFEM solves inside
the forward operator and the prior, and the data space is too small for
the iteration to matter. Part B runs the library's `CGSolver(rtol=1e-12)`
on the matrix form's operator, every inner product `u^T M v`, against a
direct solve of the Galerkin system: 99 iterations, agreement to `2.0e-14`.

The mass solves differ by arrangement: the matrix form factorizes the mass
matrix once in SciPy; the Hilbert form runs MFEM's conjugate gradients at
`1e-14` per adjoint, well conditioned and short, but a solve each time.

Part E solves in the data space, 600 dimensions, where each application of
`A Q A*` is a forward and an adjoint MFEM solve, and plain conjugate
gradients need more iterations than there are data. The app passes
`SpectralFamilySolver()` to all three estimators: it assembles the
data-space operator once, a column per datum, and diagonalizes it, after
which a posterior sample or a new damping costs two small matrix products.
The README's counts follow from that: 1227, 1203 and 1368 MFEM solves for
the point, the mean and the set, mostly the assembly. It is dense in the
number of data and is used only when passed.

MFEM's solver keeps its work vectors between calls, so `ScreenedPoisson`
serializes its solves with a lock and nothing asks for worker threads.

## 5. The estimators

**Parts A and C, the measure only.** `LinearGaussianInversion(problem,
prior)` gives the posterior mean, and
`inversion.push_forward(property_operator)(data)` the posterior on the
total source over the central square, a Gaussian on one number. Part A runs
the same data through both arrangements and finds the posterior means within
`1.2e-9`. Part C asks the same property at polynomial orders one to three;
the coarsest is the most confident, the discretization acting as a prior
nobody wrote down. Figures `hilbert_form.png` and `matrix_inverse.png`.

**Part E, the three answers** on two disc averages, in order:

- the point: `MinimumNorm(problem.parameterized(factor), level=scale.level,
  solver=solver)` with `factor` the prior's covariance factor, so that the
  estimate is smallest in the prior's norm, the norm the posterior assumes
  (the box's own `H^2` norm is the other honest choice); `for_data(data)`,
  and its `push_forward(averages @ factor)` with `propagated_covariance()`
  and `bias(whiten(truth))`, `whiten` the factor's inverse `L* Q^-1`
  through the prior's precision (tutorials 8 and 14);
  `box_inverse_point.png`;
- the measure: `LinearGaussianInversion(problem, prior, solver=solver)`,
  `sample_pointwise_std`, `pointwise_variance_at` at the discs' centers and
  `affine_map(averages)`; `box_inverse_posterior.png`, prior against
  posterior;
- the set: `BackusGilbertParker(problem, averages, ellipsoid,
  noise=noise_ball, solver=solver)(data)`, a `support_sweep` in four
  directions and its `polytope`; `box_inverse_brackets.png`, each bracket
  against what the prior's ellipsoid admits alone.

None of them knows that MFEM is inside: they see an operator on the box.

## 6. What a reader would change

- **A finite element library of your own as a space**: a `CoordinateSpace`
  with the mass matrix as Gram matrix when the problem is serial and of
  modest size, or a `MassWeightedSpace` whose mass operator and its inverse
  are that library's own calls when it should do every computation. Then the
  ladder of tutorial 12, `check_space`, `check_coordinates`,
  `check_white_noise`, before any operator.
- **Forms as operators**: bilinear forms are Galerkin matrices, linear forms
  derivative components; a solve multiplies by the mass first.
- **A foreign solve between the library's spaces**: write each transfer with
  its adjoint, the transpose between the inverse Gram factor and the Gram
  factor, and compose, as `ScreenedPoisson` does; check each piece and the
  whole with `measure=prior`, and sweep the inner tolerance.
- **The solver**: `SpectralFamilySolver` when the data are a few hundred and
  each application is expensive; the default Krylov solver otherwise.
