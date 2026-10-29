# Review of the spaces layer

Layer: `symmetric_space/` (base, fourier, circle, line, torus, plane, box,
sphere, rays), `spectral_element/` (base, axis, layered, `_points`,
`_cylindrical`, `_transform`), `interval/` (with `radial`), `disk/`, `ball/`
and `plotting/`. Every finding marked "verified" was reproduced on the
current tree (planetmodel 1.3.2) with the snippet given; scratch scripts are
in the session scratchpad, nothing under `pygeoinf2/` was changed.

## Must fix before 2.0

### Identity covariances are still one call away on every function space
`pygeoinf2/symmetric_space/base.py:832` (`invariant_measure`), `:973`
(`heat_measure`), `:1009` (`correlated_measure` and its `_from_correlations`
sibling), `:1200` (`power_measure`), `pygeoinf2/spectral_element/base.py:800`
(`spectral_measure`) — D-190 closed `sobolev_measure` and
`from_standard_deviation`, but the same covariance is accepted, with no
`unsafe=`, through every other constructor: `invariant_measure(np.ones(dim))`
is the identity, `heat_measure(0.0)` is the identity (`heat_symbol` allows a
zero length), `power_measure` with a flat per-degree power is not trace class
(its own docstring says "the dimension-free interpretation is the caller's to
check"), and `spectral_measure(np.ones(dim))` is the identity on a disk, ball
or interval. This is not hypothetical: **tutorial 9 teaches it** —
`tutorials/09_nonlinear.py:109` and `:182` build "white noise on the model
space: equal variance in every mode of `H^2`" by `invariant_measure(np.ones(X.dim),
pointwise_std=0.8)` on a circle `H^2`, which is exactly the "at order zero it
is the identity covariance ... the draws are not functions of the space" case
D-190 calls the cardinal sin, and `tests/test_laplace.py:595` copies it. D-190
listed where the refusal goes rather than deciding these may stay open; the
principle as written ("the library refuses otherwise without `unsafe=True`")
covers them, and `from_standard_deviation`'s refusal is moot while
`invariant_measure(np.ones(dim))` builds the same measure. Verified by running.
Reproduction:
```python
from pygeoinf2.symmetric_space import sphere
S = sphere.Sobolev(16, 2.0, 0.3)
S.sobolev_measure(0.0, 0.3)          # ValueError: refused
S.invariant_measure(np.ones(S.dim))  # accepted, covariance eigenvalues all 1
S.heat_measure(0.0)                  # accepted, covariance eigenvalues all 1
S.power_measure(lambda l: np.ones_like(l))   # accepted; E||x||_L2^2 is 17 at lmax 16, 65 at lmax 64
disk.Sobolev(8, 2.0, 0.3).spectral_measure(np.ones(dim))   # accepted
```
Proposed fix: give these constructors `unsafe=` and refuse at least a flat
spectrum (a multiple of the identity) and `heat_measure(length_scale <= 0)`;
for a callable symbol, walk it past the truncation as
`estimate_truncation_degree` already does and refuse when the per-degree
totals do not decay. Re-pose tutorial 9 on a Sobolev measure (its
conditioning argument can use a preconditioner, as D-191's apps do).

### The ray operator has no order guard
`pygeoinf2/symmetric_space/base.py:2601`, `pygeoinf2/symmetric_space/rays.py:1094` —
`path_integral_operator` refuses a space at or below order `(d-1)/2`, but
`ray_travel_time_operator`, whose derivative *is* that path integral (built
at `rays.py:1171` with `point_evaluation_operator(nodes, unsafe=True)`) and
whose value reads `exp(-m)` at points, takes no `unsafe=` and checks nothing,
so a nonlinear tomography on `L2` of a torus or sphere builds and runs and
returns a grid-scale derivative with no limit. Every caller in the tree
(tests, tutorials 13 and 14) uses order 2, so a guard breaks nothing.
Verified by running.
Reproduction:
```python
from pygeoinf2.symmetric_space import torus
X = torus.Lebesgue((32, 32)); paths = [((1.0, 1.0), (4.0, 2.0))]
X.path_integral_operator(paths)     # ValueError: needs a Sobolev order above 0.5
F = X.ray_travel_time_operator(paths, 1.0)
F(X.zero()), F.derivative(X.zero()) # [3.162...], built; same on sphere.Lebesgue(16)
```
Proposed fix: add `unsafe=` and call `_require_point_evaluation` (the slowness
is read at points; at minimum `_require_path_evaluation`) before tracing.

### The ray operator's linearization is memoized by object identity
`pygeoinf2/symmetric_space/rays.py:1180` — `traced_memo` returns the cached
rays whenever the model is the *same object* as last time, so a model updated
in place (`axpy` is in place by contract) gets the old rays from `F.at(m)`:
a stale value *and* a stale derivative, silently, while `F(m)` is right. The
library's own optimizers copy before stepping, so this bites user code and
any future in-place iterate. Verified by running.
Reproduction:
```python
X = plane.Sobolev((24, 24), 2.0, 0.25, bounds=[(0, 2), (0, 2)])
F = X.ray_travel_time_operator([((0.2, 0.3), (1.8, 1.5)), ((0.3, 1.7), (1.6, 0.2))], 1.0, count=12)
m = X.zero(); F.at(m)
bump = X.project_function(lambda p: 0.3*np.exp(-((p[0]-1)**2 + (p[1]-1)**2)/0.1))
X.axpy(1.0, bump, m)
F.at(m).value            # [2.0, 1.98494]      (the straight rays through zero)
F(m)                     # [1.85266, 1.83664]
F.at(X.copy(m)).value    # [1.85266, 1.83664]; derivative on bump: -0.158 stale vs -0.135
```
Proposed fix: key the memo on the components (`to_components(m).tobytes()`
or a hash of them) as well as identity, or memoize only for the
`second_derivative` call that immediately follows `at`.

### Product and correlated priors cannot be sampled with `n_jobs`
`pygeoinf2/symmetric_space/base.py:2815` (`SpectralBlockLinearOperator.__new__`),
and through `Layered.sobolev_measure` (`pygeoinf2/spectral_element/layered.py:691`)
the block constructors of `pygeoinf2/algebra/direct_sum.py:521, 646, 720, 812`
— these `__new__` methods require positional arguments, so unpickling
fails, and `samples(n, n_jobs=k>1)` (processes, as the principles ask) dies
with `BrokenProcessPool`. That covers every layered prior
(`Layered.sobolev_measure` → `from_product` → `BlockDiagonalOperator`), every
`GaussianMeasure.from_product`, every `correlated_measure`, and every
posterior built from them (`sample_pointwise_std(n_jobs=)` too). D-181 fixed
the same defect in `DirectSum.__new__` only. The tomography app is shielded
only because it re-seats its prior as one diagonal operator; D-195 says that
re-seating goes when the app is next touched, which would break its
`n_jobs` path. Verified by running.
Reproduction:
```python
X = interval.Layered([0, .5, 1], 8, order=1.0, length_scale=0.2)
pickle.loads(pickle.dumps(X.sobolev_measure(1.0)))
# TypeError: BlockDiagonalOperator.__new__() missing 1 required positional argument: 'operators'
X.sobolev_measure(1.0).samples(2, n_jobs=2)        # BrokenProcessPool (same on disk.Layered)
T = torus.Sobolev((8, 8), 2.0, 0.3)
mu = T.correlated_measure_from_correlations([T.sobolev_symbol(-2, 0.3)]*2, np.array([[1, .5], [.5, 1]]))
mu.samples(2, n_jobs=2)   # BrokenProcessPool; cloudpickle: SpectralBlockLinearOperator.__new__() missing 'space' and 'slices'
```
(The posterior of `LinearGaussianInversion` with the layered disk prior fails
the same way.) Proposed fix: `__new__(cls, *args, **kwargs)` dispatching only
when arguments are given, as D-181 did for `DirectSum`; add a process-pool
sampling test over a layered prior and a correlated measure.

### A bounded box evaluates points outside its domain by wrapping them
`pygeoinf2/symmetric_space/box.py:136` (`_to_enclosing`), `:161`, `:165` — `Box`,
`Line` and `Plane` accept any coordinate in `basis_at`, `prepare_points`,
`dirac` and `point_evaluation_operator`, move it into the periodic enclosure
and wrap it, so a station just outside `[lower, upper]` silently reads the
field at some other point of the domain. The module itself says the padding
is "a numerical device, not part of the domain" (`_separation`), and every
spectral-element space refuses such a point with `ValueError`. Verified by
running.
Reproduction:
```python
X = line.Sobolev(64, 1.0, 0.05, lower=0.0, upper=1.0)   # padding 0.1, period 1.2
x = X.sobolev_measure(1.0, 0.05).sample(rng=np.random.default_rng(0))
X.point_evaluation_operator([0.3, 1.5, -0.9])(x)   # [2.059737 2.059737 2.059737]
interval.Sobolev(16, 1.0, 0.1).point_evaluation_operator([1.5])   # ValueError
```
Proposed fix: refuse points outside `bounds` (with a relative tolerance) in
`Box.prepare_points`/`basis_at`; the internal quadratures that may step into
the padding (ball averages near an edge) wrap explicitly.

### One choice, two keywords: `dense=` on symmetric spaces, `form=` on the others
`pygeoinf2/symmetric_space/base.py:1854` (`dense: bool = False`, default
matrix-free), `pygeoinf2/spectral_element/base.py:689` and
`pygeoinf2/spectral_element/layered.py:510` (`form="auto"|"dense"|"local"`,
default dense under 256 MB) — the same decision on sibling spaces is spelled
two ways with opposite defaults, and `form=` everywhere else in the library
(36 signatures: `from_matrix`, `matrix`, `random_diagonal`, the
preconditioners) means `"components"` versus `"galerkin"` (D-4, D-25), so
`point_evaluation_operator(points, form="galerkin")` reads plausibly and
raises. This contradicts D-183's naming, not its substance (the memory budget
and the local route are right). Only seven call sites in the tree use either
spelling. Read only.
Reproduction: compare the two signatures; `grep -rn 'form: Literal' pygeoinf2`.
Proposed fix: one keyword for both families that is not `form`, e.g.
`dense: bool | None = None` with `None` the family's automatic choice; while
at it say in one place which default each family takes and why.

### `gi.Interval` is a factory for a Fourier box and collides with `gi.interval.Interval`
`pygeoinf2/symmetric_space/box.py:400`, `pygeoinf2/__init__.py:150` — the
top level re-exports `Interval`, a *function* returning a `Box` (so
`isinstance(x, gi.Interval)` raises `TypeError`, against the
"classes rather than factory functions" rule the `symmetric_space` docstring
states), while `gi.interval.Interval` is the spectral-element class with a
different first argument (modes, not grid points). `box.Interval` also
duplicates `line.Line`, which builds the same geometry as a class. The flat
`gi.Lebesgue` and `gi.Sobolev` are the N-dimensional periodic box's, one of
eight geometry pairs of those names; and `gi.Ball` (a convex set) sits beside
`gi.ball.Ball` (a space). Nothing in examples, tutorials or mini-apps uses the
top-level space names; `box.Interval` is used in one test file. Verified by
running.
Reproduction:
```python
gi.Interval(32)                 # Box((32,), bounds=((0.0, 1.0),), Lebesgue)
isinstance(x, gi.Interval)      # TypeError: isinstance() arg 2 must be a type
gi.interval.Interval            # <class 'pygeoinf2.interval.space.Interval'>
```
Proposed fix: delete `box.Interval` (use `line.Line`) and drop `Interval`,
`Lebesgue`, `Sobolev` from the top-level namespace, leaving geometry-qualified
imports; decide whether `Box`/`PeriodicBox` stay flat.

### `spectral_element.ShellTransform` is public but scheduled for deletion
`pygeoinf2/spectral_element/__init__.py:79` (`__all__`),
`pygeoinf2/spectral_element/_transform.py:28` — CURRENT_STATE §10 plans
"the pin to `>=1.4` and the deletion of `spectral_element/_transform.py`"
once planetmodel 1.4 ships `ShellTransform`; exporting it now ships a public
name that will be removed after 2.0, against "nothing provisional". Only
`ball/space.py` (privately) and `tests/test_ball.py` use it. Read only.
Reproduction: `from pygeoinf2.spectral_element import ShellTransform` works.
Proposed fix: drop it from `__all__` and the package docstring; the test
imports it from `_transform`.

## Should fix

### Equal sphere spaces whose vectors are incompatible
`pygeoinf2/symmetric_space/sphere.py:560` — `_key` omits `extend`, so
`Lebesgue(8) == Lebesgue(8, extend=True)` (and the hashes agree) though the
fields are `(18, 18)` against `(19, 19)` and every operator of one raises on
the other's vectors; meanwhile `shares_vectors_with` says False. The class
docstring (`:355`) says a Driscoll-Healy and a Gauss-Legendre space are "the
same space on either", but `_key` includes the grid kind and they compare
unequal — the docstring and the key disagree in both directions. Verified by
running.
Reproduction:
```python
a, b = sphere.Lebesgue(8), sphere.Lebesgue(8, extend=True)
a == b, hash(a) == hash(b), a.shares_vectors_with(b)   # True True False
a.spectral_operator(np.ones(a.dim))(b.from_components(np.ones(b.dim)))
# ValueError: A field has shape (18, 18), got (19, 19).
sphere.Lebesgue(8) == sphere.Lebesgue(8, grid="GLQ")  # False
```
Proposed fix: put `extend` in `_key` (and say "same components, a different
space" in both docstrings), or accept either shape in `_core_values`.

### Symmetric Lebesgue spaces forget the length scale; `_key` still counts it
`pygeoinf2/symmetric_space/fourier.py:306`, `:1174`, `pygeoinf2/symmetric_space/sphere.py:2324` —
`_rebuilt` builds a `Lebesgue` without the length scale, so
`X.with_order(0).with_order(2)` returns a space with `length_scale` 1.0, not
`X` (the spectral-element spaces keep it); and `_key` includes the length
scale at order zero, where it is "ignored", so two Lebesgue spaces with the
same metric compare unequal. The length scale is not ignored at order zero
either: it sets `_path_nodes`' default node count. Verified by running.
Reproduction:
```python
X = sphere.Sobolev(8, 2.0, 0.1); X.with_order(0.0).with_order(2.0).length_scale   # 1.0; == X: False
fourier.Lebesgue((8,)) == fourier.PeriodicBox((8,), order=0.0, length_scale=2.0)  # False, same metric
interval.Sobolev(12, 2.0, 0.1).with_order(0.0).with_order(2.0)                   # == original: True
```
Proposed fix: carry the length scale through `Lebesgue` (it is a property of
the space family) and leave it out of `_key` when the order is zero.

### A box's reference point is not a grid point, so `pointwise_std=` misses
`pygeoinf2/symmetric_space/fourier.py:394`, docstring at
`pygeoinf2/symmetric_space/base.py:739` — `PeriodicBox.reference_point` is
the origin, which `Box` inherits in *domain* coordinates: for bounds not
starting at a grid line it is off the grid (and possibly outside the domain),
though `reference_point` promises "every geometry here returns a grid point,
which `pointwise_variance` relies on". On an even axis the variance between
grid points dips (the `pointwise_variance` docstring explains why), so
`invariant_measure(..., pointwise_std=s)` calibrates to the wrong number.
Verified by running.
Reproduction:
```python
X = plane.Sobolev((6, 4), 2.0, 0.1, bounds=[(0.3, 1.3), (0.3, 1.3)])
mu = X.invariant_measure(X.sobolev_symbol(-2.0, 0.01), pointwise_std=1.0)
X.pointwise_variance_at(mu, grid_points)   # 1.0284 at every grid point, not 1
# line.Sobolev(16, 1.0, 0.2, lower=5, upper=6): reference 0.0, error 8e-4
```
Proposed fix: `Box.reference_point` returns its first grid node
(`tuple(axis[0] for axis in grid_axes)` moved inside the domain).

### A radial prior calibrated pointwise has no limit at the centre below order 3/2
`pygeoinf2/interval/radial.py:132`, `pygeoinf2/spectral_element/base.py:800`, `:872` —
on a whole-ball radial space the trace-class gate is order `> 1/2`
(`_eigenvalue_dimension = 1`, D-190, correct for draws in the space), but
point values at the centre need `> 3/2` (`point_evaluation_order` says so).
Between the two, `sobolev_measure(order, pointwise_std=...)` is accepted and
divides by a raw standard deviation that grows without bound at the centre
node, so the calibrated prior changes with the mesh there. Verified by
running (the raw variance; the calibration follows from the formula).
Reproduction:
```python
for modes in (16, 32, 64, 128):
    X = radial.Lebesgue(modes, length_scale=0.1)
    v = X.pointwise_variance(X.eigenvalues ** -1.0)
    # at r = 0: 144, 343, 757, 1590;  at r = 0.5: 1.36, 1.47, 1.53, 1.56
```
Proposed fix: a pointwise calibration is a statement about point values, so
refuse `pointwise_std=` unless the measure's order exceeds
`point_evaluation_order()` over the domain (with `unsafe=`).

### Ball and disk meshes depend on the unit of length
`pygeoinf2/ball/space.py:151`, `pygeoinf2/disk/space.py:165` — D-199 added
the `1 + 1e-9` margin to the default element length in `axis.py:123` so that
the same geometry in other units gets the same mesh, but the ball and the
disk compute their own default and lack it; 16 of 120 unit changes tried give
one element more or fewer (and so another basis, and other seeded draws),
none for the radial profile. Verified by running.
Reproduction:
```python
ball.Lebesgue(2, radius=1.0, length_scale=0.3).grid_shape[0]          # 49
ball.Lebesgue(2, radius=3480.0, length_scale=0.3*3480).grid_shape[0]  # 45 (disk: same)
```
Proposed fix: apply the same margin in both, ideally in one shared helper
for the default element length.

### Exact cap averages are always a dense matrix
`pygeoinf2/symmetric_space/sphere.py:2044` — without `count=`, the sphere's
`geodesic_ball_average_operator` assembles a `(centers, dim)` matrix whatever
`dense=` says (`dense=False` is ignored, its docstring notwithstanding), 266
MB at 2000 caps and degree 128. The exact functional is a zonal filter
followed by evaluation, so a matrix-free form exists that agrees to the
NUFFT's accuracy and passes the ladder. Verified by running.
Reproduction:
```python
S = sphere.Sobolev(64, 1.0, 0.2, radius=2.0); A = S.geodesic_ball_average_operator(centers, 0.3)
type(A).__name__   # MatrixLinearOperator (500, 4225)
B = S.with_order(0).point_evaluation_operator(centers, unsafe=True) @ S.with_order(0).spectral_operator(cap_filter) @ S.order_inclusion_operator(S.with_order(0))
max |A x - B x| = 1.3e-10; check_operator(B) worst 1.3e-14
```
Proposed fix: return the filtered evaluation by default and the matrix under
`dense=True`.

### The base `Layered` accepts disks but cannot draw them or take their jumps
`pygeoinf2/spectral_element/layered.py:215`, `:565`/`:602`,
`pygeoinf2/plotting/interval.py:85`/`:107` — the base class "takes layers
already built, of any one geometry", and accepts disks, but `jump_operator`
treats every non-axis layer as a ball (`(lat, lon)` pairs) and `plot`
dispatches every non-axis geometry to `plot_shell`; the module docstring, the
`TypeError` message and `geometry`'s docstring still list only intervals,
radial profiles and balls (D-175 left this "not done"). Verified by running.
Reproduction:
```python
L = spectral_element.Layered([disk.Sobolev(3, 2.0, .3, radius=.5), disk.Sobolev(3, 2.0, .3, inner_radius=.5)])
L.jump_operator(0, directions=[0.0, 90.0])  # TypeError: cannot unpack non-iterable float object
plotting.plot(L, x)                          # TypeError: Only layers that are balls have shells and sections.
```
Proposed fix: have the base constructor return the geometry's own `Layered`
(each geometry registering its class), or move the interface points into a
per-geometry hook; update the messages.

### `plot_network` marks the second node of a polyline as the receiver
`pygeoinf2/plotting/base.py:245` — receivers are `p[1] for p in paths`, right
for the `(start, end)` pairs of a sphere or a box, wrong for the disk's
polylines (a traced ray), whose receiver is the last node. Verified by
running.
Reproduction:
```python
ray = [(1.0, 0.0), (0.8, 20.0), (0.7, 45.0), (0.8, 70.0), (1.0, 90.0)]
ax, a = plotting.plot_network(disk.Lebesgue(4, length_scale=0.3), [ray])
a["receivers"].get_offsets()   # [[0.752, 0.274]], the receiver is at (0, 1)
```
Proposed fix: `p[-1]`.

### On a box with unequal periods a degree is not a spatial scale
`pygeoinf2/symmetric_space/fourier.py:182` — `degrees` is the magnitude of the
*integer* wavevector, so on a box whose periods differ it groups wavelengths
that differ by the ratio of the periods; the docstring's "components sharing
one are the modes of a common spatial scale" is false there, and
`power_measure`, `power_spectrum`, `coefficient_operator(lmax=)` and
`with_degree` ("an isotropic band limit") inherit it. Every non-square
`Plane` is such a box. Verified by running.
Reproduction:
```python
X = torus.Lebesgue((16, 16), lengths=(1.0, 10.0))
# wavevector (1, 0): degree 1, wavenumber 6.283; (0, 1): degree 1, wavenumber 0.628
# power_measure at degree 1 puts power on wavelengths 1 and 10
```
Proposed fix: either define a degree by the physical wavenumber in units of
the longest period, or say plainly that degrees are in grid units and are a
spatial scale only when the periods are equal.

### Degree transfer accepts boxes on different domains
`pygeoinf2/symmetric_space/fourier.py:968` — the check compares periods only,
so a `Box` on `[0, 1]` transfers into one on `[5, 6]`, although the docstring
promises `ValueError` "if the two boxes are not the same domain". Verified by
running.
Reproduction: `line.Lebesgue(16).degree_transfer_operator(line.Lebesgue(32, lower=5.0, upper=6.0))` builds.
Proposed fix: compare the coordinate maps (bounds and padding) for a `Box`.

### The spectral-element spaces offer no domain quadrature, so apps reach into planetmodel
`pygeoinf2/spectral_element/axis.py:337`, `pygeoinf2/ball/space.py:517`,
`pygeoinf2/disk/space.py:497` — since D-183 the integral functionals read
through the components and "the exact integral of such an array is a
quadrature the caller takes", but no public quadrature is offered: the
tomography app builds it from `mantle.basis[0].weights()` (planetmodel's
`SpectralBasis`) and `mantle.grid.weights` in a ball versus
`mantle.grid.weight` in a disk (`miniapps/tomography/src/problem.py:464-468`,
`bent.py:568`), and `grid` is a planetmodel `AngularGrid` on a ball, a
`RingTransform` on a disk and a string on a sphere. Read only.
Reproduction: see the cited app lines (the planet-density app builds its
own Gauss-Legendre rule against `basis_matrix` for the same reason).
Proposed fix: a `domain_weights()` (or `domain_quadrature()`) on every
spectral-element space and `Layered`, the array a caller multiplies grid
values by, so no app touches planetmodel objects.

### The planetmodel pin is open while the disk subclasses its private state
`pyproject.toml:33` (`planetmodel>=1.3`),
`pygeoinf2/spectral_element/_cylindrical.py:164-216` — `CylindricalOperatorFamily`
sets and relies on `RadialOperatorFamily`'s private attributes (`_ball`,
`_degree_zero_only`, `_M`, `_Kh`, `_Kr`, `_q`, `_vals`, `_pairs`, `_pencils`,
`_axis_*`), as D-175 recorded, with no upper bound on the version; the 1.4
release that §10 plans could rename them and break `pygeoinf2.disk` for
users of 2.0 who upgrade planetmodel. Read only.
Reproduction: `grep -n 'self\._' pygeoinf2/spectral_element/_cylindrical.py`.
Proposed fix: pin `planetmodel>=1.3,<1.4` for 2.0 (raising it when 1.4 is
checked), or move the `r dr` measure into planetmodel 1.4 so nothing private
is touched.

### `import pygeoinf2` loads planetmodel, contrary to "imported on demand"
`pygeoinf2/__init__.py:291`, `pygeoinf2/plotting/__init__.py:38` — D-161 and
§2 say `interval`, `ball` and `disk` are imported on demand, but `plotting`
imports their renderers at import time, which imports the packages and
planetmodel: 85 ms of a 0.37–0.41 s `import pygeoinf2` (D-157 measured
0.22 s after its lazy imports). Verified by running.
Reproduction: `python -X importtime -c "import pygeoinf2"` shows
`pygeoinf2.plotting.interval` → `pygeoinf2.spectral_element` → `planetmodel`
at 85 ms cumulative; `'planetmodel' in sys.modules` is True.
Proposed fix: register the spectral-element renderers from the geometry
packages (or lazily on the first `plot` call that misses), so plotting does
not import them.

### The plotting helpers draw serially, with no `n_jobs`
`pygeoinf2/plotting/distributions.py:80` — `moments` (and so
`plot_densities` and `plot_corner`) draws 20 000 samples by a list
comprehension over `measure.sample`, bypassing `measure.samples(n, rng=,
n_jobs=)`; none of the three takes `n_jobs`. (§7's point that a mixture's
marginals are closed form is separate.) Read only.
Reproduction: read `moments`.
Proposed fix: call `measure.samples(samples, rng=rng, n_jobs=n_jobs)` and add
`n_jobs=` to the three.

### A docstring still says ray tracing is out of scope
`pygeoinf2/symmetric_space/base.py:2454` — `path_integral_operator`'s `weight`
argument ends "ray tracing, where the path itself depends on the field, is out
of scope", though `ray_travel_time_operator` sits on the same class. Read only.
Reproduction: read the docstring.
Proposed fix: point to `ray_travel_time_operator` instead.

## Could fix

### The sphere's transform threshold is too high at large degree
`pygeoinf2/symmetric_space/sphere.py:62` — `_TRANSFORM_MIN_POINTS = 200`
fixes the crossover in points alone, but the direct route costs
`n × lmax²` and the transform about `lmax² log lmax`, so the crossover is
nearly constant and lower: measured 50–100 points from degree 32 to 128.
At 199 points the "matrix-free" route rebuilds a basis matrix on every
application and is 4–6 times slower than the transform. Verified by running.
Reproduction (ms per evaluation, direct / transform): lmax 64, 199 points
10.3 / 1.8; lmax 128, 199 points 33.9 / 8.9; lmax 128, 50 points 6.4 / 8.1.
Proposed fix: lower the point threshold to about 64, or decide by
`n × dim` against `dim log dim`.

### Matrix-free point evaluation on symmetric spaces has no component action
`pygeoinf2/symmetric_space/base.py:1925` — built by
`from_derivative_callables`, it has no `components_action`, so a data-space
normal operator `E C E*` round-trips through fields: two analyses and two
syntheses per application on a sphere where one of each suffices. The
spectral-element local form supplies both actions for exactly this reason
(D-183). Verified by running (4.3 ms per application at degree 64, 2000
points). Reproduction: count `to_components`/`from_components` calls in one
application of `E @ mu.covariance @ E.adjoint`.
Proposed fix: give the operator `components_action` and
`components_adjoint_action` built from `evaluate`/`accumulate` on components.

### Plotting keywords and defaults differ between renderers
`pygeoinf2/plotting/distributions.py:512`, `pygeoinf2/plotting/fourier.py:32-36`,
`pygeoinf2/plotting/sphere.py:76`, `pygeoinf2/plotting/sets.py:50`,
`pygeoinf2/plotting/ball.py:439`/`:500` — `plot_corner(colormap=)` where every
other plotter takes `cmap=`; the field `plot` turns its colorbar off by
default on a sphere and on by default on a box, a disk and a section, and
uses `viridis` on a box against `RdBu` elsewhere, and has no `title=` on a
box; `plot_set(route=)` is the last `route=` in the library, where a choice
of algorithm is `method=` elsewhere; `plot_section_points` requires `ax` and
returns `(scatter, mask)` where the point plotters return `(axes,
collection)`. Read only.
Reproduction: the cited signatures.
Proposed fix: `cmap=` everywhere, one colorbar default, `method=` on
`plot_set`, and `(axes, artist)` returns.

### Methods whose only caller is a test, and a `grid` of three types
`pygeoinf2/symmetric_space/fourier.py:1086` (`pointwise_multiply`, which
duplicates `multiply` and whose docstring contradicts `multiply`'s on
aliasing), `pygeoinf2/symmetric_space/base.py:308` (`degree_multiplicity`),
`pygeoinf2/symmetric_space/sphere.py:2030` (`spherical_cap_average`, a
one-line wrapper of `spherical_cap_integral(normalize=True)`) are used only
in tests; `grid` is a string on a sphere, an `AngularGrid` on a ball and a
`RingTransform` on a disk; `tests/test_observation.py:380, 407` read
`space._packing[1]` where `space.degrees` is public. Read only.
Reproduction: `grep -rnw pointwise_multiply pygeoinf2`.
Proposed fix: drop the three methods; name the grid attributes by what they
hold.

### `sobolev_measure(order, scale)` names its length `scale`
`pygeoinf2/symmetric_space/base.py:687`, `:920` — the constructors and
`heat_measure`/`heat_symbol` say `length_scale` (the `PeriodicBox` docstring
explains why `scale` is avoided: it is the vector-scaling operation), while
`sobolev_measure` and `sobolev_symbol` call the same length `scale`. Both are
positional-only, so the rename costs no caller anything. Read only.
Reproduction: the two signatures.
Proposed fix: rename the parameter to `length_scale` in both.

### No heat measure on the spectral-element spaces
`pygeoinf2/spectral_element/base.py` — symmetric spaces have `heat_measure`;
the spectral-element spaces do not, and their tests build it as
`spectral_measure(exp(-t (λ - 1)))` (D-175, "not done"). Read only.
Reproduction: `hasattr(disk.Sobolev(4, 2.0, .3), "heat_measure")` is False.
Proposed fix: `heat_measure(length_scale, ...)` on the base class, with the
same `length_scale > 0` rule as the identity finding above asks.

### The identity refusal raises two exception classes
`pygeoinf2/probability/gaussian.py` (`from_standard_deviation`) and
`pygeoinf2/symmetric_space/base.py:52` — the same fault, a covariance that is
not trace class on a function space, is `TypeError` from
`from_standard_deviation` and `ValueError` from `sobolev_measure`. Verified by
running. Reproduction: call both on `sphere.Sobolev(8, 2.0, 0.3)`.
Proposed fix: `ValueError` in both (the argument's value is at fault).

### CURRENT_STATE §7's point-evaluation entry is out of date
`pygeoinf2/CURRENT_STATE.md` §7, "Findings from the tomography mini-app's
build still open" — it says `point_evaluation_operator` always builds its
dense matrix and asks for a matrix-free form above some cost; D-183 built
exactly that (`form="auto"` goes local above 256 MB). The tomography app's
ray-by-ray `path_integral_operator` (`miniapps/tomography/src/problem.py`),
whose docstring still says the composed form "would be gigabytes", is now a
workaround for a gap that no longer exists. Read only.
Proposed fix: close the §7 entry; let the app use its `composed_operator`.

## What is sound

- **The ladder on every geometry.** `check_space` and `check_coordinates`
  (probes from a Sobolev prior) pass on 15 spaces: sphere (DH and GLQ),
  circle, torus with unequal periods, line, plane, a 3-D box, interval,
  radial ball and shell, disk, annulus, ball, and layered intervals and
  disks; `check_white_noise` passes on sphere, plane, disk, ball and a
  layered interval.
- **Operators.** `check_operator` passes (worst defect 1e-13 or better) on
  point evaluation in every form (NUFFT and dense on the symmetric spaces,
  dense and local on the spectral-element ones, agreeing to 1e-13 and 1e-16),
  on interfaces with `side=`; the lifted multiplication operator and its
  Lebesgue self-adjointness (`check_traits`, including the centre of a radial
  ball); order inclusions; coefficient and synthesis operators, including the
  band past the truncation; degree transfer; spectral projections;
  path integrals and averages (matrix-free and dense); cap averages, exact and
  by quadrature; the flexure operator with varying coefficients on a sphere
  and a torus; box derivatives and support projections; the correlated
  measure's factor; layered jumps in disks and balls. The inverse flexure
  operator meets its solver tolerance only, as it should.
- **Refusals that are there.** `sobolev_measure` refuses at or below `d/2` on
  every geometry, including the radial profile's eigenvalue dimension and
  `Layered`; point evaluation and Diracs refuse at or below
  `point_evaluation_order`, point-dependent on the radial space; path
  integrals refuse at or below `(d-1)/2`.
- **Orders and length scales mean the same thing across families.** The
  pointwise variance of an order-2 measure with `L = 0.1` is `1/(4L) = 2.5`
  on both the Fourier line and the spectral-element interval, and
  `3/(16L) = 1.875` on both when hosted on `H^1`; a plane and a disk agree
  with `1/(4π L²)` within the disk's truncation.
- **Integrals.** Integral functionals return the domain volume for a fitted
  constant on intervals, radial profiles, disks, annuli, balls and layered
  spaces.
- **D-195 holds.** `Layered.sobolev_measure` seats its covariance, factor and
  precision on the layered space, and the posterior keeps
  `sample_pointwise_std`. Process-parallel draws work for a single disk, ball,
  interval or sphere measure.
- **Tests.** `test_disk`, `test_ball`, `test_layered`,
  `test_spectral_element_points`, `test_spectral_element_plotting`,
  `test_rays`, `test_interval`, `test_radial`, `test_fourier_spaces`,
  `test_bounded_and_sphere`, `test_space_parity` and `test_plotting` pass.
- **No history or decision numbers** in the layer's code, docstrings or
  comments (one stale sentence noted above).

## Fixed

Package E, 2026-10-29. Tests named by file and class under `pygeoinf2/tests/`.

**Must fix**

- **The identity covariance**: package A.
- **The ray operator's order guard.** `symmetric_space/rays.py`
  `ray_travel_time_operator(unsafe=)` calls `_require_path_evaluation`, the
  guard its derivative, a path integral, applies; the method on
  `SymmetricSpace` passes `unsafe=` through. Test:
  `test_rays.py::TestRefusals::test_a_space_too_rough_for_a_path_integral_is_refused`.
- **The ray memo keyed on identity.** The memo of traced rays is keyed on
  the model's component array, compared by value. Test:
  `test_rays.py::TestAModelUpdatedInPlace` (`axpy` in place, then `F.at(m)`:
  value and derivative agree with a fresh copy and with `F(m)`).
- **Product and correlated priors cannot be sampled with `n_jobs`**: the
  `__new__` fix recorded in `algebra.md`, and
  `SpectralBlockLinearOperator.__new__`; tests there.
- **A bounded box wraps points outside its domain.**
  `symmetric_space/box.py`: `_require_inside` refuses a point outside the
  bounds (a relative `1e-12` allowed) with the spectral-element spaces'
  message shape, "Every point must lie in the domain, ...; one is at ...",
  from `basis_at`, `prepare_points` (both its paths), and so `dirac`,
  `evaluate`, `point_evaluation_operator` on both routes; `basis_matrix`
  reads the prepared points rather than converting twice. What steps into
  the padding on purpose does so unchecked: the ray tracer
  (`_Chart.evaluation_points`, `rays.py`) and the finite-difference
  gradient (`_prepared_nearby`, `fourier.py`). A bounded box no longer
  wraps its quadrature nodes (`Box._wrap`), which had moved path and ball
  nodes out of `[lower, upper]` on any box not starting at zero, and
  `Box.geodesic_ball_quadrature` refuses a ball that leaves the domain.
  Tests: `test_bounded_and_sphere.py::TestBoundedDomains::test_a_point_outside_the_domain_is_refused`
  (line, plane, box; every route), `::test_a_ball_leaving_the_domain_is_refused`,
  `::test_a_domain_away_from_the_origin_evaluates_inside_it`; the
  spectral-element side in
  `test_spectral_element_points.py::TestRefusals::test_a_point_outside_the_domain`.
- **One keyword for point evaluation.** `route="auto"|"dense"|"local"` on
  `SymmetricSpace.point_evaluation_operator` (auto is the local, matrix-free
  route there), `SpectralElementSpace.point_evaluation_operator` and
  `spectral_element.Layered.point_evaluation_operator` (auto is dense under
  256 MB there; `_evaluation_route`, `_EVALUATION_ROUTES`); `dense=` and
  `form=` gone from it; the symmetric docstring says which default each
  family takes and why. Callers updated: `tests/test_normal.py`,
  `tests/test_parallel_knobs.py`, `tests/test_spectral_element_points.py`,
  `tutorials/03_operators.py` and its notebook (regenerated),
  `miniapps/point_data/src/problem.py` and its `IMPLEMENTATION.md`. The
  path and ball operators keep `dense=`, which is not this choice. Tests:
  `test_fourier_spaces.py::TestThePointEvaluationRoute`, and the renamed
  spectral-element tests (`route=` throughout, an unknown route refused).
- **`gi.Interval` and the flat space names.** `Box`, `Interval`,
  `Lebesgue`, `PeriodicBox` and `Sobolev` are no longer re-exported by
  `pygeoinf2/__init__.py` (whose docstring says spaces are reached by
  geometry); they live in `symmetric_space`. No caller used the flat names.
  Test: `test_code_practice.py::TestSpacesAreReachedByGeometry`.
- **`ShellTransform` public.** Dropped from `spectral_element.__all__`, its
  import and its docstring; `tests/test_ball.py` imports it from
  `_transform`. Test: `test_code_practice.py::TestSpacesAreReachedByGeometry::test_the_shell_transform_is_not_public`.

**Should fix**

- **Equal sphere spaces with incompatible vectors.** `sphere.py` `_key`
  includes `extend`; the class docstring and those of `sampling` and
  `extend` say the spaces differ (they claimed one space, though `sampling`
  and the grid kind were already in the key). Test:
  `test_sphere_transform.py::TestExtendedGrid::test_the_shape_and_the_angles_close_the_grid`
  (updated: unequal, unequal hashes, no shared vectors).
- **The length scale in a Lebesgue space's key.** Left out of `_key` at
  order zero on the periodic box, the bounded box and the sphere. Test:
  `test_fourier_spaces.py::TestALebesgueSpaceHasNoLengthScale`. (The
  carrying of the length scale through `Lebesgue` is skipped, below.)
- **A box's reference point.** `Box.reference_point` is the grid node
  nearest the lower corner inside the domain. Test:
  `test_bounded_and_sphere.py::TestABoxsReferencePointIsAGridPointOfTheDomain`
  (a pointwise calibration of 1 holds at the grid points of a plane on
  `[0.3, 1.3]^2`).
- **Ball and disk meshes depend on the unit.** One helper,
  `spectral_element/base.py::_default_element_length`, applies the `1e-9`
  margin, used by `axis.py`, `ball/space.py` and `disk/space.py`. The mesh
  that changes is the unit-radius one, which took the extra element (49
  nodes against 45 at `L = 0.3`): every default ball and disk of radius one
  gets the mesh exact arithmetic gives, so seeded draws on them change.
  Tests: `test_ball.py::TestTheMeshDoesNotDependOnTheUnit`,
  `test_disk.py::TestTheMeshDoesNotDependOnTheUnit`.
- **Exact cap averages always dense.** `sphere.py`
  `geodesic_ball_average_operator` applies the zonal filter and evaluates at
  the centers unless `dense=True`, which assembles the closed-form rows
  (`_cap_weights` shared with `cap_integral_components`). Test:
  `test_observation.py::TestExactCapAveragesAreMatrixFree` (5 and 250
  centers, both transform routes; integrals too).
- **`plot_network`'s receivers.** `p[-1]`, each path's last node; docstring
  says polylines. Test: `test_spectral_element_plotting.py::TestANetworkOfPolylines`.
- **A degree on a box with unequal periods.** The docstrings of `degrees`
  and `with_degree` say a degree is in grid units and a spatial scale only
  when the periods are equal.
- **Degree transfer across domains.** `_domain_key` on `PeriodicBox` (the
  periods) and `Box` (bounds and padding); `degree_transfer_operator`
  compares them. Test: `test_bounded_and_sphere.py::TestDegreeTransferNeedsOneDomain`.
- **The plotting helpers draw serially.** `moments`, `plot_densities` and
  `plot_corner` take `n_jobs=`, drawing through `measure.samples` (and
  passing it to `as_multivariate_normal` for a Gaussian). Test:
  `test_plotting.py::TestTheDistributionPlottersTakeWorkers`.
- **The ray-tracing docstring.** `path_integral_operator`'s `weight`
  points to `ray_travel_time_operator`.
- **`plot_set` (from the examples review).** `n_jobs=` for the support
  route's directions, through a set's own `support_values` or
  `support_sweep` when it has one (warm starts, results kept), otherwise
  `parallel_map`; `color=` becomes `colors=` for the filled contours of
  the level and membership routes, so a polytope is drawn in its color.
  Test: `test_plotting.py::TestPlotSetHonorsItsKnobs`.

## Skipped

- **A radial prior calibrated pointwise below order 3/2**: the refusal
  belongs in `spectral_measure`/`sobolev_measure`, the constructors
  package A owns.
- **The base `Layered` accepting disks**: whether the base constructor
  dispatches to the geometry's own class or takes a per-geometry hook for
  interface points and drawing is a design choice.
- **A domain quadrature on the spectral-element spaces**: a new public
  method.
- **The planetmodel pin**: `pyproject.toml`, outside the package;
  `planetmodel>=1.3,<1.4` until 1.4 is checked is the reviewer's proposal.
- **`import pygeoinf2` loads planetmodel**: not local. The top-level
  `__init__` imports `ball`, `disk`, `interval` and `spectral_element`
  eagerly as well as `plotting`, so on-demand loading needs a module
  `__getattr__` there and lazy renderer registration, and a decision on
  whether `gi.ball` stays an eager attribute.
- **Carrying the length scale through `Lebesgue`**: a constructor keyword
  on seven classes, new API; the key half of the finding is done.
- **Could**: the sphere's transform threshold, a components action for
  symmetric point evaluation, plotting keyword consistency (`cmap=`,
  `route=` on `plot_set`), the test-only methods, `scale` in
  `sobolev_measure`, a heat measure on the spectral-element spaces, the
  identity refusal's exception class (package A), and the CURRENT_STATE
  §7 entry (the caller's).
