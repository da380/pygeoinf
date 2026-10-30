"""Part G: bent rays.

The forward problem as it is rather than as linearized: the travel time
through the slowness the model describes, along the ray that slowness bends.
Each two-point ray is re-traced in the perturbed disk by shooting, the ray
equations integrated from the source and the take-off angle found by a root
find on where the ray emerges (:func:`rays.bend`). By Fermat's principle a
travel time is stationary under a change of path, so its derivative with
respect to the slowness is the path integral along the current bent ray: the
linearized operator at a model is the path-integral operator of parts A to F
built on that model's rays, with the library's adjoint.

The prior is five times wider than in the other parts, a pointwise standard
deviation of 5% of the surface slowness, so that the bending of its smooth
draws matters while the problem stays well posed; the rays are kept within
70 degrees, well above the core. The derivative is checked by finite differences, the MAP model is
found by Newton-CG on the Gauss-Newton Hessian, and the answer is set
against the linear posterior mean computed on the reference rays from the
same data.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from pygeoinf import (
    CholeskySolver,
    EuclideanSpace,
    ForwardProblem,
    GaussianMeasure,
    LinearForwardProblem,
    LinearGaussianInversion,
    Linearization,
    MaximumAPosteriori,
    Operator,
)
from pygeoinf.algebra.operators import EvaluationError
from pygeoinf.numerics.optimization import NewtonCG
from pygeoinf.testing import check_derivative

from src.common import draw, draw_rays, panels, parse, part_rng, save
from src.problem import (
    CORE_RADIUS,
    GRADIENT,
    MANTLE,
    PROPERTIES,
    QUICK,
    RADIUS,
    SURFACE_VELOCITY,
    Scale,
    sobolev_prior,
    model_space,
    network,
    path_integral_operator,
    property_operator,
)
from src.rays import Ray, bend

#: The mantle under the stations, ``(r1, r2, angle1, angle2)``, where the
#: rays cross and an error is worth reading.
UNDER_THE_STATIONS = (0.7, 1.0, 0.0, 150.0)
#: No re-traced ray may go deeper than this, two hundredths above the core.
#: The limit is the core: the medium the rays are shot through has none,
#: being the mantle continued below its base, and the derivative would read
#: the core's layer there. The two hundredths are a margin. The deepest
#: reference ray turns at 0.64; in a draw from the wide prior moved by the
#: derivative check's largest step one turns at 0.59.
FLOOR = CORE_RADIUS + 0.02
#: The integrator's step of arc length.
STEP = 5e-3
#: The finite-difference steps of the derivative check, in the space's norm,
#: in which a draw from the prior measures some tens.
STEPS = (20.0, 10.0, 5.0, 1.0, 0.1)

#: The two sizes of this part: the quick scale's network, or a denser one, a
#: prior five times wider than the other parts', and the rays kept within
#: 70 degrees.
BENT_QUICK = dataclasses.replace(
    QUICK,
    name="bent, quick",
    mmax=16,
    length_scale=0.2,
    prior_std=5.0,
    max_distance=70.0,
    nodes=401,
    std_samples=0,
    n_jobs=None,
    bump_width=0.0,
)
BENT = dataclasses.replace(
    BENT_QUICK,
    name="bent",
    mmax=32,
    length_scale=0.15,
    receivers=(*(float(a) for a in np.linspace(0.0, 150.0, 16)), 270.0),
)


class RadialFunctions:
    """A layer's field as a Fourier series in angle whose coefficients are
    functions of radius: ``f(r, t) = sum_j f_j(r) w_j(t)``, with ``w_j`` the
    modes ``1 / sqrt(2 pi)``, ``cos(m t) / sqrt(pi)`` and ``sin(m t) /
    sqrt(pi)`` the layer's components are coefficients of.

    The radial functions are the layer's radial modes summed with the
    field's components, sampled on a fine grid of radii; the angular part is
    exact.
    """

    def __init__(self, layer: Any, radii: np.ndarray) -> None:
        self.layer, self.radii = layer, radii
        orders, indices = layer.orders, layer.radial_indices
        self.orders = np.arange(-layer.mmax, layer.mmax + 1)
        self._modes = np.empty((radii.size, layer.dim))
        self._term = np.searchsorted(self.orders, orders)
        for m, radial in enumerate(layer.basis):
            modes = radial.evaluate(radii)
            for sign in (1,) if m == 0 else (1, -1):
                columns = np.flatnonzero(orders == sign * m)
                self._modes[:, columns] = modes[:, indices[columns]]
        self._norm = np.where(self.orders == 0, 1.0 / np.sqrt(2.0), 1.0) / np.sqrt(
            np.pi
        )

    def coefficients(self, field: np.ndarray, /) -> np.ndarray:
        """The radial functions of a field on the grid of radii, one column
        per signed order, the Fourier normalization included."""
        weighted = self._modes * self.layer.to_components(field)
        out = np.zeros((self.radii.size, self.orders.size))
        np.add.at(out.T, self._term, weighted.T)
        return out * self._norm

    def waves(self, angles: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        """The angular modes at angles in radians, without their
        normalization, and their derivatives: ``cos(m t)`` for ``m >= 0`` and
        ``sin(|m| t)`` for ``m < 0``."""
        m = np.abs(self.orders)
        phase = np.outer(angles, m)
        cosine, sine = np.cos(phase), np.sin(phase)
        sine_mode = self.orders < 0
        value = np.where(sine_mode, sine, cosine)
        slope = np.where(sine_mode, m * cosine, -m * sine)
        return value, slope


class Slowness:
    """The slowness of the perturbed disk: the reference ``1 / v(r)`` and the
    model's perturbation, in percent of the surface slowness.

    The perturbation's radial functions are interpolated by cubic splines on
    a fine grid of the mantle's radii, its angular modes are exact, and it is
    held constant along a radius past the surface; its gradient comes with
    it.
    """

    def __init__(self, expansion: RadialFunctions, field: np.ndarray) -> None:
        self._expansion = expansion
        self._spline = CubicSpline(expansion.radii, expansion.coefficients(field))
        self._slope = self._spline.derivative()
        self._range = (expansion.radii[0], expansion.radii[-1])

    def __call__(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.hypot(x, y)
        clamped = np.clip(r, *self._range)
        waves, turning = self._expansion.waves(np.arctan2(y, x))
        functions = self._spline(clamped)
        perturbation = np.sum(functions * waves, axis=1)
        along_r = np.sum(self._slope(clamped) * waves, axis=1) * (clamped == r)
        along_theta = np.sum(functions * turning, axis=1)
        velocity = SURFACE_VELOCITY + GRADIENT * (RADIUS - r)
        slowness = 1.0 / velocity + perturbation / 100.0
        d_r = GRADIENT / velocity**2 + along_r / 100.0
        d_theta = along_theta / (100.0 * r)
        cosine, sine = x / r, y / r
        return slowness, cosine * d_r - sine * d_theta, sine * d_r + cosine * d_theta


class BentForward:
    """The travel-time perturbations through a model, along its own rays.

    A datum is ``100 (T(m) - T_0)``, in hundredths of a time unit, ``T(m)``
    the time along the two-point ray re-traced in the slowness the model
    describes and ``T_0`` the reference ray's. At the reference model it is
    zero, and its derivative is the path integral along the reference rays,
    the operator of the linear parts. The last model asked about is
    remembered with its rays, since an optimizer asks for the value, the
    derivative and the Gauss-Newton Hessian at one point in turn.
    """

    def __init__(self, scale: Scale, space: Any, pairs: list) -> None:
        self.scale, self.space, self.pairs = scale, space, pairs
        self.reference = [
            MANTLE.trace(source, receiver, nodes=scale.nodes)
            for source, receiver in pairs
        ]
        self.reference_times = np.array([ray.travel_time for ray in self.reference])
        distances = [abs((r - s + 180.0) % 360.0 - 180.0) for s, r in pairs]
        self.take_off = np.arcsin(
            [
                MANTLE.ray_parameter(d) * MANTLE.velocity(RADIUS) / RADIUS
                for d in distances
            ]
        )
        mantle = space.layers[space.index("mantle")]
        self.expansion = RadialFunctions(mantle, np.linspace(CORE_RADIUS, RADIUS, 361))
        self.traces = 0
        # Each search starts from the last model's take-off angles, which an
        # optimizer's next model is near.
        self._start, self._slope = self.take_off, None
        self._memo: tuple[bytes, np.ndarray, list[Ray], Any] | None = None
        self.operator = Operator.from_callables(
            space,
            EuclideanSpace(len(pairs)),
            lambda model: self.trace(model)[0],
            linearize=self._linearize,
        )

    def trace(self, model: Any, /) -> tuple[np.ndarray, list[Ray]]:
        """The data and the rays of a model.

        Raises:
            EvaluationError: if a ray cannot be traced in it.
        """
        key = b"".join(np.ascontiguousarray(part).tobytes() for part in model)
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1], self._memo[2]
        mantle = self.space.index("mantle")
        try:
            rays, angles, slope = bend(
                Slowness(self.expansion, model[mantle]),
                self.pairs,
                self._start,
                slope=self._slope,
                radius=RADIUS,
                bottom=FLOOR,
                step=STEP,
                nodes=self.scale.nodes,
            )
        except ValueError as error:
            raise EvaluationError(f"The rays could not be traced: {error}") from error
        self.traces += 1
        self._start, self._slope = angles, slope
        times = np.array([ray.travel_time for ray in rays])
        values = 100.0 * (times - self.reference_times)
        self._memo = (key, values, rays, None)
        return values, rays

    def _linearize(self, model: Any) -> Linearization:
        values, rays = self.trace(model)
        key, _, _, derivative = self._memo
        if derivative is None:
            derivative = path_integral_operator(self.scale, self.space, rays)
            self._memo = (key, values, rays, derivative)
        return Linearization(model, values, derivative)


@dataclass
class BentProblem:
    """Everything the part builds."""

    scale: Scale
    space: Any
    prior: GaussianMeasure
    pairs: list
    forward: BentForward
    problem: ForwardProblem
    linear: LinearForwardProblem
    truth: Any
    data: np.ndarray
    properties: Any

    def mantle_error(self, model: Any, /) -> float:
        """``|m - truth| / |truth|`` over the mantle, by its grid's quadrature."""
        return _relative(self, model, self.truth)

    def local_error(self, model: Any, /) -> float:
        """``|m - truth| / |truth|`` over :data:`UNDER_THE_STATIONS`, in the
        ``L2`` norm on a fine polar grid."""
        expansion = self.forward.expansion
        r1, r2, a1, a2 = UNDER_THE_STATIONS
        inside = (expansion.radii >= r1) & (expansion.radii <= r2)
        waves, _ = expansion.waves(np.radians(np.linspace(a1, a2, 301)))
        index = self.space.index("mantle")

        def values(field: Any) -> np.ndarray:
            return expansion.coefficients(field[index])[inside] @ waves.T

        weights = expansion.radii[inside][:, None]
        truth = values(self.truth)
        difference = values(model) - truth
        return float(
            np.sqrt(np.sum(weights * difference**2) / np.sum(weights * truth**2))
        )


def build(scale: Scale, /) -> BentProblem:
    """The space, the wider prior, the rays, the bent forward and the data."""
    rng = np.random.default_rng([scale.seed, 0])
    pairs = network(
        scale.sources,
        scale.receivers,
        min_distance=scale.min_distance,
        max_distance=scale.max_distance,
    )
    space = model_space(scale)
    prior = sobolev_prior(space, order=scale.prior_order, std=scale.prior_std)
    forward = BentForward(scale, space, pairs)
    noise = GaussianMeasure.from_standard_deviation(
        forward.operator.codomain, scale.noise_std
    )
    problem = ForwardProblem(forward.operator, error=noise)
    linear = LinearForwardProblem(
        path_integral_operator(scale, space, forward.reference), error=noise
    )
    truth = prior.sample(rng=rng)
    data = forward.operator(truth) + noise.sample(rng=rng)
    return BentProblem(
        scale,
        space,
        prior,
        pairs,
        forward,
        problem,
        linear,
        truth,
        data,
        property_operator(scale, space),
    )


def _plane(ray: Ray, /) -> np.ndarray:
    """A ray's nodes as positions in the plane."""
    t = np.radians(ray.nodes[:, 1])
    return np.column_stack([ray.nodes[:, 0] * np.cos(t), ray.nodes[:, 0] * np.sin(t)])


def node_shift(bent: Ray, reference: Ray, /) -> float:
    """The largest distance between the nodes of two rays at the same
    fraction of their length, both having the same number of nodes."""
    return float(np.hypot(*(_plane(bent) - _plane(reference)).T).max())


def derivative_check(problem: BentProblem, /, *, rng: np.random.Generator) -> dict:
    """The derivative against central differences, at the reference model
    and at two draws from the prior, for each step in :data:`STEPS`.

    One direction at each point, a draw from the prior scaled to unit norm,
    the same at every step. The library's defect is the residual over the
    larger of one and the norms of the two directional derivatives; the
    larger of those norms is returned beside it.
    """
    space, prior = problem.space, problem.prior
    points = {"reference": space.zero()}
    for index in range(2):
        points[f"draw {index + 1}"] = prior.sample(rng=rng)
    seed = int(rng.integers(2**31))
    direction = prior.sample(rng=np.random.default_rng(seed))
    direction = space.scale(1.0 / space.norm(direction), direction)
    table, sizes = {}, {}
    for name, point in points.items():
        derivative = problem.forward.operator.at(point).derivative
        sizes[name] = float(np.linalg.norm(derivative(direction)))
        table[name] = []
        for step in STEPS:
            report = check_derivative(
                problem.forward.operator,
                point,
                rng=np.random.default_rng(seed),
                measure=prior,
                step=step,
                trials=1,
                rtol=1.0,
            )
            table[name].append(report.worst)
    return {"defects": table, "sizes": sizes}


def bent_answer(
    problem: BentProblem, /, *, rng: np.random.Generator, outdir: pathlib.Path
) -> dict:
    """The derivative check, the bending, the MAP model against the linear
    mean."""
    started = time.perf_counter()
    scale, space, forward = problem.scale, problem.space, problem.forward
    print(f"\n=== G. Bent rays ({scale.name}) ===")
    core, mantle = space.layers
    print(
        f"{len(problem.pairs)} rays between {scale.min_distance:g} and "
        f"{scale.max_distance:g} degrees on the "
        f"layered disk of {space.dim} coefficients (mmax {core.mmax} and "
        f"{mantle.mmax}, length scale {scale.length_scale}); prior pointwise std "
        f"{scale.prior_std:g}%, noise {scale.noise_std}; shooting at a step of "
        f"{STEP:g}, "
        f"{scale.nodes} nodes a ray"
    )

    t0 = time.perf_counter()
    zero, rays = forward.trace(space.zero())
    shift = max(node_shift(a, b) for a, b in zip(rays, forward.reference))
    print(
        f"in the reference disk the shooting gives the radial tracer's times to "
        f"{np.abs(zero).max() / 100.0:.1e} and its nodes to {shift:.1e} "
        f"({time.perf_counter() - t0:.2f} s)"
    )

    t0 = time.perf_counter()
    check = derivative_check(problem, rng=rng)
    table = check["defects"]
    print(
        f"the derivative against central differences in one direction from the "
        f"prior at each point, of unit norm in H^2, where the truth measures "
        f"{space.norm(problem.truth):.1f}; the defect is the residual over the "
        f"larger of one and the derivative's size, which is "
        + ", ".join(f"{size:.3f}" for size in check["sizes"].values())
        + f" ({time.perf_counter() - t0:.1f} s):"
    )
    print("  step       " + " ".join(f"{name:>11s}" for name in table))
    for row, step in enumerate(STEPS):
        print(
            f"  {step:<9.0e}  "
            + " ".join(f"{values[row]:11.1e}" for values in table.values())
        )

    bent_values, bent_rays = forward.trace(problem.truth)
    linear_values = problem.linear.forward_operator(problem.truth)
    difference = bent_values - linear_values
    shifts = np.array([node_shift(a, b) for a, b in zip(bent_rays, forward.reference)])
    deepest = [
        (float(b.nodes[:, 0].min()), float(a.nodes[:, 0].min()))
        for a, b in zip(bent_rays, forward.reference)
    ]
    turning = max(abs(new - old) for old, new in deepest)
    print(
        f"the bending in the truth: travel-time perturbations of std "
        f"{np.std(bent_values):.2f}; bent less linear {np.sqrt(np.mean(difference**2)):.3f} "
        f"rms and {np.abs(difference).max():.3f} at most, against noise of "
        f"{scale.noise_std}, every one of them negative or nil, as Fermat says "
        f"({(difference <= 1e-9).all()}); the rays move by {np.median(shifts):.4f} "
        f"(median) to {shifts.max():.4f}, and their deepest points by up to "
        f"{turning:.4f}"
    )

    t0 = time.perf_counter()
    linear_mean = LinearGaussianInversion(
        problem.linear, problem.prior, solver=CholeskySolver()
    )(problem.data).expectation
    print(
        f"the linear posterior mean, on the reference rays "
        f"({time.perf_counter() - t0:.2f} s): chi-squared "
        f"{problem.linear.chi_squared(linear_mean, problem.data):.1f} as it "
        f"believes, {problem.problem.chi_squared(linear_mean, problem.data):.1f} "
        f"along its own bent rays, on {len(problem.pairs)} data; error over the "
        f"mantle {problem.mantle_error(linear_mean):.3f}, under the stations "
        f"{problem.local_error(linear_mean):.3f}"
    )

    t0 = time.perf_counter()
    traces = forward.traces
    optimizer = NewtonCG(
        preconditioner=problem.prior.covariance,
        rtol=1e-6,
        ftol=1e-10,
        max_iterations=30,
    )
    estimator = MaximumAPosteriori(
        problem.problem, problem.prior, optimizer=optimizer, gauss_newton=True
    )
    result = estimator.solve(problem.data)
    mode = result.minimizer
    _, mode_rays = forward.trace(mode)
    print(
        f"the MAP model by Newton-CG on the Gauss-Newton Hessian, from the prior's "
        f"mean: {result.iterations} iterations, {forward.traces - traces} ray tracings, "
        f"{time.perf_counter() - t0:.1f} s, {result.message}; chi-squared "
        f"{problem.problem.chi_squared(mode, problem.data):.1f}, objective "
        f"{result.value:.2f} (history "
        + ", ".join(f"{value:.2f}" for value in result.history)
        + f"); error over the mantle {problem.mantle_error(mode):.3f}, under the "
        f"stations {problem.local_error(mode):.3f}"
    )
    change = _relative(problem, mode, linear_mean)
    print(
        f"the MAP model against the linear mean: they differ by {change:.3f} of the "
        "latter"
    )
    truth_properties = problem.properties(problem.truth)
    linear_properties = problem.properties(linear_mean)
    mode_properties = problem.properties(mode)
    print(f"  {'mean over':<16s} {'linear':>8s} {'MAP':>8s} {'truth':>8s}")
    for name, a, b, c in zip(
        PROPERTIES, linear_properties, mode_properties, truth_properties
    ):
        print(f"  {name:<16s} {a:+8.3f} {b:+8.3f} {c:+8.3f}")

    fig, axes = panels(space, 2, 3)
    limit = float(np.abs(mantle.interior_values(problem.truth[1])).max())
    worst = int(np.argmax(shifts))
    moved = np.hypot(*(_plane(bent_rays[worst]) - _plane(forward.reference[worst])).T)
    center = _plane(bent_rays[worst])[int(np.argmax(moved))]
    for ax, title in zip(
        axes[0, :2],
        ("the truth: bent rays (black), reference rays (white)", "closer in"),
    ):
        draw(space, problem.truth, ax, title=title, limit=limit)
        draw_rays(ax, forward.reference, colors="white", linewidths=1.6)
        draw_rays(ax, bent_rays, colors="black", linewidths=0.6)
    axes[0, 1].set_xlim(center[0] - 0.12, center[0] + 0.12)
    axes[0, 1].set_ylim(center[1] - 0.12, center[1] + 0.12)

    ax = axes[0, 2]
    ax.set_aspect("auto")
    ax.set_axis_on()
    ax.bar(np.arange(len(difference)), difference, color="C0")
    for level in (-2.0 * scale.noise_std, 2.0 * scale.noise_std):
        ax.axhline(level, color="black", lw=0.8, ls=":")
    ax.set_xlabel("ray")
    ax.set_ylabel("bent less linear (hundredths)")
    ax.set_title(
        "the truth's travel times, bent less linear,\nagainst twice the noise",
        fontsize=10,
    )

    gap = space.subtract(mode, linear_mean)
    draw(space, linear_mean, axes[1, 0], title="linear posterior mean", limit=limit)
    draw(space, mode, axes[1, 1], title="the MAP model, on its bent rays", limit=limit)
    draw_rays(axes[1, 1], mode_rays, colors="black", linewidths=0.4, alpha=0.6)
    reach = float(np.abs(mantle.interior_values(gap[1])).max())
    draw(
        space, gap, axes[1, 2], title="the MAP model less the linear mean", limit=reach
    )
    save(
        fig,
        outdir / "bent.png",
        title="Tomography, part G: bent rays. What bending changes: the rays moved by "
        f"up to {shifts.max():.3f}, the answer by {change:.0%}",
    )
    print(f"G took {time.perf_counter() - started:.1f} s")
    return {
        "zero": zero,
        "shift_at_zero": shift,
        "defects": table,
        "derivative_sizes": check["sizes"],
        "difference": difference,
        "shifts": shifts,
        "linear_mean": linear_mean,
        "mode": mode,
        "result": result,
        "linear_chi_squared": problem.problem.chi_squared(linear_mean, problem.data),
        "mode_chi_squared": problem.problem.chi_squared(mode, problem.data),
        "change": change,
    }


def _relative(problem: BentProblem, model: Any, reference: Any, /) -> float:
    """``|model - reference| / |reference|`` over the mantle, by its grid's
    quadrature."""
    index = problem.space.index("mantle")
    mantle = problem.space.layers[index]
    weights = mantle.basis[0].weights()[:, None] * mantle.grid.weight
    base = mantle.interior_values(reference[index])
    difference = mantle.interior_values(model[index]) - base
    return float(np.sqrt(np.sum(weights * difference**2) / np.sum(weights * base**2)))


def run(scale: Scale = BENT, /, *, outdir: pathlib.Path) -> dict:
    problem = build(scale)
    return {"problem": problem} | bent_answer(
        problem, rng=part_rng(scale.seed, "g"), outdir=outdir
    )


def main() -> None:
    args = parse(__doc__, quick=BENT_QUICK, full=BENT)
    run(args.scale, outdir=args.outdir)


if __name__ == "__main__":
    main()
