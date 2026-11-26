"""The map from parameters and forcing to the solution of a dynamical system.

``S(p, w)`` is the solution ``x`` of ``dx/dt = f(x, p) + G w(t)`` from a
known initial state, read at the nodes of a fixed grid of times: an
operator from the parameters, and the forcing if there is one, to a table
of the state's values. The forcing is given the same way, by its values at
the nodes, and is the grid's interpolant between them. Nothing here knows
what space of functions those tables stand for; they are arrays under the
plain dot product, and another module gives them their meaning.

The integration is left to an adaptive integrator used as it comes, and
the derivatives are those of the differential equation, not of the
integrator's arithmetic:

* the **tangent** of a perturbation ``(dp, dw)`` solves the linearized
  equation ``dx' = f_x dx + f_p dp + G dw`` forwards from zero;
* the **adjoint** takes a table ``y`` of weights, one for each component
  at each node, and returns what pairs with a perturbation as the tangent
  pairs with ``y``. It solves ``lam' = -f_x^T lam`` backwards from zero at
  the final time, ``lam`` stepping up by ``y`` at each node it passes: a
  value read at a time is a point source there. The parameters' part is
  the integral of ``f_p^T lam``; the forcing's part, for each node, is
  the integral of ``G^T lam`` against that node's interpolating function,
  by a Gauss rule between the nodes;
* the **sensitivities** solve the linearized equation for every parameter
  at once, which gives the whole Jacobian in the parameters in one sweep
  and is the cheaper route while the parameters are few.

The tangent and the adjoint are adjoint to each other to the integrator's
tolerance. The sensitivities and their transpose are adjoint to rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from pygeoinf import EuclideanSpace, LinearOperator, Operator
from pygeoinf.algebra.direct_sum import DirectSum
from pygeoinf.algebra.linearization import Linearization
from pygeoinf.algebra.spaces import HilbertSpace
from pygeoinf.algebra.tensor_product import TensorProduct

from .system import DynamicalSystem


# A weight below this fraction of the largest is not stopped at by the adjoint.
NEGLIGIBLE = 1e-13

Interpolant = Callable[[np.ndarray], Callable[[Any], np.ndarray]]


@dataclass(frozen=True)
class TimeGrid:
    """The times a trajectory is read at and a forcing is given at.

    Attributes:
        nodes: the times, increasing, the first the start of the interval
            and the last its end.
        weights: the quadrature weight of each node, positive.
        interpolant: what turns values at the nodes, one row per
            component, into a function of time; a cubic spline through the
            nodes if omitted.
    """

    nodes: np.ndarray
    weights: np.ndarray
    interpolant: Interpolant | None = None

    @property
    def start(self) -> float:
        """The first time."""
        return float(self.nodes[0])

    @property
    def end(self) -> float:
        """The last time."""
        return float(self.nodes[-1])

    def function(self, values: np.ndarray, /) -> Callable[[Any], np.ndarray]:
        """The function of time with given values at the nodes, shape ``(components, nodes)``."""
        values = np.asarray(values, dtype=float)
        if self.interpolant is not None:
            return self.interpolant(values)
        return CubicSpline(self.nodes, values, axis=1)

    @cached_property
    def quadrature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A Gauss rule between the nodes, and each node's interpolating function on it.

        Returns:
            The rule's times, three between each pair of nodes, its
            weights, and the matrix whose entry ``[q, j]`` is the value at
            time ``q`` of the function that is one at node ``j`` and zero
            at the others: what an integral against a function given at
            the nodes is taken with.
        """
        points, weights = np.polynomial.legendre.leggauss(3)
        left, right = self.nodes[:-1, None], self.nodes[1:, None]
        times = (0.5 * (left + right) + 0.5 * (right - left) * points).ravel()
        scale = (0.5 * (right - left) * weights).ravel()
        cardinal = np.asarray(self.function(np.eye(self.nodes.size))(times)).T
        return times, scale, cardinal


class Solver:
    """The solutions of a dynamical system from a fixed initial state, and their derivatives."""

    def __init__(
        self,
        system: DynamicalSystem,
        initial_state: np.ndarray,
        grid: TimeGrid,
        /,
        *,
        method: str = "DOP853",
        rtol: float = 1e-9,
        atol: float = 1e-11,
        max_step: float | None = None,
    ) -> None:
        """
        Args:
            system: the system.
            initial_state: the state at the grid's first time.
            grid: where the trajectory is sampled and, for a forced
                system, where the forcing is given.
            method: the integrator, one of :func:`scipy.integrate.solve_ivp`'s.
            rtol: its relative tolerance.
            atol: its absolute tolerance.
            max_step: the longest step it may take. A forcing given at the
                grid's nodes is only as smooth as its interpolant, and an
                integrator of high order that strides over several nodes
                at once meets its tolerance erratically; a cap near the
                nodes' spacing keeps the solution a smooth function of
                the forcing. None for no cap.
        """
        self.system = system
        self.initial_state = np.asarray(initial_state, dtype=float)
        self.grid = grid
        self._options = {"method": method, "rtol": rtol, "atol": atol}
        if max_step is not None:
            self._options["max_step"] = float(max_step)
        # How many integrations of each kind were run.
        self.solves = {"state": 0, "tangent": 0, "adjoint": 0, "sensitivity": 0}

    def _integrate(self, kind: str, rate: Callable, start: np.ndarray, /) -> Any:
        self.solves[kind] += 1
        result = solve_ivp(
            rate,
            (self.grid.start, self.grid.end),
            start,
            dense_output=True,
            **self._options,
        )
        if not result.success:
            raise RuntimeError(f"The {kind} integration failed: {result.message}")
        return result.sol

    def _forcing(self, w: np.ndarray | None, /) -> Callable[[float], np.ndarray]:
        """``G w(t)`` as a function of time; zero for no forcing."""
        if w is None or self.system.forcing_dim == 0:
            zero = np.zeros(self.system.state_dim)
            return lambda t: zero
        through = self.grid.function(w)
        matrix = self.system.forcing_matrix
        return lambda t: matrix @ through(t)

    def solve(
        self, p: np.ndarray, w: np.ndarray | None = None, /
    ) -> Callable[[Any], np.ndarray]:
        """The state as a function of time.

        Args:
            p: the parameters.
            w: the forcing at the grid's nodes, shape ``(forcing_dim,
                nodes)``, or None for none.

        Returns:
            A function of a time or of an array of times, the state's
            components along the first axis.
        """
        system, forcing = self.system, self._forcing(w)
        p = np.asarray(p, dtype=float)
        return self._integrate(
            "state", lambda t, x: system.rate(x, p) + forcing(t), self.initial_state
        )

    def sample(self, function: Callable[[Any], np.ndarray], /) -> np.ndarray:
        """A function of time at the grid's nodes, shape ``(components, nodes)``."""
        return np.asarray(function(self.grid.nodes))

    def tangent(
        self,
        state: Callable[[Any], np.ndarray],
        p: np.ndarray,
        dp: np.ndarray,
        dw: np.ndarray | None = None,
        /,
    ) -> Callable[[Any], np.ndarray]:
        """The first-order change of the state under a change of parameters and forcing."""
        system, forcing = self.system, self._forcing(dw)
        p, dp = np.asarray(p, dtype=float), np.asarray(dp, dtype=float)

        def rate(t: float, dx: np.ndarray) -> np.ndarray:
            in_state, in_parameters = system.jacobians(state(t), p)
            return in_state @ dx + in_parameters @ dp + forcing(t)

        return self._integrate("tangent", rate, np.zeros(system.state_dim))

    def sensitivities(
        self, state: Callable[[Any], np.ndarray], p: np.ndarray, /
    ) -> np.ndarray:
        """The derivative of the sampled state in every parameter.

        Returns:
            Shape ``(state_dim, nodes, parameter_dim)``.
        """
        system = self.system
        n, m = system.state_dim, system.parameter_dim
        p = np.asarray(p, dtype=float)

        def rate(t: float, flat: np.ndarray) -> np.ndarray:
            in_state, in_parameters = system.jacobians(state(t), p)
            return (in_state @ flat.reshape(n, m) + in_parameters).ravel()

        solution = self._integrate("sensitivity", rate, np.zeros(n * m))
        return solution(self.grid.nodes).reshape(n, m, -1).transpose(0, 2, 1)

    def adjoint(
        self,
        state: Callable[[Any], np.ndarray],
        p: np.ndarray,
        y: np.ndarray,
        /,
        *,
        forcing: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """What pairs with a perturbation as its tangent pairs with a table of weights.

        ``sum_ij y_ij dx_i(t_j) == dp . g_p + sum_kj dw_k(t_j) g_w[k, j]``
        for the tangent ``dx`` of any ``(dp, dw)``.

        Args:
            state: the state the tangent was taken about.
            p: the parameters.
            y: the weights, shape ``(state_dim, nodes)``.
            forcing: whether the forcing's part is wanted. Without it only
                the nodes with a weight are stopped at.

        Returns:
            ``g_p``, the integral of ``f_p^T lam``, and ``g_w``, the
            integral of ``G^T lam`` against each node's interpolating
            function, or None.
        """
        system, grid = self.system, self.grid
        n, m = system.state_dim, system.parameter_dim
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)

        def rate(t: float, both: np.ndarray) -> np.ndarray:
            in_state, in_parameters = system.jacobians(state(t), p)
            multiplier = both[:n]
            return np.concatenate(
                [-in_state.T @ multiplier, -in_parameters.T @ multiplier]
            )

        # Backwards from the last node, stopping where there is a weight to
        # take up, where the multiplier steps. A weight that is rounding
        # beside the largest is no weight.
        floor = NEGLIGIBLE * np.abs(y).max(initial=0.0)
        stops = [j for j in range(grid.nodes.size) if np.any(np.abs(y[:, j]) > floor)]
        stops = sorted(set(stops) | {0, grid.nodes.size - 1}, reverse=True)
        both = np.zeros(n + m)
        if forcing:
            times, scale, cardinal = grid.quadrature
            between = np.zeros((n, times.size))
        self.solves["adjoint"] += 1
        step = None
        for upper, lower in zip(stops, stops[1:] + [None]):
            both[:n] += y[:, upper]
            if lower is None:
                break
            span = (grid.nodes[upper], grid.nodes[lower])
            options = dict(self._options)
            if step is not None:
                options["first_step"] = min(step, abs(span[1] - span[0]))
            result = solve_ivp(rate, span, both, dense_output=forcing, **options)
            if not result.success:
                raise RuntimeError(f"The adjoint integration failed: {result.message}")
            if result.t.size > 1:
                step = abs(result.t[-1] - result.t[-2])
            if forcing:
                # Three of the rule's times lie between each pair of nodes.
                inside = slice(3 * lower, 3 * upper)
                between[:, inside] = result.sol(times[inside])[:n]
            both = result.y[:, -1].copy()
        if not forcing:
            return both[n:], None
        return both[n:], (scale * (system.forcing_matrix.T @ between)) @ cardinal


def solution_operator(
    solver: Solver,
    /,
    *,
    forced: bool = False,
    route: Literal["adjoint", "sensitivity"] = "adjoint",
) -> Operator:
    """The solution as an operator of the library, with its derivative.

    The domain is the parameters, a Euclidean space, or with ``forced`` the
    direct sum of that and the forcing's table. The codomain is the state's
    table. A table is an array with a row for each component and a column
    for each node, a vector of the tensor product of two Euclidean spaces,
    so under the plain dot product.

    Args:
        solver: the system, its initial state, grid and integrator.
        forced: whether the forcing is an argument.
        route: how the derivative acts. ``"adjoint"`` integrates the
            linearized equation for each application and the adjoint
            equation for each application of the adjoint. ``"sensitivity"``
            integrates for the whole Jacobian in the parameters once at
            each point, after which both are products with it and adjoint
            to rounding; it is for a problem without forcing.

    Returns:
        The operator.

    Raises:
        ValueError: if the sensitivity route is asked for with forcing, or
            forcing of a system that takes none.
    """
    system, grid = solver.system, solver.grid
    if forced and system.forcing_dim == 0:
        raise ValueError("The system takes no forcing.")
    if forced and route == "sensitivity":
        raise ValueError("The sensitivity route is for a problem without forcing.")
    nodes = EuclideanSpace(grid.nodes.size)
    parameters = EuclideanSpace(system.parameter_dim)
    codomain = TensorProduct([EuclideanSpace(system.state_dim), nodes])
    domain: HilbertSpace = parameters
    if forced:
        forcing = TensorProduct([EuclideanSpace(system.forcing_dim), nodes])
        domain = DirectSum([parameters, forcing], labels=["parameters", "forcing"])

    def split(model: Any) -> tuple[np.ndarray, np.ndarray | None]:
        if not forced:
            return np.asarray(model, dtype=float), None
        return np.asarray(model[0], dtype=float), np.asarray(model[1], dtype=float)

    def linearize(model: Any) -> Linearization:
        p, w = split(model)
        state = solver.solve(p, w)
        value = solver.sample(state)
        if route == "sensitivity":
            jacobian = solver.sensitivities(state, p)

            def apply(dp: np.ndarray) -> np.ndarray:
                return jacobian @ np.asarray(dp, dtype=float)

            def apply_adjoint(y: np.ndarray) -> np.ndarray:
                return np.einsum("ijk,ij->k", jacobian, np.asarray(y))

        else:

            def apply(change: Any) -> np.ndarray:
                dp, dw = split(change)
                return solver.sample(solver.tangent(state, p, dp, dw))

            def apply_adjoint(y: np.ndarray) -> Any:
                in_parameters, in_forcing = solver.adjoint(
                    state, p, np.asarray(y), forcing=forced
                )
                return (in_parameters, in_forcing) if forced else in_parameters

        derivative = LinearOperator.from_callables(
            domain, codomain, apply, adjoint=apply_adjoint
        )
        return Linearization(model, value, derivative)

    def value(model: Any) -> np.ndarray:
        p, w = split(model)
        return solver.sample(solver.solve(p, w))

    return Operator.from_callables(domain, codomain, value, linearize=linearize)
