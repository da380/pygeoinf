"""What is seen of a trajectory: a family of observation operators.

An observation operator takes a trajectory, a function of time for each
component of the state, to a set of readings. What is read, and when, is
fixed when the operator is built; nothing here is an unknown. Three kinds
are provided, and any of them completes the forward operator.

* **Readings at times** (:func:`point_readings`): a fixed combination of
  the state's components, one component alone being the simplest, at given
  times. The times of one record need not be those of another, nor
  regular, and a component in no record is not seen at all.
* **Averages over windows** (:func:`window_averages`): the mean of such a
  combination over intervals of time, as an instrument that integrates
  would report it.
* **A function of the state at times** (:func:`pointwise_observation`):
  readings that are not linear in the state and may depend on the
  parameters too, such as the position of a point that the state and the
  parameters together place. Its derivatives are taken by the complex
  step, as the rule's are.

The first two are linear operators on the space of trajectories. The third
acts on the state at the times and the parameters, and is composed with
the first by :func:`with_parameters`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from pygeoinf import EuclideanSpace, LinearOperator, Operator
from pygeoinf.algebra.direct_sum import (
    BlockDiagonalOperator,
    ColumnLinearOperator,
    DirectSum,
)
from pygeoinf.algebra.operators import LinearFunctional
from pygeoinf.algebra.spaces import HilbertSpace
from pygeoinf.algebra.tensor_product import TensorProduct

from .system import SECOND_STEP, STEP

# What a record reads: the index of one component, or a weight for each.
Combination = int | Sequence[float]


def _combination(trajectory: TensorProduct, what: Combination, /) -> LinearOperator:
    """A combination of a trajectory's components, as a function of time."""
    components = trajectory.factors[0]
    if isinstance(what, (int, np.integer)):
        weights = np.eye(components.dim)[int(what)]
    else:
        weights = np.asarray(what, dtype=float)
    functional = LinearFunctional.from_derivative_components(components, weights)
    return trajectory.contraction(functional, axis=0)


def point_readings(
    trajectory: TensorProduct,
    records: Sequence[tuple[Combination, Sequence[float]]],
    /,
) -> LinearOperator:
    """Chosen combinations of a trajectory's components at chosen times.

    Args:
        trajectory: the space of trajectories, the tensor product of the
            state's components with the functions of time.
        records: for each record, what it reads, the index of a component
            or a weight for each component, and the times it reads at.

    Returns:
        The operator from a trajectory to its records, a vector of the
        direct sum of one Euclidean space for each.
    """
    space = trajectory.factors[1]
    return ColumnLinearOperator(
        [
            space.point_evaluation_operator(list(times))
            @ _combination(trajectory, what)
            for what, times in records
        ]
    )


def window_averages(
    trajectory: TensorProduct,
    records: Sequence[tuple[Combination, Sequence[tuple[float, float]]]],
    /,
    *,
    points: int = 8,
) -> LinearOperator:
    """The means of chosen combinations of a trajectory's components over windows of time.

    Args:
        trajectory: the space of trajectories.
        records: for each record, what it reads and the windows it
            averages over, each a ``(start, end)`` pair.
        points: the points of the Gauss rule a window's mean is taken by.

    Returns:
        The operator from a trajectory to its records, as
        :func:`point_readings` returns it.
    """
    space = trajectory.factors[1]
    reference, weights = np.polynomial.legendre.leggauss(points)
    readers = []
    for what, windows in records:
        windows = np.asarray(windows, dtype=float)
        middle, half = 0.5 * windows.sum(axis=1), 0.5 * np.diff(windows, axis=1)[:, 0]
        times = (middle[:, None] + half[:, None] * reference).ravel()
        mean = np.kron(np.eye(len(windows)), 0.5 * weights)
        values = space.point_evaluation_operator(list(times))
        average = LinearOperator.from_matrix(
            values.codomain, EuclideanSpace(len(windows)), mean, form="components"
        )
        readers.append(average @ values @ _combination(trajectory, what))
    return ColumnLinearOperator(readers)


def states_at(trajectory: TensorProduct, times: Sequence[float], /) -> LinearOperator:
    """Every component of a trajectory at given times, a row for each component."""
    space = trajectory.factors[1]
    return trajectory.lift(space.point_evaluation_operator(list(times)), axis=1)


def pointwise_observation(
    function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    states: TensorProduct,
    parameters: EuclideanSpace,
    outputs: int,
    /,
) -> Operator:
    """A function of the state and the parameters, applied at each of several times.

    Args:
        function: ``h(x, p)``, taking the state as an array of shape
            ``(components, ...)`` and the parameters as one of shape
            ``(parameters, ...)``, real or complex, and returning ``outputs``
            values along the first axis, as a system's rule is written.
        states: the states at the times, from :func:`states_at`.
        parameters: the parameters' space.
        outputs: how many values the function returns.

    Returns:
        The operator from the pair of the states and the parameters to the
        readings, an array with a row for each output and a column for
        each time.
    """
    n, count = states.shape
    m = parameters.dim
    domain = DirectSum([states, parameters], labels=["states", "parameters"])
    codomain = TensorProduct([EuclideanSpace(outputs), states.factors[1]])

    def spread(p: np.ndarray, kind: type = float) -> np.ndarray:
        return np.repeat(np.asarray(p, dtype=kind)[:, None], count, axis=1)

    def value(model: tuple) -> np.ndarray:
        x, p = model
        return np.asarray(function(np.asarray(x, dtype=float), spread(p)), dtype=float)

    def slopes(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The derivative in each state component and each parameter, at every time."""
        in_state = np.empty((outputs, n, count))
        in_parameters = np.empty((outputs, m, count))
        for k in range(n):
            moved = np.asarray(x, dtype=complex)
            moved[k] += 1j * STEP
            in_state[:, k] = np.imag(function(moved, spread(p, complex))) / STEP
        for k in range(m):
            moved = spread(p, complex)
            moved[k] += 1j * STEP
            in_parameters[:, k] = (
                np.imag(function(np.asarray(x, dtype=complex), moved)) / STEP
            )
        return in_state, in_parameters

    def linear(in_state: np.ndarray, in_parameters: np.ndarray) -> LinearOperator:
        def apply(change: tuple) -> np.ndarray:
            dx, dp = change
            return np.einsum("qnk,nk->qk", in_state, dx) + np.einsum(
                "qmk,m->qk", in_parameters, dp
            )

        def adjoint(y: np.ndarray) -> tuple:
            return (
                np.einsum("qnk,qk->nk", in_state, y),
                np.einsum("qmk,qk->m", in_parameters, y),
            )

        return LinearOperator.from_callables(domain, codomain, apply, adjoint=adjoint)

    def derivative(model: tuple) -> LinearOperator:
        x, p = model
        return linear(*slopes(np.asarray(x, dtype=float), np.asarray(p, dtype=float)))

    def second_derivative(model: tuple, change: tuple) -> LinearOperator:
        x, p = (np.asarray(part, dtype=float) for part in model)
        dx, dp = (np.asarray(part, dtype=float) for part in change)
        size = max(np.linalg.norm(dx), np.linalg.norm(dp), 1e-300)
        step = SECOND_STEP * max(np.linalg.norm(x), np.linalg.norm(p), 1.0) / size
        ahead = slopes(x + step * dx, p + step * dp)
        behind = slopes(x - step * dx, p - step * dp)
        return linear(*((a - b) / (2.0 * step) for a, b in zip(ahead, behind)))

    return Operator.from_callables(
        domain,
        codomain,
        value,
        derivative=derivative,
        second_derivative=second_derivative,
    )


def with_parameters(
    observation: Operator,
    to_states: Operator,
    parameters: LinearOperator,
    /,
) -> Operator:
    """A pointwise observation of what a model produces, the parameters read off the same model.

    Args:
        observation: from :func:`pointwise_observation`.
        to_states: the operator from a model to the states at the times.
        parameters: the linear operator that reads the parameters off a
            model; the identity when the model is the parameters.

    Returns:
        The operator from a model to the readings.
    """
    model: HilbertSpace = to_states.domain
    twice = ColumnLinearOperator([LinearOperator.identity(model), parameters])
    both = BlockDiagonalOperator(
        [to_states, LinearOperator.identity(parameters.codomain)],
        domain=twice.codomain,
        codomain=observation.domain,
    )
    return observation @ both @ twice


def first_of(model: DirectSum, /) -> LinearOperator:
    """The first summand of a direct sum's vectors, which is where a model keeps its parameters."""
    spaces = model.subspaces

    def adjoint(y: Any) -> tuple:
        return (y, *(space.zero() for space in spaces[1:]))

    return LinearOperator.from_callables(
        model, spaces[0], lambda x: x[0], adjoint=adjoint
    )
