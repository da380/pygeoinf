"""Parameters that are confined to part of the line, through coordinates that are not.

A mass or a length is positive; a ratio may be confined to an interval. An
optimizer, a Gaussian approximation and a sampler all work most simply on
the whole of a Euclidean space, so the unknown is taken to be a vector of
free coordinates ``u`` and the parameters are a function of it, one
coordinate each: ``p = exp(u)`` for a positive parameter, ``p = lower +
(upper - lower) / (1 + exp(-u))`` for one in an interval, ``p = u`` for one
that is free. A Gaussian prior on ``u`` is then a log-normal or a
logit-normal prior on ``p``, with no mass outside the allowed set.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pygeoinf import EuclideanSpace, LinearOperator, Operator

# One entry per parameter: "positive", "free", or a (lower, upper) pair.
Constraint = str | tuple[float, float]


class ParameterMap:
    """The parameters as a function of free coordinates, one coordinate each."""

    def __init__(self, constraints: Sequence[Constraint], /) -> None:
        """
        Args:
            constraints: for each parameter ``"positive"``, ``"free"`` or a
                ``(lower, upper)`` pair.

        Raises:
            ValueError: for a constraint that is none of these.
        """
        self.constraints = tuple(constraints)
        for constraint in self.constraints:
            if constraint not in ("positive", "free") and not (
                isinstance(constraint, tuple) and len(constraint) == 2
            ):
                raise ValueError(f"Not a constraint: {constraint!r}.")
        self.space = EuclideanSpace(len(self.constraints))

    def _each(self, u: np.ndarray, order: int) -> np.ndarray:
        """The map, or its first or second derivative, coordinate by coordinate."""
        out = np.empty(len(self.constraints))
        for k, (constraint, value) in enumerate(zip(self.constraints, u)):
            if constraint == "free":
                out[k] = (value, 1.0, 0.0)[order]
            elif constraint == "positive":
                out[k] = np.exp(value)
            else:
                lower, upper = constraint
                s = 1.0 / (1.0 + np.exp(-value))
                out[k] = (
                    lower + (upper - lower) * s,
                    (upper - lower) * s * (1.0 - s),
                    (upper - lower) * s * (1.0 - s) * (1.0 - 2.0 * s),
                )[order]
        return out

    def parameters(self, u: np.ndarray, /) -> np.ndarray:
        """The parameters at free coordinates."""
        return self._each(np.asarray(u, dtype=float), 0)

    def coordinates(self, p: np.ndarray, /) -> np.ndarray:
        """The free coordinates of parameters, each inside its allowed set."""
        out = np.empty(len(self.constraints))
        for k, (constraint, value) in enumerate(zip(self.constraints, p)):
            if constraint == "free":
                out[k] = value
            elif constraint == "positive":
                out[k] = np.log(value)
            else:
                lower, upper = constraint
                s = (value - lower) / (upper - lower)
                out[k] = np.log(s / (1.0 - s))
        return out

    def operator(self) -> Operator:
        """The map as an operator of the library, with two derivatives."""
        space = self.space

        def diagonal(values: np.ndarray) -> LinearOperator:
            return LinearOperator.from_matrix(
                space, space, np.diag(values), form="components"
            )

        return Operator.from_callables(
            space,
            space,
            self.parameters,
            derivative=lambda u: diagonal(self._each(np.asarray(u, dtype=float), 1)),
            second_derivative=lambda u, du: diagonal(
                self._each(np.asarray(u, dtype=float), 2) * np.asarray(du, dtype=float)
            ),
        )
