"""A finite-dimensional dynamical system and the derivatives of its rule.

The system is ``dx/dt = f(x, p) + G w(t)``: a state ``x`` of a few
components, a rule ``f`` that depends on a few parameters ``p``, and an
optional forcing ``w`` that enters through a fixed matrix ``G``. Nothing
here knows which system it is. The rule is given once, as a function of
arrays, and its derivatives in the state and in the parameters are taken
by the complex step: for an analytic rule, ``Im f(x + ih e_k) / h`` is the
derivative along ``e_k`` to rounding, with no subtraction and so no step to
tune. The rule must therefore be written with operations that accept
complex arrays (arithmetic and NumPy's elementary functions), and must
broadcast over a trailing axis, so that every perturbation is taken in one
call.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# The complex step. Any value far below the square root of the smallest
# number a derivative is to resolve will do; nothing is subtracted.
STEP = 1e-30
# The real step of the second derivatives, a central difference of
# complex-step first derivatives.
SECOND_STEP = 1e-5

Rule = Callable[[np.ndarray, np.ndarray], np.ndarray]


class DynamicalSystem:
    """``dx/dt = f(x, p) + G w(t)``, with the derivatives of ``f``."""

    def __init__(
        self,
        rule: Rule,
        state_dim: int,
        parameter_dim: int,
        /,
        *,
        forcing_matrix: np.ndarray | None = None,
        state_names: Sequence[str] | None = None,
        parameter_names: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            rule: ``f(x, p)``, taking the state as an array of shape
                ``(state_dim, ...)`` and the parameters as one of shape
                ``(parameter_dim, ...)``, real or complex, and returning the
                rate of change in the state's shape.
            state_dim: the number of state components.
            parameter_dim: the number of parameters.
            forcing_matrix: ``G``, of shape ``(state_dim, forcing_dim)``; a
                system with none takes no forcing.
            state_names: a name for each state component.
            parameter_names: a name for each parameter.

        Raises:
            ValueError: if the forcing matrix has the wrong number of rows.
        """
        self._rule = rule
        self.state_dim = int(state_dim)
        self.parameter_dim = int(parameter_dim)
        if forcing_matrix is None:
            forcing_matrix = np.zeros((self.state_dim, 0))
        self.forcing_matrix = np.asarray(forcing_matrix, dtype=float)
        if self.forcing_matrix.shape[0] != self.state_dim:
            raise ValueError("The forcing matrix needs a row for each state component.")
        self.forcing_dim = self.forcing_matrix.shape[1]
        self.state_names = tuple(
            state_names or (f"x{k}" for k in range(self.state_dim))
        )
        self.parameter_names = tuple(
            parameter_names or (f"p{k}" for k in range(self.parameter_dim))
        )
        # How many times the rule was called, a batch of perturbations
        # counting once.
        self.calls = 0

    def rate(self, x: np.ndarray, p: np.ndarray, /) -> np.ndarray:
        """``f(x, p)``."""
        self.calls += 1
        return np.asarray(self._rule(x, p), dtype=float)

    def jacobians(
        self, x: np.ndarray, p: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """``df/dx`` and ``df/dp`` at a state and parameters, in one call of the rule.

        Returns:
            The two matrices, of shapes ``(state_dim, state_dim)`` and
            ``(state_dim, parameter_dim)``.
        """
        n, m = self.state_dim, self.parameter_dim
        states = np.repeat(np.asarray(x, dtype=complex)[:, None], n + m, axis=1)
        parameters = np.repeat(np.asarray(p, dtype=complex)[:, None], n + m, axis=1)
        states[np.arange(n), np.arange(n)] += 1j * STEP
        parameters[np.arange(m), n + np.arange(m)] += 1j * STEP
        self.calls += 1
        slopes = np.imag(self._rule(states, parameters)) / STEP
        return slopes[:, :n], slopes[:, n:]

    def jacobians_at_many(self, x: np.ndarray, p: np.ndarray, /) -> np.ndarray:
        """Both Jacobians at several states at once, in one call of the rule.

        Args:
            x: the states, shape ``(state_dim, count)``.
            p: the parameters.

        Returns:
            Shape ``(count, state_dim, state_dim + parameter_dim)``: for
            each state, ``df/dx`` and then ``df/dp``, side by side.
        """
        n, m = self.state_dim, self.parameter_dim
        count = np.shape(x)[1]
        states = np.repeat(np.asarray(x, dtype=complex)[:, :, None], n + m, axis=2)
        parameters = np.empty((m, count, n + m), dtype=complex)
        parameters[:] = np.asarray(p, dtype=float)[:, None, None]
        states[np.arange(n), :, np.arange(n)] += 1j * STEP
        parameters[np.arange(m), :, n + np.arange(m)] += 1j * STEP
        self.calls += 1
        slopes = np.imag(self._rule(states, parameters)) / STEP
        return slopes.transpose(1, 0, 2)

    def second_directional(
        self, x: np.ndarray, p: np.ndarray, dx: np.ndarray, dp: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray]:
        """The derivative of both Jacobians along ``(dx, dp)``.

        A central difference of complex-step Jacobians, so accurate to about
        ten figures where the Jacobians themselves are exact.

        Returns:
            ``d(df/dx)[dx, dp]`` and ``d(df/dp)[dx, dp]``, shaped as the
            Jacobians are.
        """
        size = max(np.linalg.norm(dx), np.linalg.norm(dp), 1e-300)
        scale = max(np.linalg.norm(x), np.linalg.norm(p), 1.0)
        step = SECOND_STEP * scale / size
        ahead = self.jacobians(x + step * dx, p + step * dp)
        behind = self.jacobians(x - step * dx, p - step * dp)
        return (
            (ahead[0] - behind[0]) / (2.0 * step),
            (ahead[1] - behind[1]) / (2.0 * step),
        )
