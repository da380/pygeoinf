"""Gradient operator for interval domains."""

import logging
from typing import Optional

import numpy as np

from pygeoinf.linear_operators import LinearOperator

from ..functions import Function


class Gradient(LinearOperator):
    """
    The gradient operator (d/dx) on interval domains.

    In 1D, the gradient is simply the first derivative.

    - 'finite_difference': Numerical differentiation using finite differences
    """

    def __init__(
        self,
        domain: "Sobolev",
        /,
        *,
        fd_order: int = 2,
        fd_step: Optional[float] = None,
        boundary_treatment: str = 'one_sided'
    ):
        """
        Initialize the gradient operator.

        Args:
            domain: Function space (Lebesgue or SobolevSpace)
            fd_order: Order of finite difference stencil (2, 4, 6)
                for FD method
            fd_step: Step size for finite differences (auto-computed if None)
            boundary_treatment: How to handle boundaries for FD
                ('one_sided', 'extrapolate')
        """
        self._domain = domain

        # Create codomain with s-1 regularity
        self._codomain = domain
        self._fd_order = fd_order
        self._fd_step = fd_step
        self._boundary_treatment = boundary_treatment
        # logger
        self._log = logging.getLogger(__name__)

        # No boundary conditions needed for gradient (unlike Laplacian)
        super().__init__(domain, domain, self._apply)

        # Initialize method-specific components
        self._setup_finite_difference()

    def _setup_finite_difference(self):
        """Setup finite difference method."""
        # Determine step size if not provided
        if self._fd_step is None:
            a, b = self._domain.function_domain.a, \
                   self._domain.function_domain.b
            # Use a fraction of the domain size
            self._fd_step = (b - a) / 1000

        self._log.debug(
            "GradientOperator (finite difference, order %s) initialized with "
            "step size %.2e",
            self._fd_order,
            self._fd_step,
        )

    # ------------------------------------------------------------------
    # Small helpers for finite-difference evaluation
    # ------------------------------------------------------------------
    def _fd_stencil(self, order: int, location: str = "center"):
        """Return (offsets, coeffs) for first-derivative finite-difference.

        offsets are integer multiples of h; coeffs are the weights to be
        applied and divided by h when computing df/dx.
        """
        if order == 2:
            if location == "center":
                return np.array([-1, 1]), np.array([-0.5, 0.5])
            if location == "forward":
                # forward 2nd-order: (-3/2, 2, -1/2)
                return np.array([0, 1, 2]), np.array([-1.5, 2.0, -0.5])
            if location == "backward":
                return np.array([-2, -1, 0]), np.array([0.5, -2.0, 1.5])
        if order == 4:
            if location == "center":
                # coefficients for central 4th-order: 1/12, -2/3, 2/3,
                # -1/12
                offsets = np.array([-2, -1, 1, 2])
                coeffs = np.array([1/12, -2/3, 2/3, -1/12])
                return offsets, coeffs
            # For one-sided 4th-order coefficients, fall back to 2nd-order
            if location in ("forward", "backward"):
                return self._fd_stencil(2, location)
        raise ValueError(
            f"Unsupported fd_order={order} or location={location}"
        )

    def _safe_eval(self, func: Function, x: np.ndarray) -> np.ndarray:
        """Evaluate Function `func` on array x safely.

        Tries to call func(x) directly; if that fails (user's Function only
        accepts scalars), falls back to list comprehension.
        """
        try:
            return np.asarray(func(x))
        except Exception:
            # Fall back to Python loop
            return np.asarray([func(float(xi)) for xi in x])

    def _apply(self, f: Function) -> Function:
        """Apply gradient using finite difference method (vectorized).

        This implementation evaluates the user's function on shifted arrays
        (when possible) which greatly reduces Python-level loops and is much
        faster for array inputs. It falls back to scalar evaluation if
        necessary.
        """

        def gradient_func(x):
            scalar_input = np.isscalar(x)
            x_arr = np.asarray([x]) if scalar_input else np.asarray(x)

            h = self._fd_step
            a = self._domain.function_domain.a
            b = self._domain.function_domain.b

            # Prepare output container
            y = np.empty_like(x_arr, dtype=float)

            # Masks for boundary/interior
            left_mask = x_arr <= a + h
            right_mask = x_arr >= b - h
            interior_mask = ~(left_mask | right_mask)

            # Interior points: central stencil
            if np.any(interior_mask):
                xi = x_arr[interior_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "center")
                shifts = (xi[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                # vals.shape == (len(offs), n_points)
                y[interior_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            # Left boundary: forward one-sided
            if np.any(left_mask):
                xi = x_arr[left_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "forward")
                shifts = (xi[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                y[left_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            # Right boundary: backward one-sided
            if np.any(right_mask):
                xi = x_arr[right_mask]
                offs, coeffs = self._fd_stencil(self._fd_order, "backward")
                shifts = (xi[None, :] + offs[:, None] * h)
                vals = self._safe_eval(f, shifts)
                y[right_mask] = (coeffs[:, None] * vals).sum(axis=0) / h

            return float(y[0]) if scalar_input else y

        return Function(
            self.codomain,
            evaluate_callable=gradient_func,
            name=f"∇({getattr(f, 'name', 'f')})",
        )
