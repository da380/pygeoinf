"""
Functions on interval domains.

This module provides function objects that live on IntervalDomain in function
spaces, including L² spaces and Sobolev spaces.
"""

import numpy as np
from typing import Union, Callable, Optional
import numbers
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .l2_space import L2Space  # type: ignore[import]


class Function:
    """
    A function in L²([a,b]) with both mathematical and computational aspects.

    This class represents a function in L² space that knows about the L² space
    it belongs to. Functions can be defined via callable rules or basis
    representations.

    This serves as the base class for more specialized function spaces.
    """

    def __init__(self,
                 space: "L2Space",
                 *,
                 coefficients: Optional[np.ndarray] = None,
                 evaluate_callable: Optional[Callable] = None,
                 name: Optional[str] = None,
                 support: Optional[Union[tuple, list]] = None):
        """
        Initialize an L² function.

        Args:
            space: The L²Space this function belongs to
            coefficients: Optional finite-dimensional coefficient
                representation
            evaluate_callable: Optional callable defining the function rule
            name: Optional function name
            support: Optional compact support specification. Can be:
                    - tuple (a, b): single interval where function is nonzero
                    - list of tuples [(a1, b1), (a2, b2), ...]: multiple
                      disjoint intervals where function is nonzero
                    - None: assumes support over entire domain

        Note:
            Exactly one of coefficients or evaluate_callable must be provided.
        """

        # Check that exactly one of coefficients or evaluate_callable is
        # provided
        if (coefficients is None and evaluate_callable is None) or \
           (coefficients is not None and evaluate_callable is not None):
            raise ValueError(
                "Exactly one of 'coefficients' or 'evaluate_callable' "
                "must be provided."
            )

        self.space = space
        self.name = name

        # Support specification - list of disjoint intervals
        self.support = self._check_support(support)

        # Function representation
        self.coefficients = (
            coefficients.copy() if coefficients is not None else None
        )
        self.evaluate_callable = evaluate_callable

    @property
    def function_domain(self):
        """Get the IntervalDomain from the space."""
        return self.space._function_domain

    @property
    def has_compact_support(self) -> bool:
        """Check if function has compact support."""
        return self.support is not None

    def evaluate(self, x: Union[float, np.ndarray],
                 check_domain: Optional[bool] = None
                 ) -> Union[float, np.ndarray]:
        """
        Point evaluation of representative: f(x).

        Args:
            x: Point(s) at which to evaluate
            check_domain: Whether to check domain membership

        Returns:
            Function value(s)
        """

        # Default check_domain behavior: True only if space is provided
        if check_domain is None:
            check_domain = self.space is not None

        # Check domain membership if requested
        if check_domain and self.space is not None:
            x_array = np.asarray(x)
            if not np.all(self.space.function_domain.contains(x_array)):
                raise ValueError(
                    f"Some points not in domain {self.space.function_domain}"
                )

        # Check compact support first - return zero outside support
        if self.has_compact_support:
            is_zero = self._is_zero_at(x)
            if np.any(is_zero):
                # Create appropriate zero array/scalar
                x_array = np.asarray(x)
                zeros = np.zeros_like(x_array, dtype=float)

                # For points outside support, return zero
                if np.all(is_zero):
                    return zeros.item() if x_array.ndim == 0 else zeros

                # For mixed case, evaluate only inside support
                if x_array.ndim == 0:
                    # Scalar case - already handled above
                    pass
                else:
                    # Array case - evaluate only non-zero points
                    result = zeros.copy()
                    if isinstance(is_zero, np.ndarray):
                        inside_support = np.logical_not(is_zero)
                    else:
                        inside_support = not is_zero
                    if np.any(inside_support):
                        x_inside = x_array[inside_support]
                        if self.evaluate_callable is not None:
                            y_inside = self.evaluate_callable(x_inside)
                            result[inside_support] = y_inside
                        elif self.coefficients is not None:
                            result[inside_support] = (
                                self._evaluate_from_coefficients(x_inside)
                            )
                    return result

        # Standard evaluation when no compact support or all points inside
        if self.evaluate_callable is not None:
            return self.evaluate_callable(x)
        elif self.coefficients is not None:
            # Use the space's from_coefficient method to evaluate
            return self._evaluate_from_coefficients(x)
        else:
            raise RuntimeError("No evaluation method available")

    def integrate(self, weight: Optional[Callable] = None,
                  method: str = 'simpson', n_points: int = 100,
                  *, vectorized: Optional[bool] = None) -> float:
        """
        Integrate function over its domain: ∫[a,b] f(x) w(x) dx.

        Leverages compact support for efficient integration.

        Args:
            weight: Optional weight function w(x)
            method: Integration method
            n_points: Number of points for numerical integration

        Returns:
            Integral value
        """
        # If function has compact support, integrate only over support
        # intervals
        if self.has_compact_support:
            if weight is None:
                # Direct integration over all support intervals
                if self.evaluate_callable is not None:
                    return self.function_domain.integrate(
                        self.evaluate_callable, method=method,
                        support=self.support,
                        n_points=n_points,
                        vectorized=vectorized
                    )
                elif self.coefficients is not None:
                    # For basis representations
                    def integrand_coeffs(x):
                        return self.evaluate(x, check_domain=False)
                    return self.function_domain.integrate(
                        integrand_coeffs, method=method,
                        support=self.support,
                        n_points=n_points,
                        vectorized=vectorized
                    )
                else:
                    return 0.0
            else:
                # Weighted integration over all support intervals
                def weighted_integrand(x):
                    return self.evaluate(x, check_domain=False) * weight(x)
                return self.function_domain.integrate(
                    weighted_integrand, method=method,
                    support=self.support,
                    n_points=n_points
                )
        else:
            # No compact support - integrate over entire domain
            if weight is None:
                # Direct integration based on representation
                if self.evaluate_callable is not None:
                    return self.space.function_domain.integrate(
                        self.evaluate_callable,
                        method=method,
                        n_points=n_points,
                        vectorized=vectorized
                    )
                elif self.coefficients is not None:
                    # For basis representations, might have analytical formulas
                    def integrand_full(x):
                        return self.evaluate(x, check_domain=False)
                    return self.space.function_domain.integrate(
                        integrand_full, method=method,
                        n_points=n_points,
                        vectorized=vectorized
                    )
            else:
                def weighted_integrand_full(x):
                    return self.evaluate(x, check_domain=False) * weight(x)
                return self.space.function_domain.integrate(
                    weighted_integrand_full,
                    method=method,
                    n_points=n_points,
                    vectorized=vectorized
                )

        raise RuntimeError("No integration method available")

    def plot(self, n_points: int = 1000, figsize=(10, 6),
             use_seaborn: bool = True, **kwargs):
        """
        Plot the function.

        Args:
            n_points: Number of plot points
            figsize: Figure size as (width, height)
            use_seaborn: Whether to use seaborn styling
            **kwargs: Additional plotting arguments
        """
        import matplotlib.pyplot as plt

        if use_seaborn:
            try:
                import seaborn as sns
                if 'color' not in kwargs:
                    palette = sns.color_palette("muted", 1)
                    kwargs['color'] = palette[0]
                if 'linewidth' not in kwargs:
                    kwargs['linewidth'] = 2
            except ImportError:
                pass  # Fall back to matplotlib defaults

        x = self.function_domain.uniform_mesh(n_points)
        y = self.evaluate(x)

        plt.figure(figsize=figsize)
        plt.plot(x, y, label=self.name or "L² function", **kwargs)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Function on {self.space.function_domain}')
        if self.name:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gca()  # Return axes for further customization

    def copy(self):
        """Custom copy implementation for L2Functions."""
        if self.coefficients is not None:
            return self.__class__(
                self.space,
                coefficients=self.coefficients.copy(),
                name=self.name,
                support=self.support
            )
        else:
            return self.__class__(
                self.space,
                evaluate_callable=self.evaluate_callable,
                name=self.name,
                support=self.support
            )

    def _evaluate_from_coefficients(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate the function at x using the basis and coefficients."""
        # Get the basis functions from the parent space
        basis_functions = self.space.basis_functions
        coeffs = self.coefficients
        if coeffs is None or basis_functions is None:
            raise RuntimeError("Coefficients or basis functions not available "
                               "for evaluation.")
        if len(coeffs) != len(basis_functions):
            raise ValueError(
                f"Coefficient length {len(coeffs)} does not match "
                f"number of basis functions {len(basis_functions)}."
            )

        x_array = np.asarray(x)
        is_scalar = x_array.ndim == 0
        if is_scalar:
            x_array = x_array.reshape(1)

        # Evaluate each basis function at x_array
        basis_evals = np.array([bf.evaluate(x_array, check_domain=False)
                                for bf in basis_functions])
        # basis_evals shape: (n_basis, n_points)
        # Linear combination: sum_k c_k * phi_k(x)
        result = np.tensordot(coeffs, basis_evals, axes=([0], [0]))
        return result[0] if is_scalar else result

    def _is_zero_at(
        self, x: Union[float, np.ndarray]
    ) -> Union[bool, np.ndarray]:
        """
        Check if function is zero at given point(s) due to compact support.

        Args:
            x: Point(s) to check

        Returns:
            Boolean or array indicating if function is zero at x
        """
        if not self.has_compact_support:
            return False  # Function is not zero due to compact support

        x_array = np.asarray(x)
        is_scalar = x_array.ndim == 0

        # Function is zero outside its support intervals
        outside_support = np.ones_like(x_array, dtype=bool)
        for support_a, support_b in self.support:  # type: ignore
            # Points inside this interval are not outside support
            inside_this_interval = (
                (x_array >= support_a) & (x_array <= support_b)
            )
            outside_support = outside_support & (~inside_this_interval)

        return outside_support.item() if is_scalar else outside_support

    @staticmethod
    def _union_supports(support1, support2):
        """
        Compute union of two support specifications.

        Args:
            support1, support2: Lists of intervals or None

        Returns:
            List of disjoint intervals representing the union, or None
        """
        if support1 is None and support2 is None:
            return None
        if support1 is None:
            return support2
        if support2 is None:
            return support1

        # Merge and sort all intervals
        all_intervals = support1 + support2
        all_intervals.sort(key=lambda x: x[0])

        # Merge overlapping intervals
        merged = []
        for start, end in all_intervals:
            if merged and start <= merged[-1][1]:
                # Overlapping or adjacent - merge
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                # Non-overlapping - add new interval
                merged.append((start, end))

        return merged

    @staticmethod
    def _intersect_supports(support1, support2):
        """
        Compute intersection of two support specifications.

        Args:
            support1, support2: Lists of intervals or None

        Returns:
            List of disjoint intervals representing intersection, or None
        """
        if support1 is None or support2 is None:
            return None

        intersections = []
        for a1, b1 in support1:
            for a2, b2 in support2:
                # Find intersection of [a1, b1] and [a2, b2]
                intersect_a = max(a1, a2)
                intersect_b = min(b1, b2)
                if intersect_a < intersect_b:
                    intersections.append((intersect_a, intersect_b))

        if not intersections:
            return []

        # Sort and return (should already be disjoint by construction)
        intersections.sort(key=lambda x: x[0])
        return intersections

    def _check_support(self, support):
        if support is not None:
            if isinstance(support, tuple):
                # Convert single tuple to list for backward compatibility
                if len(support) != 2 or support[0] >= support[1]:
                    raise ValueError("Support tuple must be (a, b) with a < b")
                support = [support]
            elif isinstance(support, list):
                # Validate list of tuples
                for i, interval in enumerate(support):
                    if not isinstance(interval, tuple) or len(interval) != 2:
                        raise ValueError(
                            f"Support interval {i} must be a tuple (a, b)"
                        )
                    if interval[0] >= interval[1]:
                        raise ValueError(
                            (
                                f"Support interval {i}: a={interval[0]} "
                                f"must be < b={interval[1]}"
                            )
                        )
                # Sort intervals and check for overlaps
                support = sorted(support, key=lambda x: x[0])
                for i in range(len(support) - 1):
                    if support[i][1] > support[i+1][0]:
                        raise ValueError(
                            f"Support intervals {support[i]} and "
                            f"{support[i+1]} overlap"
                        )
            else:
                raise ValueError(
                    "Support must be a tuple (a, b) or list of tuples "
                    "[(a1, b1), (a2, b2), ...]"
                )

            # Ensure all support intervals are within domain
            domain_a = self.space.function_domain.a
            domain_b = self.space.function_domain.b
            for i, (a, b) in enumerate(support):
                if not (
                    domain_a <= a < b <= domain_b
                ):
                    raise ValueError(
                        f"Support interval {i}: ({a}, {b})"
                        f" must be within domain "
                        f"[{domain_a}, {domain_b}]"
                    )
        return support

    def __call__(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Allow f(x) syntax."""
        return self.evaluate(x, check_domain=None)

    def _binary_op(
            self, other, op, op_name,
            scalar_allowed=True,
            support_strategy='union'
            ):
        """
        General helper for binary operations (add, sub, mul).
        Args:
            other: Function or scalar
            op: Callable for the operation (e.g., operator.add)
            op_name: String for error messages
            scalar_allowed: Whether to allow scalar as other
            support_strategy: 'union' (add/sub), 'intersect' (mul)
        """
        if isinstance(other, Function):
            if self.space != other.space:
                raise ValueError(
                    f"Cannot {op_name} functions from different spaces. "
                    f"Got spaces: {self.space} and {other.space}"
                )
            if support_strategy == 'union':
                new_support = self._union_supports(self.support, other.support)
            elif support_strategy == 'intersect':
                new_support = self._intersect_supports(
                    self.support, other.support
                )
            else:
                new_support = None
            # Try coefficient operation if possible
            if (self.coefficients is not None and
                    other.coefficients is not None and
                    len(self.coefficients) == len(other.coefficients)):
                new_coeffs = op(self.coefficients, other.coefficients)
                return self.__class__(
                    self.space, coefficients=new_coeffs, support=new_support
                )
            else:
                def op_callable(x):
                    return op(self.evaluate(x), other.evaluate(x))
                return self.__class__(
                    self.space,
                    evaluate_callable=op_callable,
                    support=new_support
                )
        elif scalar_allowed and isinstance(other, numbers.Number):
            def scalar_op_callable(x):
                return op(self.evaluate(x), other)
            return self.__class__(
                self.space,
                evaluate_callable=scalar_op_callable,
                support=self.support
            )
        else:
            raise TypeError(
                (
                    f"Can only {op_name} Function or scalar "
                    f"(int, float, numbers.Number), not {type(other)}"
                )
            )

    def __add__(self, other):
        import operator
        return self._binary_op(
            other,
            operator.add,
            'add',
            scalar_allowed=True,
            support_strategy='union'
        )

    def __radd__(self, other):
        """Right addition to support scalar + Function."""
        return self.__add__(other)

    def __sub__(self, other):
        import operator
        return self._binary_op(
            other,
            operator.sub,
            'subtract',
            scalar_allowed=True,
            support_strategy='union'
        )

    def __mul__(self, other):
        import operator
        # Special case: if both have compact support and intersection is None,
        # return zero function
        if (
            isinstance(other, Function)
            and self.has_compact_support
            and other.has_compact_support
        ):
            new_support = self._intersect_supports(self.support, other.support)
            if new_support is None:
                def zero_callable(x):
                    return np.zeros_like(np.asarray(x), dtype=float)
                return Function(
                    self.space,
                    evaluate_callable=zero_callable
                )
        return self._binary_op(
            other,
            operator.mul,
            'multiply',
            scalar_allowed=True,
            support_strategy='intersect'
        )

    def __rmul__(self, other):
        """Right multiplication (for scalar * function)."""
        return self.__mul__(other)

    def __repr__(self) -> str:
        return (f"Function(domain={self.space.function_domain}, "
                f"name={self.name})")
