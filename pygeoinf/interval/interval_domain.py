"""Compact IntervalDomain used in the interval utilities.

This module provides a minimal IntervalDomain class. Sampling
helpers were removed from the public API on purpose.
"""

from typing import Callable, Optional, Tuple, Union

import math
import numpy as np


class IntervalDomain:
    def __init__(
        self,
        a: float,
        b: float,
        *,
        boundary_type: str = "closed",
        name: Optional[str] = None,
    ):
        if a >= b:
            raise ValueError("a must be < b")
        self.a = float(a)
        self.b = float(b)
        self.boundary_type = boundary_type
        # name defaults to the string representation used in tests
        if name is None:
            self.name = self._format_name()
        else:
            self.name = name

    @property
    def length(self) -> float:
        return self.b - self.a

    @property
    def center(self) -> float:
        return 0.5 * (self.a + self.b)

    @property
    def radius(self) -> float:
        return 0.5 * self.length

    def contains(self, x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        if self.boundary_type == "closed":
            return (x >= self.a) & (x <= self.b)
        if self.boundary_type == "open":
            return (x > self.a) & (x < self.b)
        if self.boundary_type == "left_open":
            return (x > self.a) & (x <= self.b)
        if self.boundary_type == "right_open":
            return (x >= self.a) & (x < self.b)
        raise ValueError("unknown boundary_type")

    def uniform_mesh(self, n: int) -> np.ndarray:
        if self.boundary_type == "closed":
            return np.linspace(self.a, self.b, n, endpoint=True)
        if self.boundary_type == "open":
            return np.linspace(self.a, self.b, n + 2)[1:-1]
        if self.boundary_type == "left_open":
            return np.linspace(self.a, self.b, n + 1)[1:]
        if self.boundary_type == "right_open":
            return np.linspace(self.a, self.b, n, endpoint=False)
        raise ValueError("unknown boundary_type")

    def interior(self) -> "IntervalDomain":
        return IntervalDomain(self.a, self.b, boundary_type="open")

    def closure(self) -> "IntervalDomain":
        return IntervalDomain(self.a, self.b, boundary_type="closed")

    def boundary_points(self) -> Tuple[float, float]:
        return (self.a, self.b)

    def integrate(
        self,
        f: Callable,
        method: str = "simpson",
        support: Optional[Tuple[float, float]] = None,
        n_points: int = 100,
        *,
        vectorized: Optional[bool] = None,
        **kwargs,
    ) -> float:
        # support may be a sequence of subintervals (e.g. [(a1,b1), (a2,b2)])
        if (
            support is not None
            and hasattr(support, "__iter__")
            and not (
                isinstance(support, (tuple, list))
                and len(support) == 2
                and all(isinstance(x, (int, float)) for x in support)
            )
        ):
            # treat as sequence of (a,b) pairs
            subintervals = list(support)
            if len(subintervals) == 0:
                return 0.0

            # compute lengths and ensure subintervals are valid and inside domain
            lengths = []
            for sub in subintervals:
                if not (isinstance(sub, (tuple, list)) and len(sub) == 2):
                    raise ValueError("each support entry must be a (a,b) pair")
                a_sub, b_sub = float(sub[0]), float(sub[1])
                if not (self.a <= a_sub < b_sub <= self.b):
                    raise ValueError("support outside domain")
                lengths.append(b_sub - a_sub)

            total_length = sum(lengths)
            if total_length <= 0:
                return 0.0

            # enforce at least 3 points per subinterval; if n_points is too small,
            # increase the effective total so every interval gets the minimum
            n_sub = len(subintervals)
            effective_total = max(n_points, 3 * n_sub)

            # raw allocation proportional to length
            raw = [effective_total * (L / total_length) for L in lengths]

            # start with floored allocation and ensure minimum 3
            alloc = [max(3, int(math.floor(r))) for r in raw]
            allocated = sum(alloc)

            # distribute any remaining points by largest fractional parts
            remainder = effective_total - allocated
            if remainder > 0:
                fracs = sorted(
                    [(raw[i] - math.floor(raw[i]), i) for i in range(n_sub)],
                    key=lambda x: x[0],
                    reverse=True,
                )
                idx = 0
                while remainder > 0:
                    alloc[fracs[idx % n_sub][1]] += 1
                    remainder -= 1
                    idx += 1

            # call integrate on each subinterval with its allocated point count
            total = 0.0
            for sub, n_i in zip(subintervals, alloc):
                total += self.integrate(
                    f,
                    method=method,
                    support=sub,
                    n_points=n_i,
                    vectorized=vectorized,
                    **kwargs,
                )
            return float(total)

        if support is None:
            # Use uniform_mesh which respects boundary_type
            xs = self.uniform_mesh(max(3, n_points))
        else:
            a, b = support
            if not (self.a <= a < b <= self.b):
                raise ValueError("support outside domain")
            # For subinterval integration, always use closed interval
            xs = np.linspace(a, b, max(3, n_points))

        def eval_mesh(xs_vals: np.ndarray) -> np.ndarray:
            if vectorized is True:
                return np.asarray(f(xs_vals))
            if vectorized is False:
                return np.fromiter((f(x) for x in xs_vals),
                                   dtype=float,
                                   count=xs_vals.size)
            try:
                out = f(xs_vals)
                arr = np.asarray(out)
                if arr.shape == ():
                    raise ValueError
                return arr
            except Exception:
                return np.fromiter((f(x) for x in xs_vals),
                                   dtype=float,
                                   count=xs_vals.size)

        ys = eval_mesh(xs)

        if method == "simpson":
            try:
                from scipy.integrate import simpson
            except Exception as exc:  # pragma: no cover - SciPy required
                raise ImportError(
                    "scipy is required for 'simpson' integration"
                ) from exc

            return float(simpson(ys, x=xs))

        if method == "trapz":
            try:
                from scipy.integrate import trapezoid as trapz
            except ImportError:
                # Fallback for older SciPy versions
                try:
                    from scipy.integrate import trapz
                except Exception as exc:  # pragma: no cover - SciPy required
                    raise ImportError(
                        "scipy is required for 'trapz' integration"
                    ) from exc

            return float(trapz(ys, x=xs))

        if method == "adaptive":
            try:
                from scipy.integrate import quad
            except Exception as exc:  # pragma: no cover - SciPy required
                raise ImportError(
                    "scipy is required for 'adaptive' integration"
                ) from exc

            return float(quad(f, a, b, **kwargs)[0])

        raise ValueError("unknown method")

    def restriction_to_subinterval(
        self,
        a: float,
        b: float,
    ) -> "IntervalDomain":
        if a >= b:
            raise ValueError("invalid subinterval")
        if not (self.a <= a and b <= self.b):
            raise ValueError("subinterval outside domain")
        return IntervalDomain(a, b, boundary_type=self.boundary_type)

    def _format_name(self) -> str:
        if self.boundary_type == "closed":
            return f"[{self.a}, {self.b}]"
        if self.boundary_type == "open":
            return f"({self.a}, {self.b})"
        if self.boundary_type == "left_open":
            return f"({self.a}, {self.b}]"
        if self.boundary_type == "right_open":
            return f"[{self.a}, {self.b})"
        return f"[{self.a}, {self.b}]"

    def __repr__(self) -> str:
        return self._format_name()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntervalDomain):
            return False
        return (
            (self.a == other.a)
            and (self.b == other.b)
            and (self.boundary_type == other.boundary_type)
        )
