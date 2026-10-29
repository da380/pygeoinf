"""A coordinate-free Sobolev space over C++ vectors.

The vectors are ``cppvec.Vector`` objects: a ``std::vector<double>`` behind a
nanobind class that exposes no buffer, no indexing and no arithmetic
operators. The space is the piecewise-linear finite element space on a
uniform mesh of ``[0, 1]`` under the ``H^1`` inner product

    (x, y) = x^T (M + l^2 K) y,

with ``M`` the mass matrix, ``K`` the stiffness matrix and ``l`` a length of
the space, and it is a plain :class:`~pygeoinf2.algebra.spaces.HilbertSpace`:
no component map, no Gram matrix the library can read, nothing NumPy can
touch. Every vector calculation the library asks for is answered by the C++
side, and the space counts what it was asked, so a run can say exactly which
members the library used.

The mesh and its matrices are assembled here from element matrices, as lists
of triplets handed to ``cppvec.SparseMatrix``; the Gram matrix ``M + l^2 K``
is factorized once by the C++ banded Cholesky, and its solve is the Riesz map
of the space. The operators built on top -- point evaluation, local averages
and the covariance of a Matérn-type prior with its factor and precision --
are written from callables with their adjoints taken through that solve.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter
from collections.abc import Sequence
from typing import Any, Hashable

import numpy as np
from numpy.random import Generator

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace, HilbertSpace, resolve_rng
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.traits import Traits

try:
    import cppvec
except ImportError:  # the built module sits next to this file
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    try:
        import cppvec
    finally:
        sys.path.pop(0)

__all__ = [
    "CppSpace",
    "point_evaluation_operator",
    "local_average_operator",
    "matern_measure",
]

_DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE

# The element matrices of piecewise-linear elements on a unit element: the
# mass matrix is h times the first, the stiffness matrix 1/h times the second.
_MASS = ((1.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 1.0 / 3.0))
_STIFFNESS = ((1.0, -1.0), (-1.0, 1.0))


def _matrix(n_elements: int, mass: float, stiffness: float, /) -> Any:
    """``mass M + stiffness K`` on the mesh, assembled from element matrices
    into a ``cppvec.SparseMatrix``."""
    h = 1.0 / n_elements
    rows, cols, values = [], [], []
    for e in range(n_elements):
        for a in range(2):
            for b in range(2):
                rows.append(e + a)
                cols.append(e + b)
                values.append(mass * h * _MASS[a][b] + stiffness / h * _STIFFNESS[a][b])
    n = n_elements + 1
    return cppvec.SparseMatrix(n, n, rows, cols, values)


class CppSpace(HilbertSpace):
    """Piecewise-linear finite elements on ``[0, 1]`` under the ``H^1`` inner
    product, over ``cppvec.Vector``.

    The inner product is ``(x, y) = x^T G y`` with ``G = M + l^2 K``,
    computed by the vector's own weighted dot product; the Riesz map ``G^-1``
    and the white-noise draw go through the C++ Cholesky factor of ``G``.
    The space is not a ``CoordinateSpace``: nothing here maps a vector to an
    array. At length zero the inner product is the mass matrix alone, the
    space ``L^2``, on which a value at a point is not a bounded functional;
    the app builds that space only to show the difference.

    Every call of a member is tallied in :attr:`calls`, by name, so that a
    run can report which operations the library asked of the backend.
    """

    def __init__(self, n_elements: int, length_scale: float) -> None:
        """
        Args:
            n_elements: the number of elements; the space has one more node.
            length_scale: ``l``, the length that weighs the derivative
                against the value in the norm.

        Raises:
            ValueError: for fewer than one element or a negative length.
        """
        if n_elements < 1:
            raise ValueError("A mesh needs at least one element.")
        if length_scale < 0.0:
            raise ValueError(f"The length scale is not negative, got {length_scale}.")
        self._n_elements = int(n_elements)
        self._n_nodes = self._n_elements + 1
        self._length_scale = float(length_scale)
        self._mass = _matrix(self._n_elements, 1.0, 0.0)
        self._stiffness = _matrix(self._n_elements, 0.0, 1.0)
        self._gram = _matrix(self._n_elements, 1.0, self._length_scale**2)
        self._gram_cholesky = self._gram.cholesky()
        self._mass_cholesky = self._mass.cholesky()
        self.calls: Counter[str] = Counter()

    # ----------------------------------------------------------------- #
    #                              Structure                            #
    # ----------------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """The number of nodes."""
        self.calls["dim"] += 1
        return self._n_nodes

    @property
    def n_elements(self) -> int:
        """The number of elements."""
        return self._n_elements

    @property
    def length_scale(self) -> float:
        """The length ``l`` of the inner product."""
        return self._length_scale

    @property
    def is_function_space(self) -> bool:
        """True: the space discretizes functions on ``[0, 1]``."""
        return True

    def _key(self) -> Hashable:
        self.calls["_key"] += 1
        return ("CppSpace", self._n_elements, self._length_scale)

    def __repr__(self) -> str:
        return (
            f"CppSpace(elements={self._n_elements}, "
            f"length_scale={self._length_scale:g}, vectors=cppvec.Vector)"
        )

    @property
    def mass(self) -> Any:
        """The mass matrix ``M``, a ``cppvec.SparseMatrix``."""
        return self._mass

    @property
    def gram(self) -> Any:
        """The Gram matrix ``M + l^2 K`` of the inner product."""
        return self._gram

    @property
    def gram_cholesky(self) -> Any:
        """The C++ Cholesky factor of the Gram matrix."""
        return self._gram_cholesky

    @property
    def mass_cholesky(self) -> Any:
        """The C++ Cholesky factor of the mass matrix."""
        return self._mass_cholesky

    def nodes(self) -> list[float]:
        """The node positions, for plotting and for building functionals."""
        return [i / self._n_elements for i in range(self._n_nodes)]

    # ----------------------------------------------------------------- #
    #                          Vector operations                        #
    # ----------------------------------------------------------------- #

    def zero(self) -> Any:
        """A new zero vector."""
        self.calls["zero"] += 1
        return cppvec.Vector(self._n_nodes)

    def copy(self, x: Any) -> Any:
        """An independent copy, made by C++."""
        self.calls["copy"] += 1
        return x.copy()

    def inner_product(self, x: Any, y: Any) -> float:
        """``x^T G y``, with ``G`` applied by the sparse matrix."""
        self.calls["inner_product"] += 1
        return x.dot_weighted(y, self._gram)

    def axpy(self, a: float, x: Any, y: Any) -> Any:
        """``y += a x`` in place, and return ``y``."""
        self.calls["axpy"] += 1
        y.axpy(float(a), x)
        return y

    def scale_inplace(self, a: float, x: Any) -> Any:
        """``x *= a`` in place, and return ``x``."""
        self.calls["scale_inplace"] += 1
        x.scale(float(a))
        return x

    def random(self, *, rng: Generator | None = None) -> Any:
        """Independent standard normal nodal values, drawn by C++.

        Only a seed crosses from the NumPy generator to the vector; the
        values never exist on the Python side.
        """
        self.calls["random"] += 1
        return self._standard_normal(rng)

    def white_noise(self, *, rng: Generator | None = None) -> Any:
        """White noise with respect to the inner product.

        With ``G = R R^T`` and ``z`` standard normal, ``R^-T z`` has
        covariance ``G^-1`` on the nodal values, which is the identity on the
        space. Applying ``R`` instead would give covariance ``G``: the wrong
        answer by two factors of the metric, which is what
        ``check_white_noise`` is for.
        """
        self.calls["white_noise"] += 1
        return self._gram_cholesky.solve_factor_transpose(self._standard_normal(rng))

    def _standard_normal(self, rng: Generator | None, /) -> Any:
        """Standard normal nodal values from a seed, uncounted."""
        vector = cppvec.Vector(self._n_nodes)
        vector.fill_random(int(resolve_rng(rng).integers(2**63)))
        return vector

    # ----------------------------------------------------------------- #
    #                          The Riesz map                            #
    # ----------------------------------------------------------------- #

    def riesz(self, b: Any) -> Any:
        """``G^-1 b``: the representer of the functional ``x -> b^T x``."""
        self.calls["riesz"] += 1
        return self._gram_cholesky.solve(b)

    def interpolate(self, values: Sequence[float]) -> Any:
        """A vector from nodal values, for building functionals."""
        if len(values) != self._n_nodes:
            raise ValueError(
                f"Expected {self._n_nodes} nodal values, got {len(values)}."
            )
        return cppvec.Vector.from_list([float(v) for v in values])

    @staticmethod
    def values(x: Any) -> np.ndarray:
        """A vector's nodal values as an array: for the figures only."""
        return np.array(x.to_list())


# --------------------------------------------------------------------- #
#                         Functionals and operators                     #
# --------------------------------------------------------------------- #


def _functional_operator(
    space: CppSpace, coefficients: list[Any], /, *, name: str
) -> LinearOperator:
    """The operator ``x -> (c_j^T x)_j`` into Euclidean space, and its adjoint.

    Each coefficient vector ``c_j`` defines the functional ``x -> c_j^T x``
    on nodal values. Its representer in the inner product is ``G^-1 c_j``,
    so the adjoint of the operator is ``y -> G^-1 (sum_j y_j c_j)``: one
    Riesz map however many functionals.
    """
    data_space = EuclideanSpace(len(coefficients))

    def value(x: Any) -> np.ndarray:
        space.calls[f"{name}.value"] += 1
        return data_space.from_components(np.array([c.dot(x) for c in coefficients]))

    def adjoint(y: np.ndarray) -> Any:
        space.calls[f"{name}.adjoint"] += 1
        load = cppvec.Vector(space.n_elements + 1)
        for weight, c in zip(data_space.to_components(y), coefficients):
            load.axpy(float(weight), c)
        return space.riesz(load)

    return LinearOperator.from_callables(space, data_space, value, adjoint=adjoint)


def _evaluation_weights(space: CppSpace, point: float, /) -> list[float]:
    """The nodal weights of the value at a point: the two nodes of the
    element containing it, by linear interpolation."""
    if not 0.0 <= point <= 1.0:
        raise ValueError(f"The point {point} is outside [0, 1].")
    n = space.n_elements
    element = min(int(point * n), n - 1)
    local = point * n - element
    weights = [0.0] * (n + 1)
    weights[element] = 1.0 - local
    weights[element + 1] = local
    return weights


def point_evaluation_operator(
    space: CppSpace, points: Sequence[float], /
) -> LinearOperator:
    """Evaluation at points of ``[0, 1]``, into ``EuclideanSpace(len(points))``.

    A piecewise-linear function is evaluated by interpolating between the two
    nodes of the element containing the point, so each functional has two
    nonzero coefficients. It is bounded on the space for any positive
    length scale, and its norm settles under refinement; at length zero,
    in ``L^2``, the norm grows as the root of the number of elements.

    Args:
        space: the space.
        points: positions in ``[0, 1]``.

    Raises:
        ValueError: for a point outside the interval.
    """
    coefficients = [
        space.interpolate(_evaluation_weights(space, point)) for point in points
    ]
    return _functional_operator(space, coefficients, name="point_evaluation")


def local_average_operator(
    space: CppSpace, intervals: Sequence[tuple[float, float]], /
) -> LinearOperator:
    """Averages over subintervals of ``[0, 1]``, into Euclidean space.

    The average of ``u`` over ``[a, b]`` is ``(1 / (b - a)) ∫_a^b u dx``, and
    the integral of each hat function over the subinterval is taken exactly.

    Args:
        space: the space.
        intervals: pairs ``(a, b)`` with ``0 <= a < b <= 1``.

    Raises:
        ValueError: for a malformed interval.
    """
    n, h = space.n_elements, 1.0 / space.n_elements

    def hat_integral(node: int, a: float, b: float) -> float:
        # ∫_a^b φ_node dx, with φ_node the hat function centered at node.
        total = 0.0
        for e, rising in ((node - 1, True), (node, False)):
            if not 0 <= e < n:
                continue
            lo, hi = max(a, e * h), min(b, (e + 1) * h)
            if hi <= lo:
                continue
            # On element e, φ_node is linear: (x - x_e)/h rising, or
            # 1 - (x - x_e)/h falling.
            x_e = e * h

            def primitive(x: float) -> float:
                s = (x - x_e) / h
                return h * (s * s / 2.0 if rising else s - s * s / 2.0)

            total += primitive(hi) - primitive(lo)
        return total

    coefficients = []
    for a, b in intervals:
        if not 0.0 <= a < b <= 1.0:
            raise ValueError(f"The interval ({a}, {b}) is not inside [0, 1].")
        weights = [hat_integral(i, a, b) / (b - a) for i in range(n + 1)]
        coefficients.append(space.interpolate(weights))
    return _functional_operator(space, coefficients, name="local_average")


# --------------------------------------------------------------------- #
#                               The prior                               #
# --------------------------------------------------------------------- #


def matern_measure(
    space: CppSpace, /, *, length_scale: float, pointwise_std: float
) -> GaussianMeasure:
    """A Gaussian measure on the space, the Matérn-type field
    ``σ^2 (I + ℓ^2 Δ)^-2``.

    With ``A = M + ℓ^2 K`` the Helmholtz matrix of the prior's length, the
    field's nodal values have covariance ``σ^2 A^-1 M A^-1``, the law of
    ``A u = σ M^(1/2) z``. As an operator on the space, whose inner product
    is ``x^T G y``, the covariance is

        Q = σ^2 A^-1 M A^-1 G,

    self-adjoint there because ``G Q`` is symmetric, of order one relative
    to the space's own metric and so trace class on it in one dimension. Its
    factor ``L = σ A^-1 R_M R_G^T``, with ``M = R_M R_M^T`` and
    ``G = R_G R_G^T`` the C++ Cholesky factors, maps the space to itself and
    satisfies ``L L* = Q``, the adjoint in the space's inner product being
    ``L* = σ R_G^-T R_M^T A^-1 G``. Its precision
    ``Q^-1 = σ^-2 G^-1 A M^-1 A`` and the factor's inverse
    ``L^-1 = σ^-1 R_G^-T R_M^-1 A``, the precision factor, are as cheap.
    Each is a few banded solves and sparse products in C++, and nothing is
    formed.

    ``σ`` is set so that the pointwise standard deviation at the middle of
    the interval is the one asked for; the natural ends raise it within a
    length scale of either end.

    Args:
        space: the space.
        length_scale: ``ℓ``, the prior's correlation length.
        pointwise_std: the field's standard deviation at the middle of the
            interval.

    Returns:
        The measure, with its covariance, its factor, its precision and its
        precision factor.

    Raises:
        ValueError: for a non-positive length or standard deviation.
    """
    if length_scale <= 0.0 or pointwise_std <= 0.0:
        raise ValueError("The length scale and the standard deviation are positive.")
    helmholtz = _matrix(space.n_elements, 1.0, float(length_scale) ** 2)
    solver = helmholtz.cholesky()
    mass, gram = space.mass, space.gram
    mass_factor, gram_factor = space.mass_cholesky, space.gram_cholesky
    # The variance at the middle for σ = 1: c^T A^-1 M A^-1 c with c the
    # nodal weights of the value there.
    middle = space.interpolate(_evaluation_weights(space, 0.5))
    spread = solver.solve(middle)
    sigma = float(pointwise_std) / np.sqrt(spread.dot_weighted(spread, mass))
    calls = space.calls

    def covariance(x: Any) -> Any:
        calls["prior.covariance"] += 1
        out = solver.solve(mass.apply(solver.solve(gram.apply(x))))
        out.scale(sigma**2)
        return out

    def factor(w: Any) -> Any:
        calls["prior.factor"] += 1
        out = solver.solve(
            mass_factor.apply_factor(gram_factor.apply_factor_transpose(w))
        )
        out.scale(sigma)
        return out

    def factor_adjoint(x: Any) -> Any:
        calls["prior.factor.adjoint"] += 1
        out = gram_factor.solve_factor_transpose(
            mass_factor.apply_factor_transpose(solver.solve(gram.apply(x)))
        )
        out.scale(sigma)
        return out

    def precision(x: Any) -> Any:
        calls["prior.precision"] += 1
        out = gram_factor.solve(helmholtz.apply(mass_factor.solve(helmholtz.apply(x))))
        out.scale(sigma**-2)
        return out

    def whiten(x: Any) -> Any:
        calls["prior.whiten"] += 1
        out = gram_factor.solve_factor_transpose(
            mass_factor.solve_factor(helmholtz.apply(x))
        )
        out.scale(1.0 / sigma)
        return out

    def whiten_adjoint(w: Any) -> Any:
        calls["prior.whiten.adjoint"] += 1
        out = gram_factor.solve(
            helmholtz.apply(
                mass_factor.solve_factor_transpose(
                    gram_factor.apply_factor_transpose(w)
                )
            )
        )
        out.scale(1.0 / sigma)
        return out

    return GaussianMeasure(
        space,
        covariance=LinearOperator.self_adjoint(space, covariance, traits=_DEFINITE),
        covariance_factor=LinearOperator.from_callables(
            space, space, factor, adjoint=factor_adjoint
        ),
        precision=LinearOperator.self_adjoint(space, precision, traits=_DEFINITE),
        precision_factor=LinearOperator.from_callables(
            space, space, whiten, adjoint=whiten_adjoint
        ),
    )
