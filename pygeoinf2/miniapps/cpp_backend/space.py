"""A coordinate-free Hilbert space over C++ vectors.

The vectors are ``cppvec.Vector`` objects: a ``std::vector<double>`` behind a
pybind11 class that exposes no buffer, no indexing and no arithmetic
operators. The space is the piecewise-linear finite element space on a
uniform mesh of ``[0, 1]``, with the mass matrix as its inner product, and it
is a plain :class:`~pygeoinf2.algebra.spaces.HilbertSpace`: no component map,
no Gram matrix the library can read, nothing NumPy can touch. Every vector
calculation the library asks for is answered by the C++ side, and the space
counts what it was asked, so a run can say exactly which members the
library used.

The mesh and its mass and stiffness matrices are assembled here from element
matrices, as lists of triplets handed to ``cppvec.SparseMatrix``; the mass
matrix is factorized once by the C++ banded Cholesky. The operators built on
top -- point evaluation, local averages, the Laplacian and a Matérn-type
prior -- are written from callables with their adjoints taken through the
mass solve, which is the Riesz map of this space.
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
    "laplacian_operator",
    "matern_measure",
]

_DEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
_SEMIDEFINITE = Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE


def _triplets(n_elements: int, local: tuple[tuple[float, ...], ...], factor: float):
    """Assemble a 2x2 element matrix times *factor* over the mesh, as triplets."""
    rows, cols, values = [], [], []
    for e in range(n_elements):
        for a in range(2):
            for b in range(2):
                rows.append(e + a)
                cols.append(e + b)
                values.append(factor * local[a][b])
    return rows, cols, values


class CppSpace(HilbertSpace):
    """Piecewise-linear finite elements on ``[0, 1]`` over ``cppvec.Vector``.

    The inner product is ``(x, y) = x^T M y`` with ``M`` the mass matrix,
    computed by the vector's own weighted dot product; the mass solve and the
    white-noise draw go through the C++ Cholesky factor. The space is not a
    ``CoordinateSpace``: nothing here maps a vector to an array.

    Every call of a member is tallied in :attr:`calls`, by name, so that a
    run can report which operations the library asked of the backend.
    """

    def __init__(self, n_elements: int) -> None:
        """
        Args:
            n_elements: the number of elements; the space has one more node.

        Raises:
            ValueError: for fewer than one element.
        """
        if n_elements < 1:
            raise ValueError("A mesh needs at least one element.")
        self._n_elements = int(n_elements)
        self._n_nodes = self._n_elements + 1
        h = 1.0 / self._n_elements
        self._mass = cppvec.SparseMatrix(
            self._n_nodes,
            self._n_nodes,
            *_triplets(self._n_elements, ((2.0, 1.0), (1.0, 2.0)), h / 6.0),
        )
        self._stiffness = cppvec.SparseMatrix(
            self._n_nodes,
            self._n_nodes,
            *_triplets(self._n_elements, ((1.0, -1.0), (-1.0, 1.0)), 1.0 / h),
        )
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
    def is_function_space(self) -> bool:
        """True: the space discretizes functions on ``[0, 1]``."""
        return True

    def _key(self) -> Hashable:
        self.calls["_key"] += 1
        return ("CppSpace", self._n_elements)

    def __repr__(self) -> str:
        return f"CppSpace(elements={self._n_elements}, vectors=cppvec.Vector)"

    @property
    def mass(self) -> Any:
        """The mass matrix, a ``cppvec.SparseMatrix``."""
        return self._mass

    @property
    def stiffness(self) -> Any:
        """The stiffness matrix of ``-d²/dx²`` with natural ends."""
        return self._stiffness

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
        """``x^T M y``, with ``M`` applied by the sparse matrix."""
        self.calls["inner_product"] += 1
        return x.dot_weighted(y, self._mass)

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
        vector = cppvec.Vector(self._n_nodes)
        vector.fill_random(int(resolve_rng(rng).integers(2**63)))
        return vector

    def white_noise(self, *, rng: Generator | None = None) -> Any:
        """White noise with respect to the mass inner product.

        With ``M = L L^T`` and ``z`` standard normal, ``L^-T z`` has
        covariance ``M^-1`` on the nodal values, which is the identity on the
        space. Applying ``L`` instead would give covariance ``M``: the wrong
        answer by two factors of the metric, which is what
        ``check_white_noise`` is for.
        """
        self.calls["white_noise"] += 1
        return self._mass_cholesky.solve_factor_transpose(self.random(rng=rng))

    # ----------------------------------------------------------------- #
    #                          The Riesz map                            #
    # ----------------------------------------------------------------- #

    def mass_solve(self, b: Any) -> Any:
        """``M^-1 b``: the representer of the functional ``x -> b^T x``."""
        self.calls["mass_solve"] += 1
        return self._mass_cholesky.solve(b)

    def interpolate(self, values: Sequence[float]) -> Any:
        """A vector from nodal values, for building functionals and plotting."""
        if len(values) != self._n_nodes:
            raise ValueError(
                f"Expected {self._n_nodes} nodal values, got {len(values)}."
            )
        return cppvec.Vector.from_list([float(v) for v in values])


# --------------------------------------------------------------------- #
#                         Functionals and operators                     #
# --------------------------------------------------------------------- #


def _functional_operator(
    space: CppSpace, coefficients: list[Any], /, *, name: str
) -> LinearOperator:
    """The operator ``x -> (c_j^T x)_j`` into Euclidean space, and its adjoint.

    Each coefficient vector ``c_j`` defines the functional ``x -> c_j^T x``
    on nodal values. Its representer in the mass inner product is
    ``M^-1 c_j``, so the adjoint of the operator is
    ``y -> M^-1 (sum_j y_j c_j)``: one mass solve however many functionals.
    """
    data_space = EuclideanSpace(len(coefficients))

    def value(x: Any):
        space.calls[f"{name}.value"] += 1
        return data_space.from_components(np.array([c.dot(x) for c in coefficients]))

    def adjoint(y: Any):
        space.calls[f"{name}.adjoint"] += 1
        load = space.zero()
        for weight, c in zip(data_space.to_components(y), coefficients):
            load.axpy(float(weight), c)
        return space.mass_solve(load)

    return LinearOperator.from_callables(space, data_space, value, adjoint=adjoint)


def point_evaluation_operator(
    space: CppSpace, points: Sequence[float], /
) -> LinearOperator:
    """Evaluation at points of ``[0, 1]``, into ``EuclideanSpace(len(points))``.

    A piecewise-linear function is evaluated by interpolating between the two
    nodes of the element containing the point, so each functional has two
    nonzero coefficients.

    Args:
        space: the space.
        points: positions in ``[0, 1]``.

    Raises:
        ValueError: for a point outside the interval.
    """
    coefficients = []
    for point in points:
        if not 0.0 <= point <= 1.0:
            raise ValueError(f"The point {point} is outside [0, 1].")
        element = min(int(point * space.n_elements), space.n_elements - 1)
        local = point * space.n_elements - element
        weights = [0.0] * (space.n_elements + 1)
        weights[element] = 1.0 - local
        weights[element + 1] = local
        coefficients.append(space.interpolate(weights))
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
        for e, sign in ((node - 1, 1.0), (node, -1.0)):
            if not 0 <= e < n:
                continue
            lo, hi = max(a, e * h), min(b, (e + 1) * h)
            if hi <= lo:
                continue
            # On element e, φ_node is linear: (x - x_e)/h rising (sign > 0)
            # or 1 - (x - x_e)/h falling (sign < 0).
            x_e = e * h

            def primitive(x: float) -> float:
                s = (x - x_e) / h
                return h * (s * s / 2.0 if sign > 0 else s - s * s / 2.0)

            total += primitive(hi) - primitive(lo)
        return total

    coefficients = []
    for a, b in intervals:
        if not 0.0 <= a < b <= 1.0:
            raise ValueError(f"The interval ({a}, {b}) is not inside [0, 1].")
        weights = [hat_integral(i, a, b) / (b - a) for i in range(n + 1)]
        coefficients.append(space.interpolate(weights))
    return _functional_operator(space, coefficients, name="local_average")


def laplacian_operator(space: CppSpace, /) -> LinearOperator:
    """``-d²/dx²`` with natural ends, as the operator ``M^-1 K`` on the space.

    Its Galerkin matrix is the stiffness matrix ``K``, symmetric, so the
    operator is self-adjoint in the mass inner product, and positive
    semidefinite because constants are in its kernel.
    """

    def value(x: Any):
        space.calls["laplacian.value"] += 1
        return space.mass_solve(space.stiffness.apply(x))

    return LinearOperator.self_adjoint(space, value, traits=_SEMIDEFINITE)


def matern_measure(
    space: CppSpace, /, *, length_scale: float, amplitude: float = 1.0
) -> GaussianMeasure:
    """A Gaussian measure with covariance ``σ² (I + ℓ² Δ)^-2``.

    With ``A = M + ℓ² K`` the Helmholtz matrix, ``(I + ℓ² Δ)^-1 = A^-1 M`` on
    the space; it is self-adjoint there, so it serves as the covariance factor
    (scaled by ``σ``) and its square is the covariance. Both are applied by
    two C++ solves with the Helmholtz Cholesky factor and two mass
    multiplications, and nothing is formed.

    Args:
        space: the space.
        length_scale: ``ℓ``, the correlation length.
        amplitude: ``σ``, the scale of the field.

    Returns:
        The measure, with a factor so it can be sampled.
    """
    ell2 = float(length_scale) ** 2
    n = space.n_elements + 1
    h = 1.0 / space.n_elements
    rows, cols, values = _triplets(space.n_elements, ((2.0, 1.0), (1.0, 2.0)), h / 6.0)
    k_rows, k_cols, k_values = _triplets(
        space.n_elements, ((1.0, -1.0), (-1.0, 1.0)), ell2 / h
    )
    helmholtz = cppvec.SparseMatrix(
        n, n, rows + k_rows, cols + k_cols, values + k_values
    )
    factor_solver = helmholtz.cholesky()
    sigma = float(amplitude)

    def smooth(x: Any):
        space.calls["matern.smooth"] += 1
        out = factor_solver.solve(space.mass.apply(x))
        out.scale(sigma)
        return out

    def covariance(x: Any):
        return smooth(smooth(x))

    return GaussianMeasure(
        space,
        covariance=LinearOperator.self_adjoint(space, covariance, traits=_DEFINITE),
        covariance_factor=LinearOperator.self_adjoint(space, smooth, traits=_DEFINITE),
    )
