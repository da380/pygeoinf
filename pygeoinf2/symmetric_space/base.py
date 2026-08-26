"""
Spaces whose basis diagonalises the Laplacian.

A symmetric space is a coordinate space with a distinguished spectral basis:
one in which the Laplace-Beltrami operator is diagonal. The space is symmetric;
the operators that respect its structure are *invariant*, which is why
:meth:`SymmetricSpace.invariant_operator` keeps that word. Everything that makes
these spaces pleasant to compute with follows from that single fact, and most
of it is already general machinery elsewhere in the package:

- an operator that is a function of the Laplacian is a
  :class:`~pygeoinf2.algebra.diagonal.DiagonalLinearOperator`, so it is closed
  under the algebra and carries an exact functional calculus;
- a Gaussian measure whose covariance is such an operator has an exact square
  root, so it can be sampled directly;
- the correction v1 writes by hand as ``sqrt(spectral_variances /
  metric_values)`` is simply ``white_noise`` on a space with a diagonal metric,
  and needs no code here at all.

What each concrete space must supply is small: the map between fields and
spectral components, the Laplacian eigenvalue attached to each component, and
the value of each basis function at a point.

See DESIGN.md section 13.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.diagonal import DiagonalLinearOperator
from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import ArrayVectorMixin, DiagonalMetricSpace
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits

__all__ = ["SymmetricSpace", "lift_formal_adjoint"]


class SymmetricSpace(ArrayVectorMixin, DiagonalMetricSpace[np.ndarray]):
    """A coordinate space whose basis diagonalises the Laplacian.

    Subclasses supply :meth:`to_components`, :meth:`from_components`,
    :attr:`laplacian_eigenvalues`, :meth:`basis_at` and ``_key``. Everything
    below is then available, and none of it needs re-implementing per space.
    """

    def __init__(self, metric_values: np.ndarray, /) -> None:
        """
        Args:
            metric_values: the Gram diagonal in the spectral basis. All ones
                for a Lebesgue space with an orthonormal basis; the Sobolev
                symbol for a Sobolev space over the same coordinate map.
        """
        super().__init__(metric_values)

    # ----------------------------------------------------------------- #
    #                         Subclass interface                        #
    # ----------------------------------------------------------------- #

    @property
    @abstractmethod
    def laplacian_eigenvalues(self) -> np.ndarray:
        """The Laplacian eigenvalue attached to each spectral component.

        Non-negative, and zero exactly on the constant mode.
        """

    @abstractmethod
    def basis_at(self, point: Any, /) -> np.ndarray:
        """The value of each basis function at a point.

        These are the derivative components of the evaluation functional, and
        so exactly what :meth:`dirac` needs. Returning them rather than a
        representer is deliberate: it is what an evaluation actually produces,
        and the metric is applied once, in the adjoint. See DESIGN.md 5.6.
        """

    # ----------------------------------------------------------------- #
    #                         Invariant operators                       #
    # ----------------------------------------------------------------- #

    @property
    def laplacian(self) -> DiagonalLinearOperator:
        """The Laplace-Beltrami operator, diagonal by construction."""
        return DiagonalLinearOperator(self, self.laplacian_eigenvalues)

    def invariant_operator(
        self,
        symbol: Callable[[np.ndarray], np.ndarray],
        /,
        *,
        traits: Traits = Traits.NONE,
    ) -> DiagonalLinearOperator:
        """The operator ``f(Laplacian)``, from a function of the eigenvalues.

        Args:
            symbol: applied to the array of Laplacian eigenvalues.
            traits: extra claims. Symmetry and definiteness are deduced from
                the resulting values, so they seldom need supplying.
        """
        values = np.asarray(symbol(self.laplacian_eigenvalues), dtype=float)
        if values.shape != (self.dim,):
            raise ValueError(
                f"The symbol returned shape {values.shape}, expected " f"{(self.dim,)}."
            )
        return DiagonalLinearOperator(self, values, traits=traits)

    def sobolev_symbol(self, order: float, scale: float, /) -> np.ndarray:
        """``(1 + scale^2 lambda)^order``, the Sobolev weight on each mode.

        The scale is a length: it sets where the weight turns over, and so what
        "smooth" means. Without it the order alone would be scale-dependent,
        which is how a Sobolev prior comes to behave differently on a sphere of
        a different radius.
        """
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        return (1.0 + scale**2 * self.laplacian_eigenvalues) ** order

    def heat_symbol(self, time: float, /) -> np.ndarray:
        """``exp(-time * lambda)``, the heat kernel's spectral weight."""
        if time < 0.0:
            raise ValueError("time must be non-negative.")
        return np.exp(-time * self.laplacian_eigenvalues)

    # ----------------------------------------------------------------- #
    #                         Invariant measures                        #
    # ----------------------------------------------------------------- #

    @property
    def reference_point(self) -> Any:
        """Any point of the domain. The space is homogeneous, so any will do.

        Used where a quantity is provably the same everywhere and one place has
        to be picked to evaluate it — the pointwise variance of an invariant
        measure being the case that matters.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not name a reference point."
        )

    def pointwise_variance(
        self,
        spectral_variances: np.ndarray | Callable[[np.ndarray], np.ndarray],
        /,
    ) -> float:
        """The variance of ``x(p)`` under the corresponding invariant measure.

        This is the number a modeller actually has an opinion about: nobody
        knows what spectral amplitude they want, and everybody knows roughly
        how big the field should be.

        The value is ``(C u, u)`` with ``u`` the representer of evaluation at
        ``p``, which comes out as ``sum_k s_k phi_k(p)^2 / g_k``. The metric
        appears because the spectral variances are the covariance *operator's*
        eigenvalues, while a sample's components carry the ``1/sqrt(g)`` of
        white noise. Dropping it is the error of DESIGN.md section 5.6 once
        more, and it is invisible on a Lebesgue space where ``g == 1``.
        """
        variances = self._resolve_variances(spectral_variances)
        basis = self.basis_at(self.reference_point)
        return float(np.sum(variances * basis**2 / self.metric_values))

    def _resolve_variances(
        self,
        spectral_variances: np.ndarray | Callable[[np.ndarray], np.ndarray],
        /,
    ) -> np.ndarray:
        """Validate a spectrum given either directly or as a symbol."""
        if callable(spectral_variances):
            variances = np.asarray(
                spectral_variances(self.laplacian_eigenvalues), dtype=float
            )
        else:
            variances = np.asarray(spectral_variances, dtype=float)
        if variances.shape != (self.dim,):
            raise ValueError(
                f"Got {variances.shape} variances for dimension {self.dim}."
            )
        if np.any(variances < 0.0):
            raise ValueError("Spectral variances must be non-negative.")
        return variances

    def invariant_measure(
        self,
        spectral_variances: np.ndarray | Callable[[np.ndarray], np.ndarray],
        /,
        *,
        expectation: np.ndarray | None = None,
        pointwise_std: float | None = None,
    ) -> GaussianMeasure:
        """A Gaussian whose covariance is diagonal in the spectral basis.

        The factor is the exact square root of the covariance, so sampling is a
        single spectral multiply of white noise — and the white noise carries
        the ``1/sqrt(g)`` that a non-trivial metric demands, rather than that
        correction being written out here.

        ``pointwise_std`` rescales the whole spectrum so that
        :meth:`pointwise_variance` comes out as its square, leaving the shape
        of the spectrum — and so the correlation length — untouched.
        """
        variances = self._resolve_variances(spectral_variances)
        if pointwise_std is not None:
            if pointwise_std <= 0.0:
                raise ValueError("pointwise_std must be positive.")
            current = self.pointwise_variance(variances)
            if current <= 0.0:
                raise ValueError(
                    "A measure with zero pointwise variance cannot be scaled "
                    "to a given standard deviation."
                )

            variances = variances * (pointwise_std**2 / current)

        factor = DiagonalLinearOperator(self, np.sqrt(variances))
        # A precision exists only when every variance is strictly positive; a
        # measure supported on a subspace has none, and should say so rather
        # than carry a regularised stand-in.
        precision = (
            DiagonalLinearOperator(self, 1.0 / variances)
            if np.all(variances > 0.0)
            else None
        )
        return GaussianMeasure(
            self,
            expectation=expectation,
            covariance_factor=factor,
            precision=precision,
        )

    def sobolev_measure(
        self,
        order: float,
        scale: float,
        /,
        *,
        amplitude: float = 1.0,
        expectation: np.ndarray | None = None,
        pointwise_std: float | None = None,
    ) -> GaussianMeasure:
        """A Gaussian prior whose covariance is a Sobolev symbol.

        The workhorse prior: smoothness set by ``order``, correlation length by
        ``scale``. Note the symbol enters with a *negative* order, since a
        covariance must decay with the eigenvalue for the measure to be
        supported on functions of that smoothness.
        """
        return self.invariant_measure(
            amplitude**2 * self.sobolev_symbol(-order, scale),
            expectation=expectation,
            pointwise_std=pointwise_std,
        )

    def heat_measure(
        self,
        time: float,
        /,
        *,
        amplitude: float = 1.0,
        expectation: np.ndarray | None = None,
        pointwise_std: float | None = None,
    ) -> GaussianMeasure:
        """A Gaussian prior with a heat-kernel covariance."""
        return self.invariant_measure(
            amplitude**2 * self.heat_symbol(time),
            expectation=expectation,
            pointwise_std=pointwise_std,
        )

    # ----------------------------------------------------------------- #
    #                        Point evaluation                           #
    # ----------------------------------------------------------------- #

    def dirac(self, point: Any, /) -> LinearFunctional:
        """The evaluation functional at a point.

        Built from derivative components, so ``dirac(p).representer`` is the
        Riesz representer and the metric is applied exactly once, inside the
        adjoint. On a Lebesgue space the Dirac is not bounded and the
        representer is meaningless; on a Sobolev space of high enough order it
        is a perfectly good function, and the difference is visible in the
        metric rather than hidden in the code.
        """
        return LinearFunctional.from_derivative_components(self, self.basis_at(point))

    def evaluate(self, x: np.ndarray, points: Sequence[Any], /) -> np.ndarray:
        """The field's values at several points.

        The generic route sums the basis at each point, which costs one
        ``basis_at`` call per point. A space whose transform can evaluate at
        scattered points directly should override this; the sphere does.
        """
        components = self.to_components(x)
        return np.array(
            [float(np.dot(self.basis_at(point), components)) for point in points]
        )

    def accumulate(self, weights: np.ndarray, points: Sequence[Any], /) -> np.ndarray:
        """The derivative components of ``x -> sum_i y_i x(r_i)``.

        The adjoint of :meth:`evaluate`, and the two must stay in step. The
        generic route sums the basis at each point; a space whose transform has
        a scattered-point adjoint should override this, as the periodic box
        does with a type-1 NUFFT.
        """
        total = np.zeros(self.dim)
        for weight, point in zip(np.asarray(weights, dtype=float), points):
            if weight != 0.0:
                total += weight * self.basis_at(point)
        return total

    def point_evaluation_operator(self, points: Sequence[Any], /) -> LinearOperator:
        """Evaluation at several points, as an operator into a Euclidean space.

        Matrix-free: nothing of size ``len(points) x dim`` is formed, so this
        is usable at the scale real acquisition geometries reach. Call
        :meth:`~pygeoinf2.algebra.operators.LinearOperator.assembled` on the
        result when the matrix is small enough to be worth storing.

        Its adjoint returns a weighted sum of Dirac *representers*, which is
        what makes an adjoint-based inversion give a function rather than an
        array of numbers.
        """
        from ..algebra.spaces import EuclideanSpace

        points = tuple(points)
        if not points:
            raise ValueError("At least one point is needed.")

        return LinearOperator.from_derivative_callables(
            self,
            EuclideanSpace(len(points)),
            lambda x: self.evaluate(x, points),
            lambda y: self.accumulate(y, points),
        )

    # ----------------------------------------------------------------- #
    #                              Geodesics                            #
    # ----------------------------------------------------------------- #

    def geodesic_distance(self, start: Any, end: Any, /) -> float:
        """The distance between two points along a shortest path."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement geodesic_distance."
        )

    def geodesic_quadrature(
        self, start: Any, end: Any, /, *, count: int
    ) -> tuple[list[Any], np.ndarray]:
        """Nodes and weights integrating along the geodesic between two points.

        The weights carry the arc-length element, so they sum to the distance
        between the endpoints and integrating the constant one gives that
        distance.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement geodesic_quadrature."
        )

    def geodesic_ball_quadrature(
        self, centre: Any, radius: float, /, *, count: int
    ) -> tuple[list[Any], np.ndarray]:
        """Nodes and weights integrating over a geodesic ball.

        The weights carry the area element, so they sum to the ball's measure.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement geodesic_ball_quadrature."
        )

    # ----------------------------------------------------------------- #
    #                         Averaging operators                       #
    # ----------------------------------------------------------------- #

    def path_average_operator(
        self,
        paths: Sequence[tuple[Any, Any]],
        /,
        *,
        count: int = 20,
        normalise: bool = True,
    ) -> LinearOperator:
        """Averages, or integrals, along a set of geodesic paths.

        The tomographic forward map. Built as ``W E``: point evaluation at the
        pooled quadrature nodes, then a sparse matrix of weights. Writing it
        this way means the adjoint is derived by the algebra rather than
        written out, which is where the metric is usually dropped.

        Args:
            paths: ``(start, end)`` pairs.
            count: quadrature nodes per path.
            normalise: divide by the path length, giving an average rather
                than an integral.
        """
        paths = tuple(paths)
        if not paths:
            raise ValueError("At least one path is needed.")

        nodes: list[Any] = []
        rows, columns, values = [], [], []
        for index, (start, end) in enumerate(paths):
            path_nodes, path_weights = self.geodesic_quadrature(start, end, count=count)
            if normalise:
                total = float(np.sum(path_weights))
                if total <= 0.0:
                    raise ValueError(
                        f"Path {index} has zero length, so it has no average."
                    )
                path_weights = path_weights / total
            offset = len(nodes)
            nodes.extend(path_nodes)
            rows.extend([index] * len(path_nodes))
            columns.extend(range(offset, offset + len(path_nodes)))
            values.extend(np.asarray(path_weights, dtype=float).tolist())

        return _weight_operator(
            len(paths), len(nodes), rows, columns, values
        ) @ self.point_evaluation_operator(nodes)

    def geodesic_ball_average_operator(
        self,
        centres: Sequence[Any],
        radius: float,
        /,
        *,
        count: int = 100,
        normalise: bool = True,
    ) -> LinearOperator:
        """Averages, or integrals, over geodesic balls of a common radius.

        The property operator of an inference problem: a handful of local
        averages, which is what a set of data can actually constrain. Same
        ``W E`` construction as :meth:`path_average_operator`.
        """
        centres = tuple(centres)
        if not centres:
            raise ValueError("At least one centre is needed.")

        nodes: list[Any] = []
        rows, columns, values = [], [], []
        for index, centre in enumerate(centres):
            ball_nodes, ball_weights = self.geodesic_ball_quadrature(
                centre, radius, count=count
            )
            if normalise:
                total = float(np.sum(ball_weights))
                if total <= 0.0:
                    raise ValueError(f"Ball {index} has zero measure.")
                ball_weights = ball_weights / total
            offset = len(nodes)
            nodes.extend(ball_nodes)
            rows.extend([index] * len(ball_nodes))
            columns.extend(range(offset, offset + len(ball_nodes)))
            values.extend(np.asarray(ball_weights, dtype=float).tolist())

        return _weight_operator(
            len(centres), len(nodes), rows, columns, values
        ) @ self.point_evaluation_operator(nodes)

    def project_function(self, function: Callable[[Any], float], /) -> np.ndarray:
        """The field obtained by sampling a function on the space's grid."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement project_function."
        )

    def random_point(self, *, rng: Generator | None = None) -> Any:
        """A point drawn uniformly from the domain."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement random_point."
        )


def _distribute(total: int, weights: np.ndarray) -> np.ndarray:
    """Split a budget of points across rings in proportion to their weights.

    Every ring gets at least one point, and the total is exact, so a rule built
    from this integrates the constant one to the right answer regardless of how
    the rounding falls.
    """
    if total <= 0:
        raise ValueError("The point budget must be positive.")
    rings = np.asarray(weights, dtype=float).size
    counts = np.ones(rings, dtype=int)
    remaining = total - rings
    if remaining <= 0:
        return counts

    positive = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if not np.any(positive > 0.0):
        counts[:remaining] += 1
        return counts

    scaled = remaining * positive / positive.sum()
    increments = np.floor(scaled).astype(int)
    counts += increments
    leftover = remaining - int(increments.sum())
    if leftover > 0:
        order = np.argsort(-(scaled - increments))
        counts[order[:leftover]] += 1
    return counts


def _weight_operator(
    rows: int,
    columns: int,
    row_indices: Sequence[int],
    column_indices: Sequence[int],
    values: Sequence[float],
) -> LinearOperator:
    """A sparse matrix between Euclidean spaces, as a linear operator.

    Both spaces are orthonormal, so the adjoint *is* the transpose and there is
    no metric to get wrong. That is the only place in this module where that is
    true, and it is why the ``W E`` factorisation is worth having: all the
    metric lives in ``E``, which is built from derivative components.
    """
    from scipy.sparse import coo_matrix

    from ..algebra.spaces import EuclideanSpace

    matrix = coo_matrix(
        (np.asarray(values, dtype=float), (row_indices, column_indices)),
        shape=(rows, columns),
    ).tocsr()
    transpose = matrix.T.tocsr()
    return LinearOperator.from_callables(
        EuclideanSpace(columns),
        EuclideanSpace(rows),
        lambda v: matrix @ v,
        adjoint=lambda y: transpose @ y,
    )


def lift_formal_adjoint(
    operator: LinearOperator,
    domain: SymmetricSpace,
    /,
    *,
    codomain: SymmetricSpace | None = None,
    traits: Traits = Traits.NONE,
) -> LinearOperator:
    """Reuse an operator defined on one metric under another.

    The idiom this exists for: derive an operator's action and its adjoint on
    the Lebesgue space, where both are easy, then use it on a Sobolev space
    over the same coordinate map. With ``(x, y)_V == c_x^T G_V c_y`` and
    likewise for ``U``,

        ``A*_V == (G_VX^-1 G_UX) . A*_U . (G_UY^-1 G_VY)``

    which is the mass-weighted lift of DESIGN.md 3.5 with the mass operator
    read off the two diagonals. Cheap, because both metrics are diagonal.

    No self-adjointness is claimed: a formally self-adjoint operator is
    self-adjoint under the new metric only if it commutes with the ratio of the
    two, which for a general operator it does not. Claim it explicitly through
    ``traits`` when it holds, and verify with ``testing.check_traits``.

    Args:
        operator: the operator on the base spaces, with a working adjoint.
        domain: the space to present the operator's domain as.
        codomain: likewise for the codomain; defaults to ``domain``.
        traits: claims about the lifted operator.
    """
    codomain = domain if codomain is None else codomain
    base_domain, base_codomain = operator.domain, operator.codomain

    if base_domain.dim != domain.dim or base_codomain.dim != codomain.dim:
        raise ValueError(
            f"The operator maps dimension {base_domain.dim} to "
            f"{base_codomain.dim}, but the target spaces are {domain.dim} and "
            f"{codomain.dim}."
        )

    domain_ratio = base_domain.metric_values / domain.metric_values
    codomain_ratio = codomain.metric_values / base_codomain.metric_values

    def value(x: np.ndarray) -> np.ndarray:
        return codomain.from_components(
            base_codomain.to_components(
                operator(base_domain.from_components(domain.to_components(x)))
            )
        )

    def adjoint(y: np.ndarray) -> np.ndarray:
        components = codomain_ratio * codomain.to_components(y)
        pulled = base_domain.to_components(
            operator.adjoint(base_codomain.from_components(components))
        )
        return domain.from_components(domain_ratio * pulled)

    return LinearOperator.from_callables(
        domain, codomain, value, adjoint=adjoint, traits=traits
    )
