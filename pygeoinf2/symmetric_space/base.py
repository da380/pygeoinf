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
from typing import TYPE_CHECKING, Any, Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.diagonal import DiagonalLinearOperator
from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import (
    ArrayVectorMixin,
    DiagonalMetricSpace,
    HilbertModule,
    HilbertSpace,
)
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits

if TYPE_CHECKING:  # pragma: no cover
    from ..numerics.solvers import IterativeSolver

__all__ = ["SymmetricSpace", "lift_formal_adjoint"]


class SymmetricSpace(
    ArrayVectorMixin, HilbertModule[np.ndarray], DiagonalMetricSpace[np.ndarray]
):
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
    def order(self) -> float:
        """The Sobolev order. Zero means the inner product is the ``L2`` one."""

    @abstractmethod
    def with_order(self, order: float, /) -> "SymmetricSpace":
        """The same coordinate map, viewed with a different Sobolev order."""

    def _coordinate_key(self) -> Hashable:
        """Identifies the component map alone, with the metric left out.

        ``_key`` identifies the *space*, so it carries the Sobolev order and
        length scale; two spaces that differ only there hold the same fields
        and are not the same space. This is the other half of that key, and
        subclasses that can separate the two override it. The default is the
        whole key, which is safe and says "only a space itself".
        """
        return self._key()

    def shares_vectors_with(self, other: HilbertSpace, /) -> bool:
        """True for another symmetric space over the same coordinate map.

        Two spaces differing only in Sobolev order hold the *same* fields —
        :meth:`with_order` says so — and differ only in how they measure them.
        Saying it lets a formal-adjoint lift move a vector between them for
        nothing instead of round-tripping through components, which on a
        spectral space is two transforms each way.
        """
        if self is other:
            return True
        if not isinstance(other, SymmetricSpace):
            return False
        return self._coordinate_key() == other._coordinate_key()

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

    def spectral_operator(
        self, values: np.ndarray, /, *, traits: Traits = Traits.NONE
    ) -> DiagonalLinearOperator:
        """A diagonal operator from an explicit value per component.

        More general than :meth:`invariant_operator`, whose symbol can only
        depend on the Laplacian eigenvalue. An operator built here need not be
        invariant — a band-limiting projection is the obvious case — so nothing
        is claimed about it beyond what the values themselves imply.

        v1 spells this ``InvariantLinearAutomorphism.from_index_function``.
        Taking an array rather than a callable on indices is what lets the
        caller write ``space.spectral_operator(f(space.degrees))``.
        """
        array = np.asarray(values, dtype=float)
        if array.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim} values, got {array.shape}.")
        return DiagonalLinearOperator(self, array, traits=traits)

    @property
    def degrees(self) -> np.ndarray:
        """The index attached to each component, for a degree-wise symbol.

        On a sphere this is the harmonic degree; on a box, the wavenumber
        magnitude rounded down. What it is for is writing a symbol that depends
        on where a component sits rather than on its eigenvalue.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not name a degree per component."
        )

    def degree_multiplicity(self, degree: int, /) -> int:
        """How many components share a degree."""
        return int(np.count_nonzero(self.degrees == degree))

    def spectral_projection_operator(
        self, /, *, lmin: int = 0, lmax: int | None = None
    ) -> DiagonalLinearOperator:
        """The projection onto a band of degrees, *within* this space.

        The companion of a coefficient operator, which maps *out* of the space
        into a Euclidean one. This stays put: it zeroes everything outside the
        band and leaves the rest alone, so it is an idempotent self-adjoint
        projection.
        """
        degrees = self.degrees
        top = degrees.max() if lmax is None else lmax
        keep = ((degrees >= lmin) & (degrees <= top)).astype(float)
        return self.spectral_operator(
            keep, traits=Traits.SELF_ADJOINT | Traits.IDEMPOTENT
        )

    def order_inclusion_operator(self, target: "SymmetricSpace", /) -> LinearOperator:
        """The identity, read from this space into one of a different order.

        The same function, viewed in a different metric. It is *not* the
        identity operator: its adjoint carries the ratio of the two metrics,
        which is exactly what makes ``H^s -> H^t`` a bounded inclusion rather
        than a relabelling.
        """
        if target.dim != self.dim:
            raise ValueError(
                f"An inclusion needs matching dimensions; {self.dim} against "
                f"{target.dim}."
            )
        return lift_formal_adjoint(LinearOperator.identity(self), self, codomain=target)

    def l2_products_operator(self, fields: Sequence[np.ndarray], /) -> LinearOperator:
        """Inner products against a set of fields, in the ``L2`` metric.

        ``x -> [(f_i, x)_L2]``. The ``L2`` products specifically, not this
        space's: the rows are the fields' own components, so the operator means
        the same thing whatever Sobolev order it is read on.
        """
        from ..algebra.spaces import EuclideanSpace

        fields = tuple(fields)
        if not fields:
            raise ValueError("At least one field is needed.")
        base = self if self.order == 0.0 else self.with_order(0.0)
        rows = np.stack([base.to_components(field) for field in fields])
        return LinearOperator.from_matrix(
            self, EuclideanSpace(len(fields)), rows, form="galerkin"
        )

    def estimate_truncation_degree(
        self, symbol: Callable[[np.ndarray], np.ndarray], /, *, tolerance: float = 1e-3
    ) -> int:
        """The smallest degree holding all but ``tolerance`` of a spectrum's power.

        For choosing a truncation from a prior rather than by habit: pass the
        spectral variances and get the degree beyond which the field has
        negligible energy.
        """
        if not 0.0 < tolerance < 1.0:
            raise ValueError(f"The tolerance lies in (0, 1), got {tolerance}.")
        power = np.asarray(symbol(self.laplacian_eigenvalues), dtype=float)
        degrees = self.degrees
        order = np.argsort(degrees)
        cumulative = np.cumsum(power[order])
        total = cumulative[-1]
        if total <= 0.0:
            return 0
        reached = np.searchsorted(cumulative, (1.0 - tolerance) * total)
        return int(degrees[order][min(reached, degrees.size - 1)])

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
        norm_std: float | None = None,
    ) -> GaussianMeasure:
        """A Gaussian whose covariance is diagonal in the spectral basis.

        The factor is the exact square root of the covariance, so sampling is a
        single spectral multiply of white noise — and the white noise carries
        the ``1/sqrt(g)`` that a non-trivial metric demands, rather than that
        correction being written out here.

        ``pointwise_std`` rescales the whole spectrum so that
        :meth:`pointwise_variance` comes out as its square, leaving the shape
        of the spectrum — and so the correlation length — untouched.
        ``norm_std`` does the same against ``E||x||^2`` instead, which is the
        total rather than the local size. They are alternatives; asking for
        both is a contradiction and is refused.
        """
        variances = self._resolve_variances(spectral_variances)
        if pointwise_std is not None and norm_std is not None:
            raise ValueError(
                "Give pointwise_std or norm_std, not both: they scale the same "
                "spectrum to two different targets."
            )
        if norm_std is not None:
            if norm_std <= 0.0:
                raise ValueError("norm_std must be positive.")
            # E||x||^2 is the trace of the covariance, which for a diagonal
            # covariance is the sum of its eigenvalues.
            current = float(np.sum(variances))
            if current <= 0.0:
                raise ValueError("A measure with no variance cannot be scaled.")
            variances = variances * (norm_std**2 / current)
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
        norm_std: float | None = None,
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
            norm_std=norm_std,
        )

    def heat_measure(
        self,
        time: float,
        /,
        *,
        amplitude: float = 1.0,
        expectation: np.ndarray | None = None,
        pointwise_std: float | None = None,
        norm_std: float | None = None,
    ) -> GaussianMeasure:
        """A Gaussian prior with a heat-kernel covariance."""
        return self.invariant_measure(
            amplitude**2 * self.heat_symbol(time),
            expectation=expectation,
            pointwise_std=pointwise_std,
            norm_std=norm_std,
        )

    # ----------------------------------------------------------------- #
    #                        Point evaluation                           #
    # ----------------------------------------------------------------- #

    def correlated_measure(
        self,
        spectral_cross_covariances: np.ndarray,
        /,
        *,
        expectation: Sequence[np.ndarray] | None = None,
        labels: Sequence[str] | None = None,
    ) -> GaussianMeasure:
        """Several fields on this domain, correlated scale by scale.

        The joint prior a coupled physical problem wants: the fields share a
        spectral basis, and at each mode their coefficients are drawn from one
        small covariance matrix rather than independently. That makes the
        correlation between the fields a function of *scale*, which is what
        distinguishes it from a single number multiplying two marginals.

        The measure lives on the direct sum of ``n`` copies of this space, and
        the ``(i, j)`` block of its covariance is diagonal with eigenvalues
        ``Sigma(k)[i, j]``. Its marginals are the invariant measures with
        spectral variances ``Sigma(k)[i, i]``.

        **Sampling is an extended Karhunen-Loeve expansion**, and it comes for
        free: the covariance *factor* is the block operator whose blocks carry
        the symmetric square roots ``L(k)``, so one draw of white noise on the
        direct sum, correlated mode by mode, is a sample. The ``1/sqrt(g)`` a
        non-trivial metric demands is carried by the white noise rather than
        written out here — which is the same argument as §13.1, one field wider.

        Args:
            spectral_cross_covariances: shape ``(dim, n, n)``; slice ``k`` is
                the covariance of the ``n`` fields' ``k``-th coefficients, and
                must be symmetric and positive semidefinite.
            expectation: one field per component. Defaults to zero.
            labels: names for the summands, for readability.
        """
        from ..algebra.direct_sum import BlockLinearOperator, DirectSum

        sigma = np.asarray(spectral_cross_covariances, dtype=float)
        if (
            sigma.ndim != 3
            or sigma.shape[0] != self.dim
            or sigma.shape[1] != sigma.shape[2]
        ):
            raise ValueError(f"Expected shape ({self.dim}, n, n), got {sigma.shape}.")
        count = sigma.shape[1]

        scale = float(np.max(np.abs(sigma))) if sigma.size else 0.0
        tolerance = 1.0e-10 * max(scale, np.finfo(float).tiny)
        if not np.allclose(sigma, np.swapaxes(sigma, -1, -2), atol=tolerance):
            raise ValueError("Each cross-covariance slice must be symmetric.")
        sigma = 0.5 * (sigma + np.swapaxes(sigma, -1, -2))

        values, vectors = np.linalg.eigh(sigma)
        if values.min(initial=0.0) < -tolerance:
            worst = int(np.argmin(values.min(axis=1)))
            raise ValueError(
                "Each cross-covariance slice must be positive semidefinite; "
                f"the one at component {worst} has smallest eigenvalue "
                f"{values.min():.3e}. For a pair of fields that means "
                "|correlation| <= 1."
            )
        values = np.clip(values, 0.0, None)
        # The symmetric square root, so that L L == Sigma and the factor is
        # its own adjoint block for block.
        roots = (vectors * np.sqrt(values)[:, None, :]) @ np.swapaxes(vectors, -1, -2)

        space = DirectSum([self] * count, labels=labels)
        factor = BlockLinearOperator(
            [
                [
                    DiagonalLinearOperator(self, np.ascontiguousarray(roots[:, i, j]))
                    for j in range(count)
                ]
                for i in range(count)
            ]
        )
        covariance = BlockLinearOperator(
            [
                [
                    DiagonalLinearOperator(self, np.ascontiguousarray(sigma[:, i, j]))
                    for j in range(count)
                ]
                for i in range(count)
            ]
        ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE)

        return GaussianMeasure(
            space,
            expectation=None if expectation is None else tuple(expectation),
            covariance=covariance,
            covariance_factor=factor,
        )

    def correlated_measure_from_correlations(
        self,
        variances: Sequence[np.ndarray],
        correlations: np.ndarray,
        /,
        *,
        expectation: Sequence[np.ndarray] | None = None,
        labels: Sequence[str] | None = None,
    ) -> GaussianMeasure:
        """The same, from marginal spectra and a correlation matrix.

        The parameterisation anyone actually has an opinion about: each field's
        own spectrum, and how strongly they are correlated. The correlation may
        be a single matrix, applying at every scale, or one per component if it
        varies with scale.

        Args:
            variances: one spectral variance array per field.
            correlations: ``(n, n)`` or ``(dim, n, n)``, with unit diagonal.
            expectation: one field per component. Defaults to zero.
            labels: names for the summands.
        """
        spectra = np.stack(
            [self._resolve_variances(variance) for variance in variances]
        )
        count = spectra.shape[0]
        matrix = np.asarray(correlations, dtype=float)
        if matrix.shape == (count, count):
            matrix = np.broadcast_to(matrix, (self.dim, count, count))
        elif matrix.shape != (self.dim, count, count):
            raise ValueError(
                f"Correlations must have shape ({count}, {count}) or "
                f"({self.dim}, {count}, {count}), got {matrix.shape}."
            )
        if not np.allclose(np.diagonal(matrix, axis1=-2, axis2=-1), 1.0):
            raise ValueError("A correlation matrix has a unit diagonal.")
        deviations = np.sqrt(spectra).T
        sigma = matrix * deviations[:, :, None] * deviations[:, None, :]
        return self.correlated_measure(sigma, expectation=expectation, labels=labels)

    def power_measure(
        self,
        power: np.ndarray | Callable[[np.ndarray], np.ndarray],
        /,
        *,
        expectation: np.ndarray | None = None,
    ) -> GaussianMeasure:
        """A Gaussian with a prescribed power *per degree*.

        The spectrum a geophysicist actually writes down: how much variance
        each degree holds in total, rather than how much each component holds.
        The two differ by the multiplicity, and dividing by it is the whole of
        this method.

        Args:
            power: total variance at each degree, indexed from zero, or a
                callable applied to the array of degrees.
            expectation: the mean. Defaults to zero.
        """
        degrees = self.degrees
        if callable(power):
            per_degree = np.asarray(power(degrees.astype(float)), dtype=float)
        else:
            values = np.asarray(power, dtype=float)
            if values.size <= degrees.max():
                raise ValueError(
                    f"Power is needed up to degree {degrees.max()}, got "
                    f"{values.size} values."
                )
            per_degree = values[degrees]
        counts = np.array([np.count_nonzero(degrees == d) for d in degrees])
        return self.invariant_measure(per_degree / counts, expectation=expectation)

    def covariance_function(
        self, measure: GaussianMeasure, distances: np.ndarray, /
    ) -> np.ndarray:
        """An invariant measure's covariance as a function of distance.

        Homogeneity is what makes this well defined: the two-point covariance
        depends on the pair of points only through the distance between them,
        so one anchor and a walk away from it gives the whole function.
        """
        anchor = self.reference_point
        field = measure.two_point_covariance(anchor)
        return self.evaluate(field, self.walk_from(anchor, distances))

    def pointwise_variance_at(
        self,
        measure: GaussianMeasure,
        points: Sequence[Any],
        /,
        *,
        rank: int = 0,
        samples: int | None = None,
        rng: Generator | None = None,
    ) -> np.ndarray:
        """``Var(x(p))`` at given points, for *any* measure on this space.

        The general counterpart of :meth:`pointwise_variance`, which is exact
        and constant but only for an invariant measure. A posterior is not
        invariant, and its pointwise variance is the interesting one.

        It is the diagonal of ``E C E*`` with ``E`` evaluation at the points,
        since ``(C u_i, u_i)`` is the variance at the ``i``-th. Exact by
        default, at one application of the covariance per point. Pass
        ``samples`` to estimate it instead, and ``rank`` to deflate first —
        which for a covariance with a decaying spectrum is the difference
        between a useful estimate and a useless one.
        """
        evaluation = self.point_evaluation_operator(points)
        operator = evaluation @ measure.covariance @ evaluation.adjoint
        if samples is None:
            return np.array(
                [
                    measure.directional_variance(self.dirac(point).representer)
                    for point in points
                ]
            )
        from ..numerics.randomised import deflated_diagonal

        return deflated_diagonal(
            operator, rank=rank, samples=samples, form="components", rng=rng
        )

    def walk_from(self, point: Any, distances: np.ndarray, /) -> list[Any]:
        """Points at given geodesic distances from a point, along one direction."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement walk_from."
        )

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

    def basis_matrix(self, points: Sequence[Any], /) -> np.ndarray:
        """The basis at many points, as a ``(len(points), dim)`` array.

        The rows of an observation operator's derivative matrix. A space whose
        transform can do this in a batch should override it; the sphere does.
        """
        return np.stack([self.basis_at(point) for point in points])

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

    def point_evaluation_operator(
        self, points: Sequence[Any], /, *, dense: bool = False
    ) -> LinearOperator:
        """Evaluation at several points, as an operator into a Euclidean space.

        Matrix-free by default: nothing of size ``len(points) x dim`` is
        formed, so this is usable at the scale real acquisition geometries
        reach.

        ``dense=True`` assembles the derivative matrix instead, which is worth
        it whenever the operator will be applied many times and the matrix
        fits. It is a separate argument rather than a call to
        :meth:`~pygeoinf2.algebra.operators.LinearOperator.assembled` because
        the rows are known in closed form — they are :meth:`basis_matrix` —
        while ``assembled`` would recover them by applying the adjoint once per
        datum, at a cost quadratic in the data. The adjoint is derived either
        way, so nothing is duplicated.

        Its adjoint returns a weighted sum of Dirac *representers*, which is
        what makes an adjoint-based inversion give a function rather than an
        array of numbers.
        """
        from ..algebra.spaces import EuclideanSpace

        points = tuple(points)
        if not points:
            raise ValueError("At least one point is needed.")
        codomain = EuclideanSpace(len(points))

        if dense:
            return LinearOperator.from_matrix(
                self, codomain, self.basis_matrix(points), form="galerkin"
            )
        return LinearOperator.from_derivative_callables(
            self,
            codomain,
            lambda x: self.evaluate(x, points),
            lambda y: self.accumulate(y, points),
        )

    # ----------------------------------------------------------------- #
    #                          Pointwise algebra                        #
    # ----------------------------------------------------------------- #

    def truncate(self, x: np.ndarray, /) -> np.ndarray:
        """The vector of this space with the same components as ``x``.

        The identity whenever the grid has exactly as many points as the space
        has dimensions, which is the case for a periodic box. It is *not* the
        identity on a sphere: the Driscoll-Healy grid is oversampled, so many
        grid arrays share a set of components and only one of them is in the
        span of the basis.

        That matters as soon as anything leaves the space. A pointwise product
        of two band-limited functions is not band-limited, so its grid array is
        one of those non-canonical representatives, and an operation that
        round-trips through components — the formal-adjoint lift does — would
        silently disagree with one that does not.
        """
        return self.from_components(self.to_components(x))

    def multiply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The pointwise product, truncated back into the space.

        Truncated rather than left on the grid so that the result depends only
        on the two vectors and not on which grid array happens to represent
        them; see :meth:`truncate`. The aliasing is still there — it is
        inherent to multiplying band-limited functions — but it is now
        committed to once, consistently, rather than resolved differently by
        each caller.
        """
        return self.truncate(np.asarray(x) * np.asarray(y))

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """The pointwise square root, truncated back into the space."""
        return self.truncate(np.sqrt(np.asarray(x)))

    def multiplication_operator(self, f: np.ndarray, /) -> LinearOperator:
        """The operator ``u -> f u``, with the metric handled.

        Multiplication is self-adjoint for the ``L2`` inner product, and a
        Sobolev space does not have that inner product — so on one, this builds
        the operator where it *is* self-adjoint and lifts it with
        :func:`lift_formal_adjoint`. The lifted operator is **not** claimed
        self-adjoint, because a formally self-adjoint operator is self-adjoint
        under the new metric only if it commutes with the ratio of the two, and
        multiplication by a varying field does not (§3.5).
        """
        if self.order == 0.0:
            return LinearOperator.self_adjoint(self, lambda u: self.multiply(f, u))
        base = self.with_order(0.0)
        return lift_formal_adjoint(base.multiplication_operator(f), self)

    # ----------------------------------------------------------------- #
    #                              Curvature                            #
    # ----------------------------------------------------------------- #

    @property
    def gaussian_curvature(self) -> float:
        """The Gaussian curvature of the domain, constant by homogeneity."""
        raise NotImplementedError(
            f"{type(self).__name__} does not state its Gaussian curvature."
        )

    def gradient_dot_product(self, f: np.ndarray, g: np.ndarray, /) -> np.ndarray:
        """``grad f . grad g``, pointwise, without ever forming a gradient.

        From the product rule for the Laplacian, with ``L == -div grad`` the
        *positive* Laplacian this package uses throughout:

        .. code-block:: text

            grad f . grad g == (f L(g) + g L(f) - L(f g)) / 2

        Every term is a pointwise product or a diagonal spectral multiply, so
        this costs three transforms and no differentiation of the grid. It is
        what makes the variable-coefficient flexure operator writable without a
        tangent frame.

        **The sign is the whole content of this method.** v1 has it the other
        way round, which is verifiable against ``grad sin . grad cos`` on a
        circle: its answer is exactly ``-1`` times the analytic one. It went
        unnoticed because every use is inside a variable-coefficient term, and
        those vanish identically when the coefficient is constant — so the
        constant-coefficient case, which is the one with a closed form to check
        against, cannot see it.
        """
        laplacian = self.laplacian
        result = self.multiply(f, laplacian(g))
        result = self.add(result, self.multiply(g, laplacian(f)))
        result = self.subtract(result, laplacian(self.multiply(f, g)))
        return self.scale(0.5, result)

    # ----------------------------------------------------------------- #
    #                               Flexure                             #
    # ----------------------------------------------------------------- #

    def _as_field(self, value: np.ndarray | float, /) -> np.ndarray:
        """A scalar as a constant field, or a field unchanged."""
        if isinstance(value, (int, float, np.floating, np.integer)):
            return self.project_function(lambda _: float(value))
        return np.asarray(value)

    def flexural_operator(
        self,
        rigidity: np.ndarray | float,
        poisson_ratio: np.ndarray | float,
        buoyancy: np.ndarray | float,
        /,
    ) -> LinearOperator:
        r"""The variable-coefficient flexure operator for a floating plate.

        The covariant fourth-order operator on a surface of Gaussian curvature
        ``K``, with ``D_eff == D (1 - nu)``:

        .. code-block:: text

            Op(w) = L(D L w)                        principal bending
                  - L(D_eff) L w                    rigidity-gradient coupling
                  + tr(Hess D_eff  Hess w)          twist coupling
                  + 2 K grad D_eff . grad w         curvature commutator
                  - K D_eff L w                     covariant softening
                  + rho_g w                         hydrostatic restoring force

        The two middle terms are produced together, coordinate-free, by
        subtracting a Bochner block built from :meth:`gradient_dot_product`;
        neither a Hessian nor a tangent frame is ever formed.

        Self-adjoint with respect to the ``L2`` inner product. On a Sobolev
        space it is built where that is true and lifted, as
        :meth:`multiplication_operator` is.

        Args:
            rigidity: the flexural rigidity ``D``, a field or a constant.
            poisson_ratio: ``nu``, a field or a constant.
            buoyancy: the restoring coefficient ``rho_g``, a field or a
                constant.
        """
        if self.order != 0.0:
            base = self.with_order(0.0)
            return lift_formal_adjoint(
                base.flexural_operator(rigidity, poisson_ratio, buoyancy), self
            )

        laplacian = self.laplacian
        curvature = self.gaussian_curvature
        rigidity_field = self._as_field(rigidity)
        effective = self.multiply(
            rigidity_field,
            self.subtract(self._as_field(1.0), self._as_field(poisson_ratio)),
        )
        buoyancy_field = self._as_field(buoyancy)
        laplacian_effective = laplacian(effective)

        def value(w: np.ndarray) -> np.ndarray:
            laplacian_w = laplacian(w)

            result = laplacian(self.multiply(rigidity_field, laplacian_w))

            # Bochner block: -bochner == tr(Hess D_eff Hess w) + 2 K grad.grad
            gradients = self.gradient_dot_product(effective, w)
            bochner = self.scale(0.5, laplacian(gradients))
            bochner = self.subtract(
                bochner,
                self.scale(0.5, self.gradient_dot_product(laplacian_effective, w)),
            )
            bochner = self.subtract(
                bochner,
                self.scale(0.5, self.gradient_dot_product(effective, laplacian_w)),
            )
            bochner = self.subtract(bochner, self.scale(curvature, gradients))

            result = self.subtract(result, bochner)
            result = self.subtract(
                result, self.multiply(laplacian_effective, laplacian_w)
            )
            if curvature != 0.0:
                result = self.subtract(
                    result,
                    self.scale(curvature, self.multiply(effective, laplacian_w)),
                )
            return self.add(result, self.multiply(buoyancy_field, w))

        return LinearOperator.self_adjoint(self, value)

    def inverse_flexural_operator(
        self,
        rigidity: np.ndarray | float,
        poisson_ratio: np.ndarray | float,
        buoyancy: np.ndarray | float,
        /,
        *,
        baseline_rigidity: float | None = None,
        baseline_buoyancy: float | None = None,
        solver: "IterativeSolver | None" = None,
    ) -> LinearOperator:
        r"""The inverse of :meth:`flexural_operator`.

        With constant coefficients the operator is invariant, so its inverse is
        exact and diagonal — the symbol is
        ``1 / (D lambda^2 - K D_eff lambda + rho_g)``. With varying
        coefficients that same symbol, built from the spatial averages, is an
        excellent preconditioner, and the solve is a preconditioned CG.

        Args:
            baseline_rigidity: the constant rigidity used for the
                preconditioner. Defaults to the spatial average.
            baseline_buoyancy: likewise for the restoring coefficient.
            solver: the iterative solver. Defaults to ``CGSolver()``.
        """
        constant = all(
            isinstance(value, (int, float, np.floating, np.integer))
            for value in (rigidity, poisson_ratio, buoyancy)
        )
        curvature = self.gaussian_curvature

        def symbol(D: float, nu: float, rho: float):
            effective = D * (1.0 - nu)
            return lambda eigenvalue: 1.0 / (
                D * eigenvalue**2 - curvature * effective * eigenvalue + rho
            )

        if constant:
            return self.invariant_operator(
                symbol(float(rigidity), float(poisson_ratio), float(buoyancy))
            )

        def average(value: np.ndarray | float) -> float:
            if isinstance(value, (int, float, np.floating, np.integer)):
                return float(value)
            one = self._as_field(1.0)
            return float(self.inner_product(value, one) / self.squared_norm(one))

        preconditioner = self.invariant_operator(
            symbol(
                average(rigidity) if baseline_rigidity is None else baseline_rigidity,
                average(poisson_ratio),
                average(buoyancy) if baseline_buoyancy is None else baseline_buoyancy,
            )
        )

        from ..numerics.solvers import CGSolver, IterativeSolver

        if solver is None:
            solver = CGSolver()
        if not isinstance(solver, IterativeSolver):
            raise TypeError(
                "A varying-coefficient flexure operator is inverted "
                f"iteratively; {type(solver).__name__} is not an "
                "IterativeSolver."
            )
        # CG needs the operator to be positive definite, and the flexure
        # operator is -- its quadratic form is the plate's bending plus
        # restoring energy -- provided D > 0 and rho_g > 0. That is a claim
        # about the *arguments*, so it is made here, by the routine choosing to
        # use CG, rather than by the operator itself. With an unphysical
        # rigidity CG will fail loudly instead of silently converging to
        # nothing.
        operator = self.flexural_operator(
            rigidity, poisson_ratio, buoyancy
        ).with_traits(Traits.POSITIVE_DEFINITE)
        return solver.with_preconditioner(preconditioner)(operator)

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
        dense: bool = False,
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

        weights = _weight_operator(len(paths), len(nodes), rows, columns, values)
        if dense:
            from ..algebra.spaces import EuclideanSpace

            matrix = weights.matrix(form="components") @ self.basis_matrix(nodes)
            return LinearOperator.from_matrix(
                self, EuclideanSpace(len(paths)), matrix, form="galerkin"
            )
        return weights @ self.point_evaluation_operator(nodes)

    def geodesic_ball_average_operator(
        self,
        centres: Sequence[Any],
        radius: float,
        /,
        *,
        count: int = 100,
        normalise: bool = True,
        dense: bool = False,
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

        weights = _weight_operator(len(centres), len(nodes), rows, columns, values)
        if dense:
            from ..algebra.spaces import EuclideanSpace

            matrix = weights.matrix(form="components") @ self.basis_matrix(nodes)
            return LinearOperator.from_matrix(
                self, EuclideanSpace(len(centres)), matrix, form="galerkin"
            )
        return weights @ self.point_evaluation_operator(nodes)

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
    over the same coordinate map.

    This is :meth:`LinearOperator.from_formal_adjoint` with the arguments in
    the order the symmetric-space code reads best, and with both sides required
    to be symmetric spaces. The general form handles direct sums, Euclidean
    sides and mass-weighted spaces too; use it directly for those.

    No self-adjointness is claimed: a formally self-adjoint operator is
    self-adjoint under the new metric only if it commutes with the ratio of the
    two, which for a general operator it does not. Claim it explicitly through
    ``traits`` when it holds, and verify with ``testing.check_traits``.

    Args:
        operator: the operator on the base spaces, with a working adjoint.
        domain: the space to present the operator's domain as.
        codomain: likewise for the codomain; defaults to ``domain``.
        traits: claims about the lifted operator.

    Returns:
        The same action, with an adjoint taken in the new metrics.

    Raises:
        ValueError: if the dimensions do not match.
    """
    return LinearOperator.from_formal_adjoint(
        domain,
        domain if codomain is None else codomain,
        operator,
        traits=traits,
    )
