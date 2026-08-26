"""
Spaces whose basis diagonalises the Laplacian.

An invariant space is a coordinate space with a distinguished spectral basis:
one in which the Laplace-Beltrami operator is diagonal. Everything that makes
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

__all__ = ["InvariantSpace", "lift_formal_adjoint"]


class InvariantSpace(ArrayVectorMixin, DiagonalMetricSpace[np.ndarray]):
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

    def invariant_measure(
        self,
        spectral_variances: np.ndarray | Callable[[np.ndarray], np.ndarray],
        /,
        *,
        expectation: np.ndarray | None = None,
    ) -> GaussianMeasure:
        """A Gaussian whose covariance is diagonal in the spectral basis.

        The factor is the exact square root of the covariance, so sampling is a
        single spectral multiply of white noise — and the white noise carries
        the ``1/sqrt(g)`` that a non-trivial metric demands, rather than that
        correction being written out here.
        """
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
        )

    def heat_measure(
        self,
        time: float,
        /,
        *,
        amplitude: float = 1.0,
        expectation: np.ndarray | None = None,
    ) -> GaussianMeasure:
        """A Gaussian prior with a heat-kernel covariance."""
        return self.invariant_measure(
            amplitude**2 * self.heat_symbol(time), expectation=expectation
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

    def point_evaluation_operator(self, points: Sequence[Any], /) -> LinearOperator:
        """Evaluation at several points, as an operator into a Euclidean space.

        Its adjoint returns a weighted sum of Dirac *representers*, which is
        what makes an adjoint-based inversion give a function rather than an
        array of numbers.
        """
        from ..algebra.spaces import EuclideanSpace

        points = tuple(points)
        if not points:
            raise ValueError("At least one point is needed.")
        matrix = np.stack([self.basis_at(point) for point in points])
        return LinearOperator.from_derivative_matrix(
            self, EuclideanSpace(len(points)), matrix
        )

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


def lift_formal_adjoint(
    operator: LinearOperator,
    domain: InvariantSpace,
    /,
    *,
    codomain: InvariantSpace | None = None,
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
