"""
References: Gaussians equivalent to a prior, carrying their density against it.

A function-space sampler is built on a reference measure equivalent to the
prior, whose part of the dynamics it solves exactly. :class:`GaussianReference`
is one such Gaussian, the prior with the law of a few functionals replaced,
and :class:`MixtureReference` a mixture of them, for a posterior with several
modes. Both are measures in their own right; the estimators that fit them to
a posterior, :meth:`~pygeoinf.inference.gaussian.LinearGaussianInversion.reference`
and :meth:`~pygeoinf.inference.laplace.LaplaceApproximation.reference`, live in
:mod:`pygeoinf.inference`, and a sampled posterior fits one to its own run.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.random import Generator

from ..algebra.operators import LinearOperator
from ..traits import Traits
from .absolutely_continuous import AbsolutelyContinuousMeasure
from .gaussian import GaussianMeasure
from .mixture import GaussianMixture

__all__ = ["GaussianReference", "MixtureReference"]


class GaussianReference(GaussianMeasure, AbsolutelyContinuousMeasure):
    """A Gaussian equivalent to a prior, carrying its own density against it.

    The reference measure a function-space sampler is built on: the part of
    the dynamics it defines is solved exactly, and only the residual
    ``Phi + log d nu / d mu_0`` is left to the integrator and the
    Metropolis step. Its two readings describe one measure. The Gaussian
    moments are this measure's, and so is ``relative_log_density``, which
    is ``log d nu / d mu_0`` and normalized: the log normalizer is exactly
    zero. That is what separates it from a Laplace
    :class:`~pygeoinf.inference.gaussian.GaussianPosterior`, whose moments are the approximation and
    whose density is the exact posterior's.

    Every reference is the prior with the law of a finite-dimensional
    *chart* replaced. The chart is ``r`` bounded functionals,
    ``l(u) == ((w_k, u - m_0))_k``, distributed under the prior as
    ``N(0, K_l)`` with ``K_l == ((w_i, C w_j))``; the reference gives it
    the law ``N(a, S)`` instead and leaves the conditional law given the
    chart as the prior's. So in every direction outside the chart the
    reference *is* the prior, which makes it equivalent to the prior by
    construction, and

    .. code-block:: text

        draw     u == xi + C W K_l^-1 (z - l(xi)),   xi ~ prior,  z ~ N(a, S)
        density  log d nu / d mu_0 (u) == log n(l(u); a, S) - log n(l(u); 0, K_l)

    The draw costs a prior draw and ``r`` precomputed vectors, the density
    ``r`` inner products, and neither needs the prior's precision. Draws
    and density are built from the one chart, so they describe the same
    measure whatever the accuracy of the vectors that went into it.

    Internally the chart is whitened to ``K_l == I``, and directions in
    which the chart functionals are dependent under the prior (an
    eigenvalue of ``K_l`` below ``rtol`` times the largest) are dropped:
    there the prior cannot tell the functionals apart, so the reference
    cannot either.

    Args:
        prior: the Gaussian prior, which must be samplable and carry a
            covariance.
        chart: the vectors ``w_k`` whose inner products with ``u - m_0``
            are the chart coordinates.
        chart_mean: ``a``, the chart's mean under the reference, one entry
            per vector.
        chart_covariance: ``S``, the chart's covariance under the
            reference, symmetric and positive definite.
        images: ``C w_k`` for each chart vector, when already computed;
            computed here otherwise, one application of the prior
            covariance per vector.
        rtol: the relative size below which a direction of ``K_l`` is
            treated as dependent and dropped.

    Raises:
        TypeError: if the prior is not a ``GaussianMeasure``.
        ValueError: if the prior has no covariance or cannot be sampled,
            the chart's mean or covariance has the wrong shape, or the
            chart's covariance is not positive definite on the directions
            kept, without which the reference is not equivalent to the
            prior. An eigenvalue within rounding of zero, relative to the
            largest, counts as zero.
    """

    def __init__(
        self,
        prior: GaussianMeasure,
        chart: Sequence[Any],
        chart_mean: np.ndarray,
        chart_covariance: np.ndarray,
        /,
        *,
        images: Sequence[Any] | None = None,
        rtol: float = 1e-10,
    ) -> None:
        _check_prior(prior)
        vectors = list(chart)
        count = len(vectors)
        mean = np.asarray(chart_mean, dtype=float).reshape(-1)
        covariance = np.asarray(chart_covariance, dtype=float)
        if mean.shape != (count,) or covariance.shape != (count, count):
            raise ValueError(
                f"A chart of {count} vectors needs a mean of shape ({count},) "
                f"and a covariance of shape ({count}, {count}); got "
                f"{mean.shape} and {covariance.shape}."
            )
        transform, whitened, whitened_images = _whiten(
            prior, vectors, images, rtol=rtol
        )
        a = transform @ mean
        s = transform @ covariance @ transform.T
        self._build(prior, whitened, whitened_images, a, 0.5 * (s + s.T))

    def _build(
        self,
        prior: GaussianMeasure,
        whitened: list[Any],
        whitened_images: list[Any],
        a: np.ndarray,
        s: np.ndarray,
        /,
    ) -> None:
        """Set the reference up from a chart already whitened under the prior.

        An eigenvalue of ``s`` within rounding of zero, relative to the
        largest, counts as zero. Forming ``s`` leaves a singular covariance
        with such an eigenvalue of either sign, and a Cholesky factorization
        alone would accept the positive ones and divide by the tiny pivot.

        Raises:
            ValueError: if the chart's covariance is not positive definite.
        """
        from ..algebra.operators import Functional, LinearFunctional
        from ..numerics.randomized import Estimate

        space = prior.domain
        prior_covariance = prior.covariance
        rank = len(whitened)
        cholesky = np.zeros((0, 0))
        if rank:
            singular = ValueError(
                "The chart's covariance is not positive definite on the "
                "directions the prior distinguishes, so the reference would be "
                "singular to the prior."
            )
            try:
                values = np.linalg.eigvalsh(s)
                floor = 10 * rank * np.finfo(float).eps * np.abs(values).max()
                if not values[0] > floor:
                    raise singular
                cholesky = np.linalg.cholesky(s)
            except np.linalg.LinAlgError:
                raise singular from None

        self._prior = prior
        self._chart = tuple(whitened)
        self._images = tuple(whitened_images)
        self._chart_mean = a
        self._chart_covariance = s
        self._cholesky = cholesky
        self._log_det_half = float(np.sum(np.log(np.diag(cholesky)))) if rank else 0.0

        if rank == 0:
            expectation = prior.expectation
            density = Functional.from_callables(
                space,
                lambda x: 0.0,
                derivative=lambda x: LinearFunctional.from_representer(
                    space, space.zero()
                ),
            )
            super().__init__(
                space,
                expectation=expectation,
                covariance=prior_covariance,
                covariance_factor=prior.covariance_factor,
                precision=prior.precision,
                precision_factor=prior.precision_factor,
                sample=None if prior.covariance_factor is not None else self._fluctuate,
                unsafe=True,
            )
        else:
            expectation = prior.expectation
            for weight, image in zip(a, whitened_images):
                expectation = space.axpy(float(weight), image, space.copy(expectation))
            reduction = np.identity(rank) - s
            definite = Traits.POSITIVE_DEFINITE & prior_covariance.traits
            covariance_operator = LinearOperator.self_adjoint(
                space,
                self._apply_covariance,
                traits=Traits.POSITIVE_SEMIDEFINITE | definite,
            )
            self._reduction = reduction
            density = Functional.from_callables(
                space,
                self._density_value,
                derivative=lambda x: LinearFunctional.from_representer(
                    space, self._density_gradient(x)
                ),
            )
            super().__init__(
                space,
                expectation=expectation,
                covariance=covariance_operator,
                sample=self._fluctuate,
                unsafe=True,
            )
        self._init_density(prior, density, log_normalizer=Estimate(0.0, 0.0, 0))

    @classmethod
    def _from_whitened(
        cls,
        prior: GaussianMeasure,
        whitened: list[Any],
        whitened_images: list[Any],
        a: np.ndarray,
        s: np.ndarray,
        /,
    ) -> "GaussianReference":
        """A reference on a chart already whitened, which is kept as it is:
        what the components of a mixture are built from, so that they share
        one chart exactly."""
        reference = cls.__new__(cls)
        reference._build(prior, list(whitened), list(whitened_images), a, s)
        return reference

    # ----------------------------------------------------------------- #
    #                            Constructors                           #
    # ----------------------------------------------------------------- #

    @classmethod
    def from_prior(cls, prior: GaussianMeasure, /) -> "GaussianReference":
        """The prior as its own reference: a chart of no dimensions.

        Its density against the prior is zero, and it draws as the prior
        does.

        Args:
            prior: the Gaussian prior.

        Returns:
            The trivial reference.
        """
        return cls(prior, [], np.zeros(0), np.zeros((0, 0)))

    @classmethod
    def from_gaussian(
        cls,
        prior: GaussianMeasure,
        chart: Sequence[Any],
        /,
        *,
        mean: Any,
        covariance: LinearOperator,
        rtol: float = 1e-10,
    ) -> "GaussianReference":
        """The reference giving a chart the law a Gaussian gives it.

        The chart's mean and covariance are read off ``N(mean,
        covariance)``: ``a_k == (w_k, mean - m_0)`` and
        ``S_ij == (w_i, covariance w_j)``. Outside the chart the result is
        the prior. When the Gaussian's density against the prior depends on
        the model only through the chart -- a low-rank update of the prior
        in those directions, its mean shifted within them -- the result is
        that Gaussian; otherwise it is the nearest measure of chart form,
        and still a valid reference.

        Args:
            prior: the Gaussian prior.
            chart: the chart vectors.
            mean: the Gaussian's mean.
            covariance: the Gaussian's covariance, applied once per chart
                vector.
            rtol: as for the constructor.

        Returns:
            The reference.
        """
        space = prior.domain
        vectors = list(chart)
        offset = space.subtract(mean, prior.expectation)
        a = np.array([space.inner_product(w, offset) for w in vectors])
        mapped = [covariance(w) for w in vectors]
        s = np.array(
            [[space.inner_product(w, image) for image in mapped] for w in vectors]
        )
        return cls(prior, vectors, a, s.reshape(len(vectors), len(vectors)), rtol=rtol)

    # ----------------------------------------------------------------- #
    #                              Reading                              #
    # ----------------------------------------------------------------- #

    @property
    def prior(self) -> GaussianMeasure:
        """The prior this reference is equivalent to; also its ``reference``."""
        return self._prior

    @property
    def rank(self) -> int:
        """The chart's dimension after dependent directions are dropped."""
        return len(self._chart)

    @property
    def chart(self) -> tuple[Any, ...]:
        """The whitened chart vectors, ``(w_i, C w_j) == delta_ij``."""
        return self._chart

    @property
    def chart_mean(self) -> np.ndarray:
        """The whitened chart's mean under the reference."""
        return self._chart_mean.copy()

    @property
    def chart_covariance(self) -> np.ndarray:
        """The whitened chart's covariance under the reference."""
        return self._chart_covariance.copy()

    def fluctuation(self, *, rng: Generator | None = None) -> Any:
        """One centered draw, ``u - m``: a draw of ``N(0, K)``.

        What a momentum refresh and a prior Ornstein-Uhlenbeck step draw.

        Args:
            rng: the generator.

        Returns:
            A vector of the domain.
        """
        if self.rank == 0:
            return self._prior_fluctuation(rng)
        return self._fluctuate(rng)

    def coordinates(self, x: Any, /) -> np.ndarray:
        """The whitened chart coordinates of a model, ``((w_k, x - m_0))_k``.

        Args:
            x: a model.

        Returns:
            The ``rank`` coordinates.
        """
        space = self.domain
        offset = space.subtract(x, self._prior.expectation)
        return np.array([space.inner_product(w, offset) for w in self._chart])

    def pair_log_density(self, x: Any, y: Any, correlation: float, /) -> float:
        """``log d eta / d eta_0 (x, y)``: a correlated pair's law against the
        prior's.

        ``eta`` is the law of ``(x, y)`` with ``x`` drawn from this reference
        and ``y`` from its Ornstein-Uhlenbeck step of correlation ``c``,
        ``y == m + c (x - m) + sqrt(1 - c^2) xi``; ``eta_0`` the same for the
        prior. ``eta`` is ``eta_0`` with the law of the ``2 r`` chart
        coordinates replaced, so the density is ``2 r``-dimensional; in the
        coordinates ``p == (z_x + z_y) / sqrt(2 (1 + c))`` and
        ``q == (z_x - z_y) / sqrt(2 (1 - c))`` it splits into two
        ``r``-dimensional Gaussian ratios. It is what a Metropolis-Hastings
        ratio needs when the reference changes with the state.

        Args:
            x: the first model.
            y: the second.
            correlation: ``c``, in ``[0, 1)``.

        Returns:
            The log density, zero for the prior.

        Raises:
            ValueError: for a correlation outside ``[0, 1)``.
        """
        from scipy.linalg import solve_triangular

        if not 0.0 <= correlation < 1.0:
            raise ValueError(f"A correlation lies in [0, 1), got {correlation}.")
        if self.rank == 0:
            return 0.0
        zx, zy = self.coordinates(x), self.coordinates(y)
        p = (zx + zy) / np.sqrt(2.0 * (1.0 + correlation))
        q = (zx - zy) / np.sqrt(2.0 * (1.0 - correlation))
        center = self._chart_mean * np.sqrt(2.0 / (1.0 + correlation))
        dp = solve_triangular(self._cholesky, p - center, lower=True)
        dq = solve_triangular(self._cholesky, q, lower=True)
        return float(
            -0.5 * (dp @ dp + dq @ dq)
            - 2.0 * self._log_det_half
            + 0.5 * (p @ p + q @ q)
        )

    # ----------------------------------------------------------------- #
    #                             Internals                             #
    # ----------------------------------------------------------------- #

    def _prior_fluctuation(self, rng: Generator | None) -> Any:
        """A centered prior draw."""
        prior = self._prior
        return prior.domain.subtract(prior.sample(rng=rng), prior.expectation)

    def _fluctuate(self, rng: Generator | None) -> Any:
        """``xi + C W (z - l(xi))`` centered: the Matheron rule."""
        from ..algebra.spaces import resolve_rng

        rng = resolve_rng(rng)
        space = self.domain
        draw = self._prior_fluctuation(rng)
        if self.rank == 0:
            return draw
        coordinates = np.array([space.inner_product(w, draw) for w in self._chart])
        target = self._cholesky @ rng.standard_normal(self.rank)
        for weight, image in zip(target - coordinates, self._images):
            draw = space.axpy(float(weight), image, draw)
        return draw

    def _apply_covariance(self, x: Any) -> Any:
        """``C x - U (I - S) U* x`` with ``U`` the whitened chart's images."""
        space = self.domain
        result = self._prior.covariance(x)
        pairings = np.array([space.inner_product(image, x) for image in self._images])
        for weight, image in zip(self._reduction @ pairings, self._images):
            result = space.axpy(-float(weight), image, result)
        return result

    def _density_value(self, x: Any) -> float:
        """``log n(l; a, S) - log n(l; 0, I)`` at the chart coordinates."""
        from scipy.linalg import solve_triangular

        coordinates = self.coordinates(x)
        deviation = solve_triangular(
            self._cholesky, coordinates - self._chart_mean, lower=True
        )
        return float(
            -0.5 * deviation @ deviation
            - self._log_det_half
            + 0.5 * coordinates @ coordinates
        )

    def _density_gradient(self, x: Any) -> Any:
        """``W* (l - S^-1 (l - a))``, the representer of the density's derivative."""
        from scipy.linalg import cho_solve

        space = self.domain
        coordinates = self.coordinates(x)
        weights = coordinates - cho_solve(
            (self._cholesky, True), coordinates - self._chart_mean
        )
        result = space.zero()
        for weight, w in zip(weights, self._chart):
            result = space.axpy(float(weight), w, result)
        return result


class MixtureReference(GaussianMixture, AbsolutelyContinuousMeasure):
    """The prior with a finite chart's law replaced by a Gaussian mixture.

    :class:`GaussianReference` with ``N(a, S)`` on the chart replaced by
    ``sum_j p_j N(a_j, S_j)``: a mixture of Gaussian references that share
    one chart and differ only in its law, so it is equivalent to the prior
    and its density against it is ``r``-dimensional,

    .. code-block:: text

        log d nu / d mu_0 (u) == log sum_j p_j n(l(u); a_j, S_j) - log n(l(u); 0, I)

    in the whitened chart, normalized, with a gradient. It is what a posterior
    with several modes is fitted by, one component a mode.

    A rotation preserves only Gaussians, so a sampler uses it through its
    components: a chain carries the label of one, whose exact flows it
    follows, and changes label by :meth:`transport`, the affine map in the
    chart that carries one component onto another exactly.

    Args:
        prior: the Gaussian prior, samplable and carrying a covariance.
        chart: the vectors ``w_k`` of the chart, as for
            :class:`GaussianReference`.
        weights: the components' weights ``p_j``, normalized here.
        means: the chart's mean under each component, shape ``(count, r)``.
        covariances: the chart's covariance under each component, shape
            ``(count, r, r)``, each positive definite.
        images: ``C w_k`` for each chart vector, when already computed.
        rtol: the relative size below which a chart direction is dropped.

    Raises:
        TypeError: if the prior is not a ``GaussianMeasure``.
        ValueError: if the prior has no covariance or cannot be sampled,
            the chart is empty after dependent directions are dropped, the
            parameters have the wrong shapes or weights, or a component's
            covariance is not positive definite.
    """

    def __init__(
        self,
        prior: GaussianMeasure,
        chart: Sequence[Any],
        weights: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        /,
        *,
        images: Sequence[Any] | None = None,
        rtol: float = 1e-10,
    ) -> None:
        from ..algebra.operators import Functional, LinearFunctional
        from ..numerics.randomized import Estimate

        _check_prior(prior)
        vectors = list(chart)
        count = len(vectors)
        means = np.asarray(means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)
        components = means.shape[0] if means.ndim == 2 else 0
        if (
            components == 0
            or means.shape != (components, count)
            or covariances.shape != (components, count, count)
        ):
            raise ValueError(
                f"A chart of {count} vectors needs means of shape (components, "
                f"{count}) and covariances of shape (components, {count}, "
                f"{count}); got {means.shape} and {covariances.shape}."
            )
        transform, whitened, whitened_images = _whiten(
            prior, vectors, images, rtol=rtol
        )
        if not whitened:
            raise ValueError("A mixture reference needs a chart the prior can see.")
        parts = []
        for mean, covariance in zip(means, covariances):
            s = transform @ covariance @ transform.T
            parts.append(
                GaussianReference._from_whitened(
                    prior, whitened, whitened_images, transform @ mean, 0.5 * (s + s.T)
                )
            )
        super().__init__(parts, weights=weights)
        self._prior = prior
        self._chart = tuple(whitened)
        self._images = tuple(whitened_images)
        self._log_weights = np.log(np.maximum(self.weights, np.finfo(float).tiny))
        space = prior.domain
        density = Functional.from_callables(
            space,
            self._density_value,
            derivative=lambda x: LinearFunctional.from_representer(
                space, self._density_gradient(x)
            ),
        )
        self._init_density(prior, density, log_normalizer=Estimate(0.0, 0.0, 0))

    # ----------------------------------------------------------------- #
    #                              Reading                              #
    # ----------------------------------------------------------------- #

    @property
    def prior(self) -> GaussianMeasure:
        """The prior this reference is equivalent to; also its ``reference``."""
        return self._prior

    @property
    def components(self) -> list[GaussianReference]:
        """The Gaussian references mixed, sharing one whitened chart."""
        return list(self._components)

    @property
    def rank(self) -> int:
        """The chart's dimension after dependent directions are dropped."""
        return len(self._chart)

    @property
    def chart(self) -> tuple[Any, ...]:
        """The whitened chart vectors, shared by every component."""
        return self._chart

    def coordinates(self, x: Any, /) -> np.ndarray:
        """The whitened chart coordinates of a model, ``((w_k, x - m_0))_k``.

        Args:
            x: a model.

        Returns:
            The ``rank`` coordinates.
        """
        space = self.domain
        offset = space.subtract(x, self._prior.expectation)
        return np.array([space.inner_product(w, offset) for w in self._chart])

    def responsibilities(self, x: Any, /) -> np.ndarray:
        """``P(j | u)``: each component's share of the density at a model.

        Args:
            x: a model.

        Returns:
            The probabilities, one per component.
        """
        terms = self._log_terms(self.coordinates(x))
        terms = np.exp(terms - terms.max())
        return terms / terms.sum()

    def transport(
        self, x: Any, source: int, destination: int, /, *, centered: bool = False
    ) -> Any:
        """Carry a model from one component to another, exactly.

        In the chart, ``z' == a_k + L_k L_j^-1 (z - a_j)`` with ``L`` each
        component's Cholesky factor; outside it nothing changes, so the model
        moves by ``C W (z' - z)``, ``r`` precomputed vectors. The map pushes
        component ``j`` onto component ``k``, which is what makes a change of
        label an exact Metropolis move. A velocity, which has mean zero under
        every component, is carried with *centered*.

        Args:
            x: the model, or a velocity with *centered*.
            source: the component ``j`` it is drawn from.
            destination: the component ``k`` it is carried to.
            centered: carry a centered vector: the means are left out.

        Returns:
            The carried vector.
        """
        from scipy.linalg import solve_triangular

        space = self.domain
        start, end = self._components[source], self._components[destination]
        if centered:
            z = np.array([space.inner_product(w, x) for w in self._chart])
            mapped = end._cholesky @ solve_triangular(start._cholesky, z, lower=True)
        else:
            z = self.coordinates(x)
            deviation = solve_triangular(
                start._cholesky, z - start._chart_mean, lower=True
            )
            mapped = end._chart_mean + end._cholesky @ deviation
        result = space.copy(x)
        for weight, image in zip(mapped - z, self._images):
            result = space.axpy(float(weight), image, result)
        return result

    # ----------------------------------------------------------------- #
    #                             Internals                             #
    # ----------------------------------------------------------------- #

    def _log_terms(self, z: np.ndarray, /) -> np.ndarray:
        """``log p_j + log n(z; a_j, S_j)``, less the shared ``2 pi`` term."""
        from scipy.linalg import solve_triangular

        terms = np.empty(len(self._components))
        for j, component in enumerate(self._components):
            deviation = solve_triangular(
                component._cholesky, z - component._chart_mean, lower=True
            )
            terms[j] = (
                self._log_weights[j]
                - 0.5 * deviation @ deviation
                - component._log_det_half
            )
        return terms

    def _density_value(self, x: Any) -> float:
        """``logsumexp_j(...) - log n(z; 0, I)`` at the chart coordinates."""
        z = self.coordinates(x)
        terms = self._log_terms(z)
        largest = terms.max()
        return float(largest + np.log(np.exp(terms - largest).sum()) + 0.5 * z @ z)

    def _density_gradient(self, x: Any) -> Any:
        """``W* (z - sum_j r_j S_j^-1 (z - a_j))``, ``r_j`` the responsibilities."""
        from scipy.linalg import cho_solve

        space = self.domain
        z = self.coordinates(x)
        terms = self._log_terms(z)
        shares = np.exp(terms - terms.max())
        shares /= shares.sum()
        weights = z.copy()
        for share, component in zip(shares, self._components):
            weights -= share * cho_solve(
                (component._cholesky, True), z - component._chart_mean
            )
        result = space.zero()
        for weight, w in zip(weights, self._chart):
            result = space.axpy(float(weight), w, result)
        return result

    def __repr__(self) -> str:
        return f"MixtureReference({len(self._components)} components, rank {self.rank})"


def _check_prior(prior: Any, /) -> None:
    """Refuse a prior a reference cannot be built on.

    Raises:
        TypeError: if the prior is not a ``GaussianMeasure``.
        ValueError: if it has no covariance or cannot be sampled.
    """
    if not isinstance(prior, GaussianMeasure):
        raise TypeError(
            f"A reference is built on a GaussianMeasure prior, not a "
            f"{type(prior).__name__}."
        )
    if prior.covariance is None:
        raise ValueError("A reference needs a prior that carries its covariance.")
    if not prior.can_sample:
        raise ValueError("A reference needs a prior that can be sampled.")


def _whiten(
    prior: GaussianMeasure,
    vectors: list[Any],
    images: Sequence[Any] | None,
    /,
    *,
    rtol: float,
) -> tuple[np.ndarray, list[Any], list[Any]]:
    """The chart whitened under the prior, dependent directions dropped.

    Returns ``(T, whitened, images)`` with ``T`` the ``(rank, count)`` map
    from the given chart's coordinates to the whitened ones.

    Raises:
        ValueError: for a number of images other than the number of vectors.
    """
    space = prior.domain
    count = len(vectors)
    if images is None:
        images = [prior.covariance(w) for w in vectors]
    images = list(images)
    if len(images) != count:
        raise ValueError(f"Expected {count} images, one per chart vector.")
    if not count:
        return np.zeros((0, 0)), [], []
    gram = np.array(
        [[space.inner_product(w, image) for image in images] for w in vectors]
    )
    gram = 0.5 * (gram + gram.T)
    values, vecs = np.linalg.eigh(gram)
    largest = float(values.max()) if values.size else 0.0
    keep = values > rtol * largest if largest > 0.0 else values > np.inf
    transform = (vecs[:, keep] / np.sqrt(values[keep])).T
    whitened: list[Any] = []
    whitened_images: list[Any] = []
    for row in transform:
        w = space.zero()
        image = space.zero()
        for weight, vector, mapped in zip(row, vectors, images):
            w = space.axpy(float(weight), vector, w)
            image = space.axpy(float(weight), mapped, image)
        whitened.append(w)
        whitened_images.append(image)
    return transform, whitened, whitened_images
