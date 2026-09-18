"""What the spectral-element spaces share: everything but the geometry.

A space here is a coordinate space in the eigenbasis of ``A = 1 - div(L^2
grad)`` on a padded mesh. Once a geometry says how fields go to coefficients
and back, which eigenvalue of ``A`` belongs to each coefficient, and what the
eigenfunctions are worth at a point, the rest is the same on an interval and
in a ball, and is written here once: the Sobolev scale, the pointwise algebra,
point evaluation and its guard, and the Gaussian measures.

It is the part of :class:`~pygeoinf2.symmetric_space.base.SymmetricSpace`
that does not lean on homogeneity, under the same names. What is missing is
what does lean on it -- a reference point, geodesics, one number for the
pointwise variance of an invariant prior -- and what takes its place is a
variance *field*; see :meth:`SpectralElementSpace.spectral_measure`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Hashable, Sequence

import numpy as np

from ..algebra.diagonal import DiagonalLinearOperator
from ..algebra.operators import LinearFunctional, LinearOperator
from ..algebra.spaces import (
    ArrayVectorMixin,
    DiagonalMetricSpace,
    EuclideanSpace,
    HilbertModule,
    HilbertSpace,
)
from ..probability.gaussian import GaussianMeasure
from ..traits import Traits

__all__ = ["LengthScale", "SpectralElementSpace"]

#: A length scale: one number, or a function of position along the mesh.
type LengthScale = float | Callable[[np.ndarray], np.ndarray]

#: The default padding at each end, in units of the length scale there. The
#: kernel of ``A^-p`` has fallen to about a tenth at ``sqrt(8 nu)`` length
#: scales, with ``nu`` the smoothness of the draws -- 3.5 at ``nu = 3/2`` --
#: and the boundary's mark with it.
_PADDING_LENGTH_SCALES = 4.0

#: Mesh nodes per half wavelength of the shortest mode kept, under the default
#: element length. The upper part of a spectral-element spectrum is the mesh's
#: and not the operator's, so the modes kept should be the lower part of what
#: the mesh can hold.
_NODES_PER_MODE = 2.5


def _clamped_length_scale(
    length_scale: LengthScale, lower: float, upper: float, /
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """The length scale as a function on the whole mesh, and its values across
    ``[lower, upper]``.

    It is asked about the domain only, where it is defined; the padding takes
    its value at the nearer end.

    Raises:
        ValueError: if it is not positive and finite across the domain.
    """
    if callable(length_scale):

        def clamped(x: np.ndarray) -> np.ndarray:
            values = length_scale(np.clip(np.asarray(x, dtype=float), lower, upper))
            return np.broadcast_to(np.asarray(values, dtype=float), np.shape(x))

    else:

        def clamped(x: np.ndarray) -> np.ndarray:
            return np.full(np.shape(x), float(length_scale))

    probe = clamped(np.linspace(lower, upper, 257))
    if not np.all(np.isfinite(probe)) or np.any(probe <= 0.0):
        raise ValueError("The length scale must be positive and finite.")
    return clamped, probe


def _resolved_padding(
    padding: float | tuple[float, float] | None, probe: np.ndarray, /
) -> tuple[float, float]:
    """The padding below and above: as given, or four length scales, those at
    the ends of the domain.

    Raises:
        ValueError: if either is negative.
    """
    if padding is None:
        pads = (
            _PADDING_LENGTH_SCALES * float(probe[0]),
            _PADDING_LENGTH_SCALES * float(probe[-1]),
        )
    elif np.ndim(padding) == 0:
        pads = (float(padding), float(padding))
    else:
        below, above = padding
        pads = (float(below), float(above))
    if min(pads) < 0.0:
        raise ValueError("The padding must be non-negative.")
    return pads


def _sampled_with_extension(
    nodes: np.ndarray,
    lower: float,
    upper: float,
    sample: Callable[[np.ndarray], np.ndarray],
    extension: str,
    /,
) -> np.ndarray:
    """A function's values along a padded mesh, from its values on the domain.

    ``sample`` takes positions in ``[lower, upper]`` and returns the values
    there, positions along the first axis; it is never asked about anywhere
    else. Across the padding the values are ``2 f(end) - f(mirror)``, the odd
    reflection through the end's own value, which matches the function and its
    slope there; or ``f(end)``, the constant, which matches the function alone.
    A mirror image that would fall past the far end of the domain stops at it.

    Raises:
        ValueError: for an extension that is neither.
    """
    if extension not in ("odd", "constant"):
        raise ValueError("extension must be 'odd' or 'constant'.")
    nodes = np.asarray(nodes, dtype=float)
    values = np.array(sample(np.clip(nodes, lower, upper)), dtype=float)
    if extension == "odd":
        for end, outside in ((lower, nodes < lower), (upper, nodes > upper)):
            if np.any(outside):
                mirror = np.clip(2.0 * end - nodes[outside], lower, upper)
                values[outside] = 2.0 * values[outside] - sample(mirror)
    return values


def _length_scale_key(
    length_scale: LengthScale,
    clamped: Callable[[np.ndarray], np.ndarray],
    nodes: np.ndarray,
    /,
) -> Hashable:
    """What identifies the operator: the number, or the nodal values of the
    function, a callable itself being equal only to itself."""
    if callable(length_scale):
        return clamped(nodes).tobytes()
    return float(length_scale)


class SpectralElementSpace(
    ArrayVectorMixin, HilbertModule[np.ndarray], DiagonalMetricSpace[np.ndarray]
):
    """A space in the leading eigenfunctions of ``A`` on a padded mesh.

    A vector is the array of a field's values on the padded grid; its
    components are the coefficients of the kept eigenfunctions; the metric is
    ``eigenvalues ** order``. Subclasses supply :meth:`to_components`,
    :meth:`from_components`, :attr:`eigenvalues`, :attr:`grid_shape`,
    :attr:`interior_mask`, :meth:`basis_matrix`, :meth:`_squared_modes`,
    :attr:`spatial_dimension`, ``_coordinate_key`` and ``_class_for_order``.
    """

    def __init__(self, eigenvalues: np.ndarray, order: float, /) -> None:
        """
        Args:
            eigenvalues: the eigenvalue of ``A`` belonging to each component.
            order: the Sobolev order, of which the metric is the power.
        """
        self._order = float(order)
        super().__init__(np.asarray(eigenvalues, dtype=float) ** self._order)

    # ----------------------------------------------------------------- #
    #                         Subclass interface                        #
    # ----------------------------------------------------------------- #

    @property
    @abstractmethod
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalue of ``A`` attached to each component, at least one.
        Every metric and covariance here is a power of these."""

    @property
    @abstractmethod
    def spatial_dimension(self) -> int:
        """The dimension of the domain the fields live on."""

    @property
    @abstractmethod
    def grid_shape(self) -> tuple[int, ...]:
        """The shape of a vector: the padded grid's."""

    @property
    @abstractmethod
    def interior_mask(self) -> np.ndarray:
        """A boolean grid array, true on the domain and false on the padding."""

    @abstractmethod
    def basis_matrix(self, points: Sequence[Any], /) -> np.ndarray:
        """The basis at many points of the domain, as a ``(len(points), dim)``
        array: the rows of an observation operator's derivative matrix."""

    @abstractmethod
    def _squared_modes(self, weights: np.ndarray, /) -> np.ndarray:
        """``sum_k weights_k phi_k(p)^2`` at every point of the grid."""

    @abstractmethod
    def _coordinate_key(self) -> Hashable:
        """Identifies the mesh, the operator and the truncation, with the
        metric left out: the fields, and not how they are measured."""

    @abstractmethod
    def _class_for_order(self, order: float, /) -> type:
        """The subclass that names a space of this geometry and that order
        (DECISIONS.md D-3)."""

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    @property
    def order(self) -> float:
        """The Sobolev order. Zero means the inner product is the ``L2`` one."""
        return self._order

    def _key(self) -> Hashable:
        return (self._coordinate_key(), self._order)

    def shares_vectors_with(self, other: HilbertSpace, /) -> bool:
        """True for another space over the same mesh, operator and modes,
        whatever its order: the fields are the same and only their measure
        differs, so a formal-adjoint lift passes a vector straight through."""
        if self is other:
            return True
        if not isinstance(other, SpectralElementSpace):
            return False
        return self._coordinate_key() == other._coordinate_key()

    def with_order(self, order: float, /) -> "SpectralElementSpace":
        """The same fields, measured with a different Sobolev order.

        The eigenproblem is not solved again: the new space shares this
        one's basis.

        Args:
            order: the new order.

        Returns:
            The space, as ``Lebesgue`` at order zero and ``Sobolev`` otherwise
            (DECISIONS.md D-3).
        """
        order = float(order)
        new = object.__new__(self._class_for_order(order))
        new.__dict__.update(self.__dict__)
        SpectralElementSpace.__init__(new, self.eigenvalues, order)
        return new

    # ----------------------------------------------------------------- #
    #                          Spectral operators                       #
    # ----------------------------------------------------------------- #

    def spectral_operator(
        self, values: np.ndarray, /, *, traits: Traits = Traits.NONE
    ) -> DiagonalLinearOperator:
        """A diagonal operator from an explicit value per component.

        A function of ``A`` is ``space.spectral_operator(f(space.eigenvalues))``.

        Args:
            values: one per component.
            traits: further claims. Self-adjointness and definiteness are
                deduced from the values, so they seldom need supplying.

        Returns:
            The diagonal operator.

        Raises:
            ValueError: for the wrong number of values.
        """
        array = np.asarray(values, dtype=float)
        if array.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim} values, got {array.shape}.")
        return DiagonalLinearOperator(self, array, traits=traits)

    def order_inclusion_operator(
        self, target: "SpectralElementSpace", /
    ) -> LinearOperator:
        """The inclusion into the same fields under another order.

        Args:
            target: a space over the same mesh, operator and modes.

        Returns:
            The operator, whose action is the identity on vectors and whose
            adjoint carries the ratio of the two metrics.

        Raises:
            ValueError: if the target does not hold the same fields.
        """
        if not self.shares_vectors_with(target):
            raise ValueError("The target is not the same fields under another order.")
        return LinearOperator.from_formal_adjoint(
            self, target, LinearOperator.identity(self)
        )

    # ----------------------------------------------------------------- #
    #                          Pointwise algebra                        #
    # ----------------------------------------------------------------- #

    def zero(self) -> np.ndarray:
        """A new array of zeros on the padded grid."""
        return np.zeros(self.grid_shape)

    def truncate(self, x: np.ndarray, /) -> np.ndarray:
        """The vector in the span of the kept modes with the components of
        ``x``: what a product, or a sampled function, is taken for."""
        return self.from_components(self.to_components(x))

    def interior_values(self, x: np.ndarray, /) -> np.ndarray:
        """A field's values on the part of the grid that lies in the domain.

        Args:
            x: a vector of this space.

        Returns:
            The values, with the padding's left out.
        """
        return np.asarray(x)[self._interior_index]

    @property
    def _interior_index(self) -> Any:
        """What indexes the domain out of a grid array. The mask by default,
        which flattens; a geometry whose domain is a slab gives the slice."""
        return self.interior_mask

    def multiply(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The pointwise product, left on the grid (DECISIONS.md D-87): the
        exact product there, and not its projection onto the kept modes,
        which is :meth:`truncate`'s to give."""
        return np.asarray(x) * np.asarray(y)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """The pointwise square root, left on the grid."""
        return np.sqrt(np.asarray(x))

    def multiplication_operator(self, f: np.ndarray, /) -> LinearOperator:
        """The operator ``u -> f u``, with the metric handled.

        Self-adjoint on the Lebesgue space, and claimed so: analysis is the
        transpose of synthesis under the grid's own quadrature weights, none
        of them negative, so the form is ``sum_j w_j f_j u_j v_j`` whatever
        ``f`` is. On a Sobolev space it is built there and lifted, claiming
        nothing, since multiplication by a varying field does not commute
        with the metric.

        Args:
            f: the field to multiply by.

        Returns:
            The operator.
        """
        if self._order == 0.0:
            return LinearOperator.self_adjoint(self, lambda u: self.multiply(f, u))
        base = self.with_order(0.0)
        return LinearOperator.from_formal_adjoint(
            self, self, base.multiplication_operator(f)
        )

    def support_projection(self) -> LinearOperator:
        """Multiplication by the domain's indicator: the field with its
        padding set to zero. An orthogonal projection on the Lebesgue space,
        up to the truncation; lifted, and claiming nothing, on a Sobolev one."""
        return self.multiplication_operator(self.interior_mask.astype(float))

    # ----------------------------------------------------------------- #
    #                          Point evaluation                         #
    # ----------------------------------------------------------------- #

    def _require_point_evaluation(self, what: str, /, *, unsafe: bool) -> None:
        """Refuse point evaluation at or below half the spatial dimension
        (DECISIONS.md D-11), where it has no representer."""
        threshold = self.spatial_dimension / 2.0
        if unsafe or self._order > threshold:
            return
        raise ValueError(
            f"{what} needs a Sobolev order above {threshold:g} on a "
            f"{self.spatial_dimension}-dimensional domain, and this space has "
            f"order {self._order:g}. Below that a point evaluation is not a "
            f"bounded functional: it has no representer, and the one this "
            f"would return is a mesh-scale artifact with no limit as the modes "
            f"rise. Raise the order, or pass unsafe=True if you want to see "
            f"that for yourself."
        )

    def basis_at(self, point: Any, /) -> np.ndarray:
        """The value of each kept eigenfunction at a point of the domain: the
        derivative components of evaluation there."""
        return self.basis_matrix([point])[0]

    def evaluate(self, x: np.ndarray, points: Sequence[Any], /) -> np.ndarray:
        """The values at several points of the field with the components of
        ``x``, which off the grid is the only field there is to evaluate."""
        return self.basis_matrix(points) @ self.to_components(x)

    def dirac(self, point: Any, /, *, unsafe: bool = False) -> LinearFunctional:
        """The evaluation functional at a point of the domain.

        Args:
            point: where to evaluate.
            unsafe: build it even on a space too rough to admit it.

        Returns:
            The functional, from derivative components, so that the metric is
            applied once and in the adjoint.

        Raises:
            ValueError: if the order is at or below half the spatial
                dimension and *unsafe* is not set, or the point lies outside
                the domain.
        """
        self._require_point_evaluation("A Dirac functional", unsafe=unsafe)
        return LinearFunctional.from_derivative_components(self, self.basis_at(point))

    def point_evaluation_operator(
        self, points: Sequence[Any], /, *, unsafe: bool = False
    ) -> LinearOperator:
        """Evaluation at several points, as an operator into a Euclidean space.

        Assembled: the rows are :meth:`basis_matrix`, known in closed form.
        The adjoint is derived from them and returns a weighted sum of Dirac
        representers.

        Args:
            points: where to evaluate.
            unsafe: build it even on a space too rough to admit it.

        Returns:
            The operator.

        Raises:
            ValueError: if no points are given, one lies outside the domain,
                or the order is at or below half the spatial dimension and
                *unsafe* is not set.
        """
        points = list(points)
        if not points:
            raise ValueError("At least one point is needed.")
        self._require_point_evaluation("A point evaluation operator", unsafe=unsafe)
        return LinearOperator.from_matrix(
            self,
            EuclideanSpace(len(points)),
            self.basis_matrix(points),
            form="galerkin",
        )

    # ----------------------------------------------------------------- #
    #                              Measures                             #
    # ----------------------------------------------------------------- #

    def _resolve_variances(self, spectral_variances: np.ndarray, /) -> np.ndarray:
        """Validate one variance per component."""
        variances = np.asarray(spectral_variances, dtype=float)
        if variances.shape != (self.dim,):
            raise ValueError(
                f"Got {variances.shape} variances for dimension {self.dim}."
            )
        if np.any(variances < 0.0):
            raise ValueError("Spectral variances must be non-negative.")
        return variances

    def pointwise_variance(self, spectral_variances: np.ndarray, /) -> np.ndarray:
        """The variance of ``x(p)`` at every grid point, under a diagonal
        covariance.

        ``sum_k s_k phi_k(p)^2 / g_k``, as on a symmetric space and with the
        metric for the same reason (DECISIONS.md D-26) -- but a field and not
        a number, since nothing here is homogeneous: the eigenfunctions of a
        varying ``L`` are larger where it is smaller, and those of any ``L``
        feel the ends of the mesh.

        Args:
            spectral_variances: the covariance operator's eigenvalues, one per
                component.

        Returns:
            The variance on the padded grid, a vector of the space.

        Raises:
            ValueError: for the wrong number of variances, or a negative one.
        """
        variances = self._resolve_variances(spectral_variances)
        return self._squared_modes(variances / self.metric_values)

    def spectral_measure(
        self,
        spectral_variances: np.ndarray,
        /,
        *,
        expectation: np.ndarray | None = None,
        pointwise_std: float | np.ndarray | None = None,
    ) -> GaussianMeasure:
        """A Gaussian whose covariance is diagonal in the eigenbasis, or is
        one such rescaled from place to place.

        Without ``pointwise_std`` the covariance is the diagonal operator of
        the variances, its factor the exact square root, and a draw one
        synthesis.

        With it the draws are those of the diagonal measure multiplied by the
        field ``pointwise_std / raw_std``, with ``raw_std`` the square root of
        :meth:`pointwise_variance`, so that the standard deviation at every
        grid point is the one asked for and the correlation structure is
        untouched. The covariance is then ``S C S*`` with ``S`` the
        multiplication, the factor ``S C^1/2``, and nothing about it is
        diagonal; there is no closed-form precision. A symmetric space does
        this with one number because its raw standard deviation *is* one
        number.

        Args:
            spectral_variances: the diagonal covariance's eigenvalues, one per
                component.
            expectation: the mean. Zero if omitted.
            pointwise_std: the standard deviation of the field, one number or
                a vector of the space giving it point by point, on the
                padding too.

        Returns:
            The measure.

        Raises:
            ValueError: for negative variances, the wrong number of them, a
                standard deviation that is not positive, or a raw variance
                that vanishes somewhere.
        """
        variances = self._resolve_variances(spectral_variances)
        diagonal = DiagonalLinearOperator(self, np.sqrt(variances))
        if pointwise_std is None:
            precision = (
                DiagonalLinearOperator(self, 1.0 / variances)
                if np.all(variances > 0.0)
                else None
            )
            return GaussianMeasure(
                self,
                expectation=expectation,
                covariance_factor=diagonal,
                precision=precision,
            )

        target = np.broadcast_to(
            np.asarray(pointwise_std, dtype=float), self.grid_shape
        )
        if np.any(target <= 0.0):
            raise ValueError("pointwise_std must be positive.")
        raw = self.pointwise_variance(variances)
        if np.any(raw <= 0.0):
            raise ValueError(
                "A measure with zero pointwise variance somewhere cannot be "
                "scaled to a given standard deviation."
            )
        scaling = self.multiplication_operator(target / np.sqrt(raw))
        return GaussianMeasure(
            self, expectation=expectation, covariance_factor=scaling @ diagonal
        )

    def sobolev_measure(
        self,
        order: float,
        /,
        *,
        amplitude: float = 1.0,
        expectation: np.ndarray | None = None,
        pointwise_std: float | np.ndarray | None = None,
    ) -> GaussianMeasure:
        """A Gaussian prior with covariance ``amplitude^2 A^-order``.

        The workhorse prior, and a Matern field: its draws have smoothness
        ``nu = order + self.order - d/2`` on a ``d``-dimensional domain, and
        the correlation length ``L`` the space was built with, varying as
        that does. There is no ``scale`` argument, as a symmetric space's
        has, because the basis that makes this diagonal is the eigenbasis of
        one ``A``.

        Args:
            order: the Sobolev order of the measure. The draws are functions
                when ``order + self.order`` exceeds half the spatial
                dimension.
            amplitude: an overall scale on the spectrum.
            expectation: the mean. Zero if omitted.
            pointwise_std: calibrate by the pointwise standard deviation, one
                number or a field; see :meth:`spectral_measure`.

        Returns:
            The measure.
        """
        return self.spectral_measure(
            amplitude**2 * self.eigenvalues ** (-float(order)),
            expectation=expectation,
            pointwise_std=pointwise_std,
        )
