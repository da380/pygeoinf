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
variance *field*; see :meth:`SpectralElementSpace.pointwise_variance` and
:meth:`SpectralElementSpace.spectral_measure`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Hashable, Literal, Sequence

import warnings
import numpy as np

from ..algebra.diagonal import DiagonalLinearOperator
from ..symmetric_space.base import _require_trace_class
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
from ._points import LocalEvaluation

__all__ = ["Boundary", "LengthScale", "SpectralElementSpace"]

#: A length scale: one number, or a function of position along the mesh.
type LengthScale = float | Callable[[np.ndarray], np.ndarray]

#: The condition at the ends of the mesh: ``"robin"``, the geometries' default,
#: for the Robin condition matched to the operator, ``None`` for the natural
#: (Neumann) one, or the Robin coefficients themselves, one for both ends or a
#: ``(lower, upper)`` pair with ``None`` for a natural end.
type Boundary = None | str | float | tuple[float | None, float | None]

#: The default padding at each end under the natural condition, in units of
#: the length scale there. The kernel of ``A^-p`` has fallen to about a tenth
#: at ``sqrt(8 nu)`` length scales, with ``nu`` the smoothness of the draws --
#: 3.5 at ``nu = 3/2`` -- and the boundary's mark with it.
_PADDING_LENGTH_SCALES = 4.0

#: The same under ``boundary="robin"``, with which two length scales of
#: padding do what four do under the natural condition.
_ROBIN_PADDING_LENGTH_SCALES = 2.0

#: The Robin coefficient of ``boundary="robin"``, in units of ``1 / L``. One
#: would be the decay of the solution of ``A u = 0`` and is exact for ``A^-1``
#: on a line; the priors in use are higher powers of ``A^-1``, for which 0.7
#: does better.
_ROBIN_COEFFICIENT = 0.7

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
    padding: float | tuple[float, float] | None,
    probe: np.ndarray,
    boundary: Boundary,
    /,
) -> tuple[float, float]:
    """The padding below and above: as given, or by default four length
    scales (two under ``boundary="robin"``), taking the length scale at each
    end of the domain.

    Raises:
        ValueError: if either is negative.
    """
    if padding is None:
        matched = isinstance(boundary, str) and boundary == "robin"
        scales = _ROBIN_PADDING_LENGTH_SCALES if matched else _PADDING_LENGTH_SCALES
        pads = (scales * float(probe[0]), scales * float(probe[-1]))
    elif np.ndim(padding) == 0:
        pads = (float(padding), float(padding))
    else:
        below, above = padding
        pads = (float(below), float(above))
    if min(pads) < 0.0:
        raise ValueError("The padding must be non-negative.")
    return pads


def _resolved_boundary(
    boundary: Boundary,
    probe: np.ndarray,
    ends: tuple[float, float] | None,
    /,
) -> tuple[float, float]:
    """The Robin coefficients at the two ends of the mesh.

    ``"robin"`` ends the mesh as if it went on: the condition the decaying
    solution of ``A u = 0`` satisfies there, ``u' = -+ gamma u``. On a line
    that solution is ``exp(-+ x / L)`` and ``gamma = 1 / L``; along a radius it
    is ``exp(-+ r / L) / r`` and ``gamma = 1 / L +- 1 / r``, the curvature
    adding at the outer end and subtracting at the inner, where it is not
    small. That is exact for ``A^-1``; for the higher powers priors are made
    of, ``0.7 / L`` in place of ``1 / L`` does better, and is what is used. An
    inner coefficient that would be negative is held at zero.

    Args:
        boundary: the condition asked for.
        probe: the length scale across the domain, its ends first and last.
        ends: the radii of the two ends of the mesh, under ``r^2 dr``; the
            curvature terms are left out if it is ``None``, as on a line. An
            end at the centre takes no condition.

    Raises:
        ValueError: for a name that is not ``"robin"``, or a negative
            coefficient.
    """
    if boundary is None:
        return (0.0, 0.0)
    if isinstance(boundary, str):
        if boundary != "robin":
            raise ValueError("boundary must be None, 'robin', a number or a pair.")
        lower = _ROBIN_COEFFICIENT / float(probe[0])
        upper = _ROBIN_COEFFICIENT / float(probe[-1])
        if ends is not None:
            lower = max(lower - 1.0 / ends[0], 0.0) if ends[0] > 0.0 else 0.0
            upper = upper + 1.0 / ends[1]
        return (lower, upper)
    if np.ndim(boundary) == 0:
        pair = (float(boundary), float(boundary))
    else:
        lower, upper = boundary
        pair = (
            0.0 if lower is None else float(lower),
            0.0 if upper is None else float(upper),
        )
    if min(pair) < 0.0:
        raise ValueError("Robin coefficients must be non-negative.")
    return pair


#: Singular values of a fit over the domain are kept down to this fraction of
#: the largest. The kept modes are those of the padded mesh and are nearly
#: dependent on the domain alone, so the fit is ill-posed by construction; this
#: is what picks the small solution out.
_FIT_RCOND = 1e-10

#: The ways a sampled function is continued past the domain.
_EXTENSIONS = ("odd", "constant", "fit")

#: A fit whose values on the padding exceed this multiple of its largest value
#: on the domain has leaked: see ``_checked_fit``.
_FIT_LEAK = 10.0


def _checked_fit(field: np.ndarray, interior: np.ndarray, /) -> np.ndarray:
    """*field*, after a warning if the fit has leaked into the padding.

    The fit over the domain is ill-posed: the kept modes restricted to the
    domain are nearly dependent, and the smallest solution that matches the
    values there can carry values on the padding thousands of times larger
    than any on the domain, which only the function's smoothness keeps in
    check. A function in the span is recovered exactly and a smooth one
    nearly so; a function with a jump or a kink is matched on the domain to
    the truncation's own error while the padding runs away. Nothing that
    reads the domain is affected; the norm of such a vector is. The remedy
    is to sample the function instead, continued past the domain by
    ``extension="constant"`` or ``"odd"``.
    """
    inside = np.abs(field[interior]).max(initial=0.0)
    outside = np.abs(field[~interior]).max(initial=0.0)
    if outside > _FIT_LEAK * max(inside, np.finfo(float).tiny):
        warnings.warn(
            f"The fit over the domain has leaked into the padding: values up "
            f"to {outside:.3g} there against {inside:.3g} on the domain. The "
            "function is not smooth enough for the fit; sample it with "
            "extension='constant' or 'odd' instead.",
            RuntimeWarning,
            stacklevel=3,
        )
    return field


def _fitted_over_the_domain(
    modes: np.ndarray, weights: np.ndarray, values: np.ndarray, /
) -> np.ndarray:
    """The coefficients of the kept modes that best give ``values`` on the
    domain, in its own quadrature, and nothing about the padding.

    ``modes`` is ``(domain nodes, kept modes)``, ``weights`` the quadrature of
    the domain's elements, ``values`` ``(domain nodes,)`` or that by any number
    of columns. By a truncated singular value decomposition, so the answer is
    the smallest that fits.
    """
    root = np.sqrt(weights)
    left, singular, right = np.linalg.svd(root[:, None] * modes, full_matrices=False)
    keep = singular > _FIT_RCOND * singular[0]
    target = (root * values.T).T
    return right[keep].T @ ((left[:, keep].T @ target).T / singular[keep]).T


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
        ValueError: for an extension other than ``"odd"`` or ``"constant"``.
            ``"fit"`` is not a continuation and is the caller's to handle
            before coming here.
    """
    if extension not in ("odd", "constant"):
        raise ValueError(
            "extension must be 'odd' or 'constant' here; 'fit' is handled by "
            "the caller."
        )
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


#: The memory a dense point evaluation may take, in bytes: under ``form="auto"``
#: an evaluation whose matrix, eight bytes per point per component, would be
#: larger is matrix-free instead. A memory budget and not a speed threshold:
#: the matrix-free form does per point about the work of a row of the matrix,
#: so an application costs about the same either way, and only the dense form
#: holds the rows. On a ball of 4223 components at 83 000 points, the dense
#: matrix 2.8 GB, an application took 0.07 s dense and 0.045 s or 0.15 s
#: matrix-free, with the harmonics at the points kept or recomputed.
_DENSE_EVALUATION_LIMIT = 256 * 2**20

#: The forms a point evaluation can take.
_EVALUATION_FORMS = ("auto", "dense", "local")


def _evaluation_form(form: str, points: int, dim: int, /) -> str:
    """``"dense"`` or ``"local"``: the form asked for, or under ``"auto"`` the
    dense one while its matrix fits :data:`_DENSE_EVALUATION_LIMIT`.

    Raises:
        ValueError: for a form that is none of ``"auto"``, ``"dense"`` and
            ``"local"``.
    """
    if form not in _EVALUATION_FORMS:
        raise ValueError(f"form must be 'auto', 'dense' or 'local', got {form!r}.")
    if form == "auto":
        return "local" if 8 * points * dim > _DENSE_EVALUATION_LIMIT else "dense"
    return form


def _local_operator(
    space: HilbertSpace, codomain: EuclideanSpace, evaluation: Any, /
) -> LinearOperator:
    """Evaluation at points as an operator, from a matrix-free evaluation.

    *space* has coordinates, a spectral-element space or a layered sum of
    them; *evaluation* has the methods of :class:`LocalEvaluation`, ``values``
    from components and ``derivative_components`` from data. The action reads
    a vector through its components; the adjoint takes the derivative
    components the evaluation spreads the data onto, applies the inverse
    metric, which is diagonal, and synthesizes. Both are also given on
    components, so that a composition with a covariance in the eigenbasis
    stays there.
    """

    def components_adjoint(data: np.ndarray) -> np.ndarray:
        return space.solve_gram(evaluation.derivative_components(data))

    return LinearOperator.from_callables(
        space,
        codomain,
        lambda x: evaluation.values(space.to_components(x)),
        adjoint=lambda data: space.from_components(components_adjoint(data)),
        components_action=evaluation.values,
        components_adjoint_action=components_adjoint,
    )


class SpectralElementSpace(
    ArrayVectorMixin, HilbertModule[np.ndarray], DiagonalMetricSpace[np.ndarray]
):
    """A space in the leading eigenfunctions of ``A`` on a padded mesh.

    A vector is the array of a field's values on the padded grid; its
    components are the coefficients of the kept eigenfunctions; the metric is
    ``eigenvalues ** order``. Subclasses supply :meth:`to_components`,
    :meth:`from_components`, :attr:`eigenvalues`, :attr:`grid_shape`,
    :attr:`interior_mask`, :attr:`bounds`, :meth:`basis_matrix`,
    ``_local_evaluation``, :meth:`_squared_modes`, :attr:`spatial_dimension`,
    ``_coordinate_key`` and ``_class_for_order``, and tag the geometry they
    name with ``_geometry``.
    """

    def __init__(self, eigenvalues: np.ndarray, order: float, /) -> None:
        """
        Args:
            eigenvalues: the eigenvalue of ``A`` belonging to each component.
            order: the Sobolev order, of which the metric is the power.
        """
        self._order = float(order)
        super().__init__(np.asarray(eigenvalues, dtype=float) ** self._order)

    @property
    def is_function_space(self) -> bool:
        """True: the space discretises functions on its domain."""
        return True

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

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        """The two ends of the domain proper along the coordinate the mesh
        runs along: the endpoints of an interval, the inner and outer radius
        of a ball. Where layers of one geometry meet."""

    @abstractmethod
    def basis_matrix(self, points: Sequence[Any], /) -> np.ndarray:
        """The basis at many points of the domain, as a ``(len(points), dim)``
        array: the rows of an observation operator's derivative matrix."""

    @abstractmethod
    def _local_evaluation(self, points: Sequence[Any], /) -> LocalEvaluation:
        """Evaluation at the points from the components, and its transpose,
        through the elements the points lie in and without the basis matrix.

        Raises:
            ValueError: if a point lies outside the domain.
        """

    @abstractmethod
    def _squared_modes(self, weights: np.ndarray, /) -> np.ndarray:
        """``sum_k weights_k phi_k(p)^2`` at every point of the grid."""

    @abstractmethod
    def _coordinate_key(self) -> Hashable:
        """Identifies the mesh, the operator and the truncation, with the
        metric left out: the fields, and not how they are measured."""

    @abstractmethod
    def _class_for_order(self, order: float, /) -> type:
        """The subclass that names a space of this geometry and that order:
        the geometry's ``Lebesgue`` at order zero, its ``Sobolev`` otherwise."""

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
            The space, as the geometry's ``Lebesgue`` at order zero and its
            ``Sobolev`` otherwise.
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
        """The pointwise product, left on the grid: the exact product there,
        and not its projection onto the kept modes, which is
        :meth:`truncate`'s to give."""
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

    def point_evaluation_order(
        self, /, *, points: Sequence[Any] | None = None
    ) -> float:
        """The Sobolev order a space must exceed for points to have values.

        Half the spatial dimension, wherever the points are; a geometry in
        which it depends on them overrides this.

        Args:
            points: the points in question. Any point of the domain if
                omitted.

        Returns:
            The order that must be exceeded.
        """
        return self.spatial_dimension / 2.0

    def _require_point_evaluation(
        self,
        what: str,
        /,
        *,
        points: Sequence[Any] | None = None,
        side: str | None = None,
        unsafe: bool,
    ) -> None:
        """Refuse point evaluation at or below :meth:`point_evaluation_order`,
        where it has no representer, unless *unsafe*. ``points`` are the ones
        in question, any point of the domain when omitted; ``side`` is unused
        here and belongs to the signature a layered space shares."""
        threshold = self.point_evaluation_order(points=points)
        if unsafe or self._order > threshold:
            return
        raise ValueError(
            f"{what} needs a Sobolev order above {threshold:g} at these "
            f"points, and this space has "
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
        ``x``, which off the grid is the only field there is to evaluate.

        Matrix-free, as :meth:`point_evaluation_operator` is under
        ``form="local"``, and the same to rounding as the rows of
        :meth:`basis_matrix` applied to the components.

        Args:
            x: a vector of this space.
            points: where to evaluate.

        Returns:
            One value per point.

        Raises:
            ValueError: if a point lies outside the domain.
        """
        return self._local_evaluation(list(points)).values(self.to_components(x))

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
        self._require_point_evaluation(
            "A Dirac functional", points=[point], unsafe=unsafe
        )
        return LinearFunctional.from_derivative_components(self, self.basis_at(point))

    def point_evaluation_operator(
        self,
        points: Sequence[Any],
        /,
        *,
        unsafe: bool = False,
        form: Literal["auto", "dense", "local"] = "auto",
    ) -> LinearOperator:
        """Evaluation at several points, as an operator into a Euclidean space.

        Two forms, which agree to rounding and differ in memory. ``"dense"``
        assembles the rows, :meth:`basis_matrix`, known in closed form: it
        holds eight bytes per point per component, and an application is a
        product with that matrix. ``"local"`` holds no matrix. Along the mesh
        a field is the polynomial through its nodal values on each element,
        so a point's value needs only the nodes of its element and, in a ball
        or a disk, the harmonics at its angles: the form holds the ``ngll``
        Lagrange weights of each point and, in a ball or a disk, the angular
        functions at the points, all of them while those fit 256 MB and
        otherwise one chunk of points at a time, recomputed in every
        application. An application synthesizes the coefficient functions at
        the nodes, which is cheap next to the rest, and does per point about
        the work of a row of the matrix: measured on a ball, a little less
        than the dense product with the angular functions kept, and about
        twice it when they are recomputed. ``"auto"``, the default, is the
        dense form while its matrix fits 256 MB, which keeps the matrix at
        hand for whatever reads it, and the local form beyond.

        Both read a vector through its components, the field the space
        holds, so they agree on any grid array, a pointwise product left off
        the span included, and the adjoint is exact on all of them.

        The adjoint returns a weighted sum of Dirac representers: the data
        give derivative components through the transposed rows, the inverse
        of the diagonal metric is applied, and the vector is synthesized.
        Under ``"local"`` the transposed rows are the same contractions run
        back -- each datum times the angular functions at its point spread
        onto the nodes of its element, and the coefficient functions so
        accumulated projected onto the kept modes -- so there is no system to
        solve. The operator's action on components, and its adjoint's, go
        through no transform, and a composition with a covariance in the
        eigenbasis stays on them.

        Args:
            points: where to evaluate.
            unsafe: build it even on a space too rough to admit it.
            form: ``"auto"``, ``"dense"`` or ``"local"``.

        Returns:
            The operator.

        Raises:
            ValueError: if no points are given, one lies outside the domain,
                the form is none of the three, or the order is at or below
                the order :meth:`point_evaluation_order` gives and *unsafe*
                is not set.
        """
        points = list(points)
        if not points:
            raise ValueError("At least one point is needed.")
        form = _evaluation_form(form, len(points), self.dim)
        self._require_point_evaluation(
            "A point evaluation operator", points=points, unsafe=unsafe
        )
        codomain = EuclideanSpace(len(points))
        if form == "dense":
            return LinearOperator.from_matrix(
                self, codomain, self.basis_matrix(points), form="galerkin"
            )
        return _local_operator(self, codomain, self._local_evaluation(points))

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

        ``sum_k s_k phi_k(p)^2 / g_k``, as on a symmetric space: the metric
        value ``g_k`` enters because the basis functions ``phi_k`` are
        orthogonal in the space's inner product with squared norm ``g_k``,
        not orthonormal. The result is a
        field and not a number, since nothing here is homogeneous: the
        eigenfunctions of a varying ``L`` are larger where it is smaller, and
        those of any ``L`` feel the ends of the mesh.

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
        unsafe: bool = False,
    ) -> GaussianMeasure:
        """A Gaussian prior with covariance ``amplitude^2 A^-order``.

        The workhorse prior, and a Matern field with the correlation length
        ``L`` the space was built with, varying as that does. There is no
        ``scale`` argument, as a symmetric space's has, because the basis
        that makes this diagonal is the eigenbasis of one ``A``.

        The covariance is trace class on this space, so that this is a
        Gaussian measure on it with draws in it, exactly when ``order``
        exceeds half the spatial dimension ``d``, whatever the space's own
        order: a draw then has ``order - d/2`` more derivatives than the
        space asks. At or below that the "measure" has no draws in the
        space (at order zero it is the identity covariance), and it is
        refused.

        Args:
            order: the Sobolev order of the measure, above ``d/2``.
            amplitude: an overall scale on the spectrum.
            expectation: the mean. Zero if omitted.
            pointwise_std: calibrate by the pointwise standard deviation, one
                number or a field; see :meth:`spectral_measure`.
            unsafe: build it at an order at or below ``d/2`` anyway, for a
                test that means it.

        Returns:
            The measure.

        Raises:
            ValueError: for an order at or below half the spatial dimension
                without *unsafe*.
        """
        _require_trace_class(self, order, unsafe)
        return self.spectral_measure(
            amplitude**2 * self.eigenvalues ** (-float(order)),
            expectation=expectation,
            pointwise_std=pointwise_std,
        )
