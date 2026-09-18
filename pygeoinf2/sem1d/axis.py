"""Spaces of functions of one coordinate: what the interval and the radial
profile share.

Both are the degree-zero member of ``planetmodel``'s operator family on a
padded one-dimensional mesh, truncated to its leading eigenfunctions. They
differ in the measure -- ``dx`` on a line, ``r^2 dr`` along a radius -- and in
what follows from it: whether the mesh may cross zero, how the components are
scaled, where a point has a value. Everything else is here.
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable, Hashable, Sequence

import numpy as np
from planetmodel.randomfield import (
    RadialOperatorFamily,
    SpectralBasis,
    padded_mesh,
    restriction,
)

from ..algebra.operators import LinearFunctional
from .base import (
    _NODES_PER_MODE,
    Boundary,
    LengthScale,
    SpectralElementSpace,
    _clamped_length_scale,
    _length_scale_key,
    _resolved_boundary,
    _fitted_over_the_domain,
    _resolved_padding,
    _sampled_with_extension,
)

__all__ = ["AxisSpace"]


class AxisSpace(SpectralElementSpace):
    """Fields of one coordinate on ``[lower, upper]``, in the leading
    eigenfunctions of ``A`` on a padded mesh.

    Subclasses name the measure and the scale of the components, and supply
    ``_class_for_order``, :attr:`spatial_dimension` and ``_geometry``.
    """

    #: The measure, as ``planetmodel`` names it: "one" for dx, "r2" for r^2 dr.
    _weight: str = "one"
    #: What the planetmodel coefficients are multiplied by to give components.
    _scale: float = 1.0
    #: The tag of the geometry in the coordinate key.
    _geometry: str = "axis"

    def __init__(
        self,
        modes: int,
        lower: float,
        upper: float,
        /,
        *,
        order: float,
        length_scale: LengthScale,
        padding: float | tuple[float, float] | None,
        boundary: Boundary,
        ngll: int,
        element_length: float | None,
    ) -> None:
        """
        Args:
            modes: how many eigenfunctions of ``A`` on the padded mesh to keep.
            lower: the lower end of the domain.
            upper: the upper end.
            order: the Sobolev order.
            length_scale: ``L`` in ``A``, a number or a function of position.
            padding: the mesh's reach past each end.
            boundary: the condition at the ends of the mesh.
            ngll: Gauss-Lobatto-Legendre nodes per element.
            element_length: the longest an element may be.

        Raises:
            ValueError: if the ends are out of order, a length scale is not
                positive, or the mesh cannot hold the modes asked for.
        """
        lower, upper = float(lower), float(upper)
        if not lower < upper:
            raise ValueError("The ends of the domain must be in order.")
        modes = int(modes)
        if modes < 1:
            raise ValueError("At least one mode is needed.")

        clamped, probe = _clamped_length_scale(length_scale, lower, upper)
        pads = _resolved_padding(padding, probe, boundary)
        ends = None
        if self._weight == "r2":
            # The centre is a regular point: the mesh stops there.
            pads = (min(pads[0], lower), pads[1])
            ends = (lower - pads[0], upper + pads[1])
        robin = _resolved_boundary(boundary, probe, ends)

        ngll = int(ngll)
        if element_length is None:
            padded_length = upper - lower + pads[0] + pads[1]
            element_length = min(
                (ngll - 1) * padded_length / (_NODES_PER_MODE * modes),
                0.5 * float(probe.min()),
            )
        element_length = float(element_length)
        if element_length <= 0.0:
            raise ValueError("The element length must be positive.")

        mesh = padded_mesh(
            lower,
            upper,
            pad=pads,
            weight=self._weight,
            ngll=ngll,
            drmax=element_length,
        )
        if modes > mesh.nglob:
            raise ValueError(
                f"A mesh of {mesh.nglob} nodes cannot hold {modes} modes; "
                f"shorten element_length."
            )
        family = RadialOperatorFamily(
            mesh, kappa=lambda x: clamped(x) ** 2, weight=self._weight, robin=robin
        )
        self._basis = SpectralBasis(
            family, 0, restrict=restriction(mesh, lower, upper), nmodes=modes
        )

        self._bounds = (lower, upper)
        self._padding = pads
        self._robin = robin
        self._ngll = ngll
        self._element_length = element_length
        self._length_scale = length_scale
        self._length_scale_key = _length_scale_key(length_scale, clamped, mesh.rglob)
        super().__init__(self._basis.theta, order)

    # ----------------------------------------------------------------- #
    #                              Identity                             #
    # ----------------------------------------------------------------- #

    @property
    def bounds(self) -> tuple[float, float]:
        """The two ends of the domain proper."""
        return self._bounds

    @property
    def padding(self) -> tuple[float, float]:
        """The padding below and above the domain."""
        return self._padding

    @property
    def robin(self) -> tuple[float, float]:
        """The Robin coefficients at the two ends of the mesh, zero meaning
        the natural condition."""
        return self._robin

    @property
    def length_scale(self) -> LengthScale:
        """The length scale of ``A``, as it was given."""
        return self._length_scale

    @property
    def basis(self) -> SpectralBasis:
        """``planetmodel``'s truncated eigenbasis, which does the numerics."""
        return self._basis

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalue of ``A`` attached to each component: ascending, and
        at least one under the natural condition. Every metric and covariance
        here is a power of these."""
        return self._basis.theta

    def _coordinate_key(self) -> Hashable:
        """The mesh, the operator and the truncation, with the metric left out.

        Tagged by geometry and not by ``type(self)``, as the boxes' are:
        ``Lebesgue`` and ``Sobolev`` are views of one set of fields.
        """
        return (
            self._geometry,
            self._basis.nmodes,
            self._bounds,
            self._padding,
            self._robin,
            self._ngll,
            self._element_length,
            self._length_scale_key,
        )

    # ----------------------------------------------------------------- #
    #                          The coordinate map                       #
    # ----------------------------------------------------------------- #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The coefficients of the kept eigenfunctions: the projection in the
        mesh's own quadrature."""
        return self._scale * self._basis.analyse(x)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The nodal values on the padded mesh."""
        return self._basis.synthesise(
            np.asarray(c, dtype=float) / self._scale, physical=False
        )

    def components_of(self, vectors: Sequence[np.ndarray], /) -> np.ndarray:
        """The components of several vectors, in one product."""
        vectors = tuple(vectors)
        if not vectors:
            return np.zeros((self.dim, 0))
        return self._scale * self._basis.analyse(np.stack(vectors, axis=1))

    def vectors_from(self, columns: np.ndarray, /) -> list[np.ndarray]:
        """The vectors whose components are the columns, in one product."""
        values = self._basis.synthesise(
            np.asarray(columns, dtype=float) / self._scale, physical=False
        )
        return [np.array(values[:, j]) for j in range(values.shape[1])]

    # ----------------------------------------------------------------- #
    #                               The mesh                            #
    # ----------------------------------------------------------------- #

    @property
    def grid_shape(self) -> tuple[int]:
        """One axis: the nodes of the padded mesh."""
        return (self._basis.family.mesh.nglob,)

    @property
    def nodes(self) -> np.ndarray:
        """Where a vector's entries sit: the nodes of the padded mesh."""
        return self._basis.family.mesh.rglob

    @cached_property
    def interior_mask(self) -> np.ndarray:
        """A boolean array, true on the domain and false on the padding."""
        mask = np.zeros(self.nodes.size, dtype=bool)
        mask[self._basis.restriction.nodes] = True
        return mask

    @property
    def interior_nodes(self) -> np.ndarray:
        """The nodes of the domain proper, both ends included."""
        return self._basis.r

    @property
    def _interior_index(self) -> slice:
        return self._basis.restriction.nodes

    def project_function(
        self, function: Callable[[float], float], /, *, extension: str = "fit"
    ) -> np.ndarray:
        """The field of this space that a function of position names.

        The function is never called outside the domain, where it need not be
        defined. What is returned depends on what is asked of the padding.

        ``"fit"``, the default, asks nothing of it: the field *in the span of
        the kept modes* that best matches the function on the domain, in the
        domain's own quadrature. The modes are those of the padded mesh and
        nearly dependent on the domain alone, so the fit is a truncated one
        and picks the smallest answer. It converges as the function is smooth:
        measured on ``cos`` over ``[-1, 2]``, 3e-5, 3e-8 and 7e-14 at 32, 64
        and 128 modes, with coefficients no larger than the others'
        (DECISIONS.md D-124). It is what a mean, a reference model or a
        synthetic truth wants.

        The other two return the function's *own values at the nodes* and
        continue them across the padding, leaving :meth:`truncate` to settle
        the result into the span -- at 4e-3, 1e-3 and 3e-4 on the same
        ``cos``, since a continued function is only as smooth as its
        continuation and does not satisfy the mesh's end condition.
        ``"constant"`` gives the padding the value at the nearer end, and
        keeps a positive function positive there, which a standard deviation
        handed to ``pointwise_std=`` must be. ``"odd"`` gives it ``2 f(end) -
        f(mirror)``, which matches the slope too; under the natural condition
        and its longer padding that is ten to twenty times the better at an
        end, and under the default Robin one about two (D-119).

        Args:
            function: called with a position in the domain.
            extension: ``"fit"``, ``"odd"`` or ``"constant"``.

        Returns:
            The field. Under ``"fit"`` it is in the span of the kept modes;
            otherwise it holds the function's values on the domain's nodes.

        Raises:
            ValueError: for an extension that is none of these.
        """

        def sample(positions: np.ndarray) -> np.ndarray:
            return np.array([float(function(float(x))) for x in positions])

        if extension == "fit":
            interior = self._basis.modes()[self._basis.restriction.nodes]
            fitted = _fitted_over_the_domain(
                interior, self._basis.weights(), sample(self.interior_nodes)
            )
            return self._basis.synthesise(fitted, physical=False)
        return _sampled_with_extension(self.nodes, *self._bounds, sample, extension)

    # ----------------------------------------------------------------- #
    #                        Integrals and points                       #
    # ----------------------------------------------------------------- #

    def integral_functional(self) -> LinearFunctional:
        """The integral of a field over the domain proper, against the
        space's own measure.

        By the quadrature of the domain's elements alone, so the padding
        contributes nothing, its share of the end nodes included.
        """
        interior = self._basis.modes()[self._basis.restriction.nodes]
        return LinearFunctional.from_derivative_components(
            self, self._scale * (self._basis.weights() @ interior)
        )

    def basis_matrix(self, points: Sequence[float], /) -> np.ndarray:
        """The basis at many points, as a ``(len(points), dim)`` array. Exact,
        through the polynomial each eigenfunction is on each element.

        Raises:
            ValueError: if a point lies outside the domain.
        """
        positions = np.asarray(points, dtype=float).reshape(-1)
        return self._basis.evaluate(positions) / self._scale

    def _squared_modes(self, weights: np.ndarray, /) -> np.ndarray:
        return (self._basis.modes() ** 2) @ weights / self._scale**2
