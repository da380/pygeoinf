"""A bounded interval, in the eigenbasis of ``A = 1 - d/dx (L^2 d/dx)``.

The counterpart of :mod:`~pygeoinf2.symmetric_space.line` for a length scale
``L(x)`` that varies, and for a mesh that is not uniform. The line embeds its
interval in a circle and diagonalizes the Laplacian there by an FFT; this one
embeds it in a longer interval, meshes that with Gauss-Lobatto-Legendre
spectral elements, and solves for the eigenfunctions of ``A`` on the mesh. The
padding plays the same part in both: ``A`` needs a boundary condition at each
end of its mesh, a natural (Neumann) one here, and whichever is chosen
distorts a field within about a correlation length of where it acts. Moving
the ends away from ``[lower, upper]`` moves the distortion out of the domain.

In the eigenbasis every power of ``A`` is diagonal, so the conveniences of a
symmetric space carry over with ``theta``, the eigenvalues of ``A``, in the
place of ``1 + L^2 k^2``: the ``H^s`` inner product is ``sum theta^s c^2``, a
Sobolev prior has covariance ``theta^-p``, and a draw is one synthesis. What
does *not* carry over is homogeneity. The pointwise variance of such a prior
depends on where it is asked for, which is why :meth:`Interval.sobolev_measure`
calibrates by a standard deviation *field* rather than by one number.

**The conventions are the package's.** A vector is the array of the field's
values at the nodes of the padded mesh, as a ``Line``'s is its values on the
padded grid; its components are the coefficients of the kept eigenfunctions;
a product is left on the nodes, untruncated (DECISIONS.md D-87); point
evaluation is refused below the order that admits it (D-11).

The numerics are ``planetmodel.randomfield``'s: ``padded_mesh``,
``RadialOperatorFamily`` under its plain measure, and the ``SpectralBasis`` of
degree zero, which is reachable as :attr:`Interval.basis`.
"""

from __future__ import annotations

from functools import cached_property
from typing import Callable, Hashable, Sequence

import numpy as np
from numpy.random import Generator
from planetmodel.randomfield import (
    RadialOperatorFamily,
    SpectralBasis,
    padded_mesh,
    restriction,
)

from ..algebra.operators import LinearFunctional
from .base import (
    _NODES_PER_MODE,
    LengthScale,
    SpectralElementSpace,
    _clamped_length_scale,
    _length_scale_key,
    _resolved_padding,
    _sampled_with_extension,
)

__all__ = ["Interval", "Lebesgue", "Sobolev"]


class Interval(SpectralElementSpace):
    """Fields on ``[lower, upper]``, in the leading eigenfunctions of ``A``."""

    def __init__(
        self,
        modes: int,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        order: float = 0.0,
        length_scale: LengthScale = 1.0,
        padding: float | tuple[float, float] | None = None,
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: how many eigenfunctions of ``A`` on the padded mesh to
                keep, which is the dimension of the space.
            lower: the left endpoint.
            upper: the right endpoint.
            order: the Sobolev order. Zero gives the Lebesgue space.
            length_scale: ``L`` in ``A = 1 - d/dx (L^2 d/dx)``, a number or a
                function of position. It is asked about ``[lower, upper]``
                only; the padding takes its value at the nearer endpoint.
            padding: how far the mesh extends past each endpoint, one length
                or a ``(below, above)`` pair. Four length scales by default,
                those at the endpoints.
            ngll: Gauss-Lobatto-Legendre nodes per element.
            element_length: the longest an element may be. By default short
                enough that the mesh has two and a half nodes for every mode
                kept and an element is no longer than half the smallest
                length scale.

        Raises:
            ValueError: if the endpoints are out of order, a length scale is
                not positive, or the mesh cannot hold the modes asked for.
        """
        lower, upper = float(lower), float(upper)
        if not lower < upper:
            raise ValueError("The endpoints must satisfy lower < upper.")
        modes = int(modes)
        if modes < 1:
            raise ValueError("At least one mode is needed.")

        clamped, probe = _clamped_length_scale(length_scale, lower, upper)
        pads = _resolved_padding(padding, probe)

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
            lower, upper, pad=pads, weight="one", ngll=ngll, drmax=element_length
        )
        if modes > mesh.nglob:
            raise ValueError(
                f"A mesh of {mesh.nglob} nodes cannot hold {modes} modes; "
                f"shorten element_length."
            )
        family = RadialOperatorFamily(
            mesh, kappa=lambda x: clamped(x) ** 2, weight="one"
        )
        self._basis = SpectralBasis(
            family, 0, restrict=restriction(mesh, lower, upper), nmodes=modes
        )

        self._bounds = (lower, upper)
        self._padding = pads
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
        """The endpoints ``(lower, upper)`` of the domain proper."""
        return self._bounds

    @property
    def padding(self) -> tuple[float, float]:
        """The padding below and above the domain."""
        return self._padding

    @property
    def length_scale(self) -> LengthScale:
        """The length scale of ``A``, as it was given."""
        return self._length_scale

    @property
    def spatial_dimension(self) -> int:
        """One."""
        return 1

    @property
    def domain_volume(self) -> float:
        """The length of the domain proper, excluding the padding."""
        return self._bounds[1] - self._bounds[0]

    @property
    def basis(self) -> SpectralBasis:
        """``planetmodel``'s truncated eigenbasis, which does the numerics."""
        return self._basis

    @property
    def eigenvalues(self) -> np.ndarray:
        """The eigenvalue of ``A`` attached to each component: ascending, and
        at least one. Every metric and covariance here is a power of these."""
        return self._basis.theta

    def _coordinate_key(self) -> Hashable:
        """The mesh, the operator and the truncation, with the metric left out.

        Tagged by geometry and not by ``type(self)``, as the boxes' are:
        ``Lebesgue`` and ``Sobolev`` are views of one set of fields.
        """
        return (
            "radial_interval",
            self._basis.nmodes,
            self._bounds,
            self._padding,
            self._ngll,
            self._element_length,
            self._length_scale_key,
        )

    def _class_for_order(self, order: float, /) -> type:
        return Lebesgue if order == 0.0 else Sobolev

    def __repr__(self) -> str:
        lower, upper = self._bounds
        return (
            f"{type(self).__name__}(modes={self.dim}, bounds=[{lower:g}, "
            f"{upper:g}], order={self._order:g})"
        )

    # ----------------------------------------------------------------- #
    #                          The coordinate map                       #
    # ----------------------------------------------------------------- #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        """The coefficients of the kept eigenfunctions: ``Phi^T M x``, the
        projection in the mesh's own quadrature."""
        return self._basis.analyse(x)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        """The nodal values ``Phi c`` on the padded mesh."""
        return self._basis.synthesise(c, physical=False)

    def components_of(self, vectors: Sequence[np.ndarray], /) -> np.ndarray:
        """The components of several vectors, in one product."""
        vectors = tuple(vectors)
        if not vectors:
            return np.zeros((self.dim, 0))
        return self._basis.analyse(np.stack(vectors, axis=1))

    def vectors_from(self, columns: np.ndarray, /) -> list[np.ndarray]:
        """The vectors whose components are the columns, in one product."""
        values = self._basis.synthesise(
            np.asarray(columns, dtype=float), physical=False
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
        """The nodes of the domain proper, endpoints included."""
        return self._basis.r

    @property
    def _interior_index(self) -> slice:
        return self._basis.restriction.nodes

    def project_function(
        self, function: Callable[[float], float], /, *, extension: str = "odd"
    ) -> np.ndarray:
        """Sample a function at the nodes, continuing it across the padding.

        The function is never called outside the domain, where it need not be
        defined. A padding node is given ``2 f(end) - f(mirror)``, with the
        mirror image taken in the nearer endpoint: the odd reflection through
        the endpoint's own value, which continues the function with its slope.
        Holding it constant instead leaves a kink at the endpoint, and the
        kept modes come to a kink at first order -- measured on ``cos`` over
        ``[-1, 2]``, the error at the endpoint is ten to twenty times that of
        the reflection, whose error over the domain is that of knowing the
        function on the padding (DECISIONS.md D-119). Nothing has to bring the
        field to zero, the far ends of the mesh being free and not periodic.

        Args:
            function: called with a position in ``[lower, upper]``.
            extension: ``"odd"``, the reflection, or ``"constant"``, the
                endpoint's value. The reflection of a positive function need
                not be positive; the constant is.

        Returns:
            The sampled field, which :meth:`truncate` settles into the span of
            the kept modes.

        Raises:
            ValueError: for an extension that is neither.
        """
        return _sampled_with_extension(
            self.nodes,
            *self._bounds,
            lambda positions: np.array([float(function(float(x))) for x in positions]),
            extension,
        )

    def random_point(self, *, rng: Generator | None = None) -> float:
        """A point of the domain, drawn uniformly."""
        rng = np.random.default_rng() if rng is None else rng
        return float(rng.uniform(*self._bounds))

    # ----------------------------------------------------------------- #
    #                        Integrals and points                       #
    # ----------------------------------------------------------------- #

    def integral_functional(self) -> LinearFunctional:
        """The integral of a field over the domain proper.

        By the quadrature of the elements of ``[lower, upper]`` alone, so the
        padding contributes nothing, its share of the endpoint nodes included.
        """
        interior = self._basis.modes()[self._basis.restriction.nodes]
        return LinearFunctional.from_derivative_components(
            self, self._basis.weights() @ interior
        )

    def basis_matrix(self, points: Sequence[float], /) -> np.ndarray:
        """The basis at many points, as a ``(len(points), dim)`` array. Exact,
        through the polynomial each eigenfunction is on each element.

        Raises:
            ValueError: if a point lies outside the domain.
        """
        return self._basis.evaluate(np.asarray(points, dtype=float).reshape(-1))

    def _squared_modes(self, weights: np.ndarray, /) -> np.ndarray:
        return (self._basis.modes() ** 2) @ weights


class Lebesgue(Interval):
    """The ``L2`` space on an interval."""

    def __init__(
        self,
        modes: int,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        length_scale: LengthScale = 1.0,
        padding: float | tuple[float, float] | None = None,
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: eigenfunctions kept, the dimension of the space.
            lower, upper: the endpoints.
            length_scale: ``L`` in ``A``, which fixes the basis and the
                default padding though not this space's inner product.
            padding: the mesh's reach past each endpoint.
            ngll: nodes per element.
            element_length: the longest an element may be.
        """
        super().__init__(
            modes,
            lower=lower,
            upper=upper,
            order=0.0,
            length_scale=length_scale,
            padding=padding,
            ngll=ngll,
            element_length=element_length,
        )


class Sobolev(Interval):
    """The Sobolev space ``H^order`` on an interval, with the inner product
    ``(A^order u, v)``."""

    def __init__(
        self,
        modes: int,
        order: float,
        length_scale: LengthScale,
        /,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
        padding: float | tuple[float, float] | None = None,
        ngll: int = 5,
        element_length: float | None = None,
    ) -> None:
        """
        Args:
            modes: eigenfunctions kept, the dimension of the space.
            order: the Sobolev order.
            length_scale: ``L`` in ``A``, a number or a function of position:
                the length at which the Sobolev weight turns over.
            lower, upper: the endpoints.
            padding: the mesh's reach past each endpoint.
            ngll: nodes per element.
            element_length: the longest an element may be.
        """
        super().__init__(
            modes,
            lower=lower,
            upper=upper,
            order=order,
            length_scale=length_scale,
            padding=padding,
            ngll=ngll,
            element_length=element_length,
        )
