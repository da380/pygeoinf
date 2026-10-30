"""The layered disk: several disk spaces end to end, one field per layer."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..algebra.operators import LinearOperator
from ..algebra.spaces import EuclideanSpace
from ..spectral_element import layered as _layered
from ..traits import Traits

from .space import Disk


class Layered(_layered.Layered):
    """Piecewise-continuous fields of position in the plane: a :class:`Disk`
    in each of the concentric rings between consecutive radii.

    A direct sum of the layers, one summand per layer on its own padded mesh
    with its own length scale and truncation, that knows its summands are
    adjacent in space: which layer a point is in, the field's two values at
    an interface and the jump between them, integrals over the whole, one
    function sampled into every layer, and priors given layer by layer, under
    which the layers are independent. It compares equal to the plain direct
    sum of its layers, which is what a block operator or a product measure
    lands on. Layers built by hand go in through :meth:`from_layers`.
    The jump across an interface, a circle, is taken at given angles.
    """

    def __init__(
        self,
        boundaries: Sequence[float],
        mmax: int | list[int],
        /,
        *,
        labels: Sequence[str] | None = None,
        **options: Any,
    ) -> None:
        """
        Args:
            boundaries: the radii bounding the rings, ascending. A first
                radius of zero makes the innermost layer a whole disk.
            mmax: the largest angular order kept in each layer, for all or as
                a list.
            labels: names for the layers, so one can be reached by name.
            **options: the keyword arguments of :class:`Disk` --
                ``order``, ``length_scale``, ``radial_modes``,
                ``max_eigenvalue``, ``padding``, ``boundary``, ``ngll``,
                ``element_length`` -- each given once for all layers or, as a
                list, layer by layer. A tuple, such as a padding pair, is
                one value.

        Raises:
            ValueError: for fewer than two boundaries, or a list of the
                wrong length.
        """
        super().__init__(
            _layered.layers_between(
                Disk, boundaries, mmax, ("inner_radius", "radius"), options
            ),
            labels=labels,
        )

    def jump_operator(
        self,
        interface: int,
        /,
        *,
        directions: Sequence[float] | None = None,
        unsafe: bool = False,
    ) -> LinearOperator:
        """The jump of the field across an interface: its value just outside,
        less its value just inside.

        What a reflection coefficient sees, and what a prior on a
        discontinuity is a statement about. Conditioning on its vanishing is
        how two layers are made continuous at the circle they share.

        Args:
            interface: which interface, by position among :attr:`interfaces`.
            directions: the angles, in degrees anticlockwise from the ``x``
                axis, at which the jump is taken.
            unsafe: build it even in layers too rough to admit it.

        Returns:
            The operator into a Euclidean space of one dimension per angle.

        Raises:
            ValueError: if no angles are given, or a layer's order is too low
                and *unsafe* is not set.
            IndexError: if there is no such interface.
        """
        position = self.interfaces[interface]
        if directions is None:
            raise ValueError("In disks the jump is taken at given directions.")
        points = [(position, float(angle)) for angle in np.ravel(directions)]
        if not points:
            raise ValueError("At least one direction is needed.")
        for side in ("below", "above"):
            self._require_point_evaluation(
                "A jump operator", points=points, side=side, unsafe=unsafe
            )
        matrix = self.basis_matrix(points, side="above") - self.basis_matrix(
            points, side="below"
        )
        # Onto: the jumps at distinct interfaces and directions are
        # independent functionals, so an exact conditioning on them is well
        # posed and says so.
        return LinearOperator.from_matrix(
            self,
            EuclideanSpace(len(points)),
            matrix,
            form="galerkin",
            traits=Traits.SURJECTIVE,
        )
