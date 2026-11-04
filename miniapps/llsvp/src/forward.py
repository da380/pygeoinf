"""The forward map: the anomalies on the shell, and what the data see of them.

The data seen by the model are, for a thermal triple ``t``, a level set ``phi``
and contrasts ``c``,

.. code-block:: text

    d == T t + G(phi) c

with ``T`` linear, the path and cap averages of ``vs`` and ``vp`` and the
modes' weighted spherical-harmonic coefficients of all three, and ``G(phi)``
the same functionals of the indicator of ``{phi > 0}``, one column per
contrast. So the map is linear in ``t`` and ``c`` and nonlinear only in
``phi``.

The indicator is sharp. It is evaluated where the data need it, at the
quadrature nodes of the paths and the caps, and on a fine grid of one degree
at every truncation for the modes' coefficients, never as a band-limited
field, which would ring, and never on the coarse grid, whose rows a sharp edge
falls between.

Its derivative in ``phi`` is zero wherever it exists, which is everywhere but on
the provinces' edges, so the forward map's derivative is taken to be

.. code-block:: text

    F'(t, phi, c)[dt, dphi, dc] == T dt + G(phi) dc

That is what the gradient-based blocks want, on ``t`` and ``c``, and they never
ask for the rest; the level set's block is pCN, which needs no gradient. What
this derivative is not is the derivative of the map as a whole, so a sampler that
follows it across the level set would be blind to the provinces' edges.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pygeoinf import LinearOperator, Operator
from pygeoinf.algebra.direct_sum import BlockDiagonalLinearOperator, DirectSum
from pygeoinf.algebra.spaces import EuclideanSpace
from pygeoinf.symmetric_space.sphere import Sobolev

from .shell import FIELDS, REGIONS, Network, Scale


def record_operator(scale: Scale, space: Sobolev, /) -> LinearOperator:
    """The means of the three thermal fields over the caps of :data:`REGIONS`:
    what a chain records, into a space of one coordinate per field and cap.

    Args:
        scale: the caps' radius.
        space: the sphere of one field.

    Returns:
        The operator on the direct sum of the three fields.
    """
    caps = space.geodesic_ball_average_operator(
        [np.array(center) for _, center in REGIONS], scale.region_radius, dense=True
    )
    return BlockDiagonalLinearOperator([caps] * len(FIELDS))


def pooled(quadratures: list, /) -> tuple[np.ndarray, sparse.csr_matrix]:
    """The pooled nodes of several quadratures and the weights of each,
    normalized to averages: one row per quadrature."""
    nodes = np.vstack([np.asarray(points) for points, _ in quadratures])
    data, columns, pointers = [], [], [0]
    start = 0
    for points, weights in quadratures:
        weights = np.asarray(weights)
        data.append(weights / weights.sum())
        columns.append(np.arange(start, start + weights.size))
        start += weights.size
        pointers.append(start)
    weights = sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(columns), pointers),
        shape=(len(quadratures), start),
    )
    return nodes, weights


class Forward:
    """The forward map of a network on a sphere.

    The data are stacked as the S arcs, the S caps, the P arcs, the P caps and
    the modes' coefficients, mode after mode and degree after degree. The
    slices of the kinds are :attr:`slices`.

    Attributes:
        space: the sphere of one field.
        thermal_space: the direct sum of the three thermal fields.
        data_space: the Euclidean space of the data.
        thermal: ``T``, the linear map from the thermal fields to the data.
        network: where the data are.
        degrees: the degrees of the modes' coefficients.
        labels: the degree and order of each mode's coefficients.
        slices: the slice of the data of each kind.
    """

    def __init__(
        self,
        space: Sobolev,
        network: Network,
        scale: Scale,
        /,
        *,
        n_jobs: int | None = None,
    ) -> None:
        """
        Args:
            space: the sphere of one field.
            network: where the data are.
            scale: the quadratures and the modes' degrees.
            n_jobs: workers for assembling the path and cap matrices.
        """
        self.space = space
        self.thermal_space = DirectSum([space] * len(FIELDS), labels=FIELDS)
        self._scale = scale
        self.network = network
        self.degrees = tuple(d for d in scale.degrees if d <= space.lmax)
        sizes = {
            "s_arcs": len(network.s_arcs),
            "s_caps": len(network.s_caps),
            "p_arcs": len(network.p_arcs),
            "p_caps": len(network.p_caps),
            "modes": network.kernels.shape[0] * sum(2 * d + 1 for d in self.degrees),
        }
        self.slices: dict[str, slice] = {}
        start = 0
        for kind, size in sizes.items():
            self.slices[kind] = slice(start, start + size)
            start += size
        self.data_space = EuclideanSpace(start)
        labels = [(d, m) for d in self.degrees for m in range(-d, d + 1)]
        self.labels = labels
        self._positions = np.array([space.component_of(d, m) for d, m in labels])
        self._fine = space.with_degree(max(scale.indicator_degree, space.lmax))
        self._lift = space.degree_transfer_operator(self._fine)
        self._fine_positions = np.array(
            [self._fine.component_of(d, m) for d, m in labels]
        )
        self._assemble(n_jobs)
        self._prepare_indicator()
        self._response_key: bytes | None = None
        self._response: np.ndarray | None = None

    def _rows(self, arcs: tuple, caps: np.ndarray, n_jobs: int | None, /) -> np.ndarray:
        """The rows of one wave's arcs and caps, on one field's components."""
        rows = []
        if arcs:
            rows.append(
                self.space.path_average_operator(
                    arcs, count=self._scale.arc_nodes, dense=True, n_jobs=n_jobs
                ).matrix(form="components")
            )
        if len(caps):
            rows.append(
                self.space.geodesic_ball_average_operator(
                    list(caps), self._scale.cap_radius, dense=True, n_jobs=n_jobs
                ).matrix(form="components")
            )
        return np.vstack(rows) if rows else np.empty((0, self.space.dim))

    def _assemble(self, n_jobs: int | None, /) -> None:
        """The dense matrix of ``T`` on the fields' components, by field."""
        network, dim = self.network, self.space.dim
        s_rows = self._rows(network.s_arcs, network.s_caps, n_jobs)
        p_rows = self._rows(network.p_arcs, network.p_caps, n_jobs)
        n_modes = network.kernels.shape[0]
        count = len(self._positions)
        picks = np.zeros((count, dim))
        picks[np.arange(count), self._positions] = 1.0
        blocks = []
        n_s, n_p = s_rows.shape[0], p_rows.shape[0]
        for k in range(len(FIELDS)):
            rows = [
                s_rows if k == 0 else np.zeros((n_s, dim)),
                p_rows if k == 1 else np.zeros((n_p, dim)),
                np.vstack(
                    [network.kernels[mode, k] * picks for mode in range(n_modes)]
                ),
            ]
            blocks.append(np.vstack(rows))
        self.matrix = np.hstack(blocks)
        self.thermal = LinearOperator.from_matrix(
            self.thermal_space, self.data_space, self.matrix, form="components"
        )

    def _prepare_indicator(self) -> None:
        """The quadratures the indicator is integrated by."""
        space, scale, network = self.space, self._scale, self.network
        arcs = [
            space.geodesic_quadrature(start, end, count=scale.arc_nodes)
            for start, end in (*network.s_arcs, *network.p_arcs)
        ]
        caps = [
            space.geodesic_ball_quadrature(
                center, scale.cap_radius, count=scale.cap_nodes
            )
            for center in (*network.s_caps, *network.p_caps)
        ]
        self._arc_nodes, self._arc_weights = pooled(arcs) if arcs else (None, None)
        self._cap_nodes, self._cap_weights = pooled(caps) if caps else (None, None)
        self._arc_points = (
            None if arcs == [] else space.prepare_points(list(self._arc_nodes))
        )
        self._cap_points = (
            None if caps == [] else space.prepare_points(list(self._cap_nodes))
        )

    def indicator_averages(self, phi: Any, /) -> tuple[np.ndarray, np.ndarray]:
        """The share of each arc, and of each cap, inside ``{phi > 0}``.

        Args:
            phi: the level set.

        Returns:
            One array per arc, and one per cap, S before P in each.
        """
        arcs = np.empty(0)
        caps = np.empty(0)
        if self._arc_points is not None:
            inside = self.space.evaluate(phi, self._arc_points) > 0.0
            arcs = self._arc_weights @ inside.astype(float)
        if self._cap_points is not None:
            inside = self.space.evaluate(phi, self._cap_points) > 0.0
            caps = self._cap_weights @ inside.astype(float)
        return arcs, caps

    def indicator_coefficients(self, phi: Any, /) -> np.ndarray:
        """The spherical-harmonic coefficients of ``{phi > 0}`` the modes see.

        Args:
            phi: the level set.

        Returns:
            One entry for each degree and order of :attr:`labels`.
        """
        fine = self._fine
        inside = (fine.grid_values(self._lift(phi)) > 0.0).astype(float)
        return fine.to_components(fine.from_grid_values(inside))[self._fine_positions]

    def response(self, phi: Any, /) -> np.ndarray:
        """``G(phi)``: the data of a unit contrast in each field inside
        ``{phi > 0}``, one column per field.

        Args:
            phi: the level set.

        Returns:
            An array of one row per datum and one column per contrast. The
            last level set's is kept, so that the blocks that hold it fixed
            pay for it once.
        """
        key = self.space.grid_values(phi).tobytes()
        if key == self._response_key:
            return self._response
        network = self.network
        arcs, caps = self.indicator_averages(phi)
        n_s_arcs, n_s_caps = len(network.s_arcs), len(network.s_caps)
        coefficients = self.indicator_coefficients(phi)
        response = np.zeros((self.data_space.dim, len(FIELDS)))
        response[self.slices["s_arcs"], 0] = arcs[:n_s_arcs]
        response[self.slices["s_caps"], 0] = caps[:n_s_caps]
        response[self.slices["p_arcs"], 1] = arcs[n_s_arcs:]
        response[self.slices["p_caps"], 1] = caps[n_s_caps:]
        response[self.slices["modes"], :] = np.kron(
            network.kernels, coefficients[:, None]
        )
        self._response_key, self._response = key, response
        return response

    def __call__(self, model: Any, /) -> np.ndarray:
        """The data of a model of the whole space.

        Args:
            model: the triple ``(thermal, level set, contrasts)``.

        Returns:
            The data, ``T t + G(phi) c``.
        """
        thermal, phi, contrasts = model
        return self.thermal(thermal) + self.response(phi) @ np.asarray(contrasts)

    def derivative(self, domain: Any, model: Any, /) -> LinearOperator:
        """``T dt + G(phi) dc``: the map's derivative in the thermal fields and
        the contrasts, and zero in the level set.

        Args:
            domain: the whole model space.
            model: where to linearize, of which only the level set matters.

        Returns:
            The linear operator on the whole model space.
        """
        gain = self.response(model[1])

        def value(direction: Any) -> np.ndarray:
            return self.thermal(direction[0]) + gain @ np.asarray(direction[2])

        def adjoint(residual: np.ndarray) -> tuple:
            return (
                self.thermal.adjoint(residual),
                self.space.zero(),
                gain.T @ residual,
            )

        return LinearOperator.from_callables(
            domain, self.data_space, value, adjoint=adjoint
        )

    def operator(self, domain: Any, /) -> Operator:
        """The forward map as a library operator, with the derivative above.

        Args:
            domain: the whole model space.

        Returns:
            The operator.
        """
        return Operator.from_callables(
            domain,
            self.data_space,
            self,
            derivative=lambda model: self.derivative(domain, model),
        )

    def grouped_operator(self, domain: Any, /) -> Operator:
        """The same map on the grouped sum, ``((thermal, contrasts), level set)``.

        Args:
            domain: the grouped model space.

        Returns:
            The operator, with the derivative's level-set part zero as before.
        """

        def value(model: Any) -> np.ndarray:
            return self((model[0][0], model[1], model[0][1]))

        def derivative(model: Any) -> LinearOperator:
            gain = self.response(model[1])

            def action(direction: Any) -> np.ndarray:
                linear = direction[0]
                return self.thermal(linear[0]) + gain @ np.asarray(linear[1])

            def adjoint(residual: np.ndarray) -> tuple:
                linear = (self.thermal.adjoint(residual), gain.T @ residual)
                return linear, self.space.zero()

            return LinearOperator.from_callables(
                domain, self.data_space, action, adjoint=adjoint
            )

        return Operator.from_callables(
            domain, self.data_space, value, derivative=derivative
        )
