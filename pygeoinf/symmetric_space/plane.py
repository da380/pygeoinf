from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List, Union

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pygeoinf.linear_operators import LinearOperator
from .symmetric_space import AbstractSymmetricLebesgueSpace, SymmetricSobolevSpace
from .torus import Lebesgue as TorusLebesgue
from .torus import Sobolev as TorusSobolev


class Lebesgue(AbstractSymmetricLebesgueSpace):
    """
    Implementation of the Lebesgue space L² on a compact rectangle in the 2D plane.

    Functions are assumed to have support limited to the chosen rectangular interval
    [ax, bx] x [ay, by]. The coordinate basis for the space is through 2D Fourier
    expansions. Details of the implementation are handled by smoothly embedding
    the function into a larger 2D periodic Torus.
    """

    def __init__(
        self,
        kmax: int,
        /,
        *,
        ax: float = 0.0,
        bx: float = 1.0,
        cx: float = 0.1,
        ay: float = 0.0,
        by: float = 1.0,
        cy: float = 0.1,
    ):
        """
        Args:
            kmax: The maximum Fourier degree to be represented for both dimensions.
            ax: The lower x-coordinate of the interval. Defaults to 0.0.
            bx: The upper x-coordinate of the interval. Defaults to 1.0.
            cx: The padding distance for the x-domain. Defaults to 0.1.
            ay: The lower y-coordinate of the interval. Defaults to 0.0.
            by: The upper y-coordinate of the interval. Defaults to 1.0.
            cy: The padding distance for the y-domain. Defaults to 0.1.
        """
        self._ax, self._bx, self._cx = ax, bx, cx
        self._ay, self._by, self._cy = ay, by, cy

        length_x = bx - ax + 2 * cx
        length_y = by - ay + 2 * cy
        radius_x = length_x / (2 * np.pi)
        radius_y = length_y / (2 * np.pi)

        self._torus_space = TorusLebesgue(kmax, radius_x=radius_x, radius_y=radius_y)

        AbstractSymmetricLebesgueSpace.__init__(
            self, 2, kmax, self._torus_space.dim, False
        )

    # ---------------------------------------------- #
    #                   Properties                   #
    # -----------------------------------------------#

    @property
    def kmax(self) -> int:
        return self._torus_space.kmax

    @property
    def bounds_x(self) -> Tuple[float, float, float]:
        """Returns (ax, bx, cx)."""
        return self._ax, self._bx, self._cx

    @property
    def bounds_y(self) -> Tuple[float, float, float]:
        """Returns (ay, by, cy)."""
        return self._ay, self._by, self._cy

    @property
    def torus_space(self) -> TorusLebesgue:
        """Returns the isomorphic space of functions on the extended Torus."""
        return self._torus_space

    # ---------------------------------------------- #
    #                   Factories                    #
    # -----------------------------------------------#

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        /,
        *,
        ax: float = 0.0,
        bx: float = 1.0,
        cx: float = 0.1,
        ay: float = 0.0,
        by: float = 1.0,
        cy: float = 0.1,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        dummy = cls(max(1, min_degree), ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy)
        optimal_degree = dummy.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )
        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)
        return cls(optimal_degree, ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy)

    @classmethod
    def from_heat_kernel_prior(cls, kernel_scale: float, /, **kwargs) -> Lebesgue:
        return cls.from_covariance(cls.heat_kernel(kernel_scale), **kwargs)

    @classmethod
    def from_sobolev_kernel_prior(
        cls, kernel_order: float, kernel_scale: float, /, **kwargs
    ) -> Lebesgue:
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale), **kwargs
        )

    # ------------------------------------------------------ #
    #                     Public methods                     #
    # ------------------------------------------------------ #

    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns two 1D arrays of the flattened X and Y coordinates."""
        th_x, th_y = self._torus_space.points()

        # Reshape for transformation, then flatten again
        dim_side = 2 * self.kmax
        th_x_2d = th_x.reshape((dim_side, dim_side))
        th_y_2d = th_y.reshape((dim_side, dim_side))

        # The torus meshgrid is typically (th_x_1d[:, None], th_y_1d[None, :])
        # Transform unique 1D axes to save computation
        unique_th_x = th_x_2d[:, 0]
        unique_th_y = th_y_2d[0, :]

        x_1d = [self.angle_to_point_x(th) for th in unique_th_x]
        y_1d = [self.angle_to_point_y(th) for th in unique_th_y]

        X, Y = np.meshgrid(x_1d, y_1d, indexing="ij")
        return X.flatten(), Y.flatten()

    def project_function(self, f: Callable[[Tuple[float, float]], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given 2D function.

        Applies a separable 2D raised-cosine window to smoothly taper the
        function to zero within the padding regions.
        """
        X_flat, Y_flat = self.points()
        Z_flat = np.fromiter((f((x, y)) for x, y in zip(X_flat, Y_flat)), float)
        Z = Z_flat.reshape((2 * self.kmax, 2 * self.kmax))

        # Reconstruct 1D axes for efficient mask building
        x = X_flat.reshape((2 * self.kmax, 2 * self.kmax))[:, 0]
        y = Y_flat.reshape((2 * self.kmax, 2 * self.kmax))[0, :]

        # Build X mask
        mask_x = np.ones_like(x)
        left_idx_x = (x < self._ax) & (x >= self._ax - self._cx)
        if np.any(left_idx_x):
            x_norm = (x[left_idx_x] - (self._ax - self._cx)) / self._cx
            mask_x[left_idx_x] = 0.5 * (1 - np.cos(np.pi * x_norm))

        right_idx_x = (x > self._bx) & (x <= self._bx + self._cx)
        if np.any(right_idx_x):
            x_norm = (x[right_idx_x] - self._bx) / self._cx
            mask_x[right_idx_x] = 0.5 * (1 + np.cos(np.pi * x_norm))

        mask_x[(x < self._ax - self._cx) | (x > self._bx + self._cx)] = 0.0

        # Build Y mask
        mask_y = np.ones_like(y)
        bottom_idx_y = (y < self._ay) & (y >= self._ay - self._cy)
        if np.any(bottom_idx_y):
            y_norm = (y[bottom_idx_y] - (self._ay - self._cy)) / self._cy
            mask_y[bottom_idx_y] = 0.5 * (1 - np.cos(np.pi * y_norm))

        top_idx_y = (y > self._by) & (y <= self._by + self._cy)
        if np.any(top_idx_y):
            y_norm = (y[top_idx_y] - self._by) / self._cy
            mask_y[top_idx_y] = 0.5 * (1 + np.cos(np.pi * y_norm))

        mask_y[(y < self._ay - self._cy) | (y > self._by + self._cy)] = 0.0

        # Combine separable masks
        mask_2d = mask_x[:, np.newaxis] * mask_y[np.newaxis, :]
        return Z * mask_2d

    def point_to_angle(self, p: Tuple[float, float]) -> Tuple[float, float]:
        x, y = p
        th_x = (x - self._ax + self._cx) / self._torus_space.radius_x
        th_y = (y - self._ay + self._cy) / self._torus_space.radius_y
        return th_x, th_y

    def angle_to_point(self, th: Tuple[float, float]) -> Tuple[float, float]:
        th_x, th_y = th
        x = self.angle_to_point_x(th_x)
        y = self.angle_to_point_y(th_y)
        return x, y

    def angle_to_point_x(self, th_x: float) -> float:
        return self._ax - self._cx + self._torus_space.radius_x * th_x

    def angle_to_point_y(self, th_y: float) -> float:
        return self._ay - self._cy + self._torus_space.radius_y * th_y

    # ------------------------------------------------------ #
    #           Delegations to SymmetricHilbertSpace         #
    # ------------------------------------------------------ #

    def to_coefficients(self, u: np.ndarray) -> np.ndarray:
        return self._torus_space.to_coefficients(u)

    def from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        return self._torus_space.from_coefficients(coeff)

    def to_coefficient_operator(self, kmax: int) -> LinearOperator:
        torus_op = self._torus_space.to_coefficient_operator(kmax)
        return LinearOperator(
            self,
            torus_op.codomain,
            lambda u: torus_op(u),
            adjoint_mapping=lambda c: torus_op.adjoint(c),
        )

    def from_coefficient_operator(self, kmax: int) -> LinearOperator:
        torus_op = self._torus_space.from_coefficient_operator(kmax)
        return LinearOperator(
            torus_op.domain,
            self,
            lambda c: torus_op(c),
            adjoint_mapping=lambda u: torus_op.adjoint(u),
        )

    def wavevector_indices(self, kx: int, ky: int) -> List[int]:
        return self._torus_space.wavevector_indices(kx, ky)

    def spectral_projection_operator(
        self, modes: List[Tuple[int, int]]
    ) -> LinearOperator:
        torus_op = self._torus_space.spectral_projection_operator(modes)
        return LinearOperator(
            self,
            torus_op.codomain,
            lambda u: torus_op(u),
            adjoint_mapping=lambda c: torus_op.adjoint(c),
        )

    def integer_to_index(self, i: int) -> int:
        return self._torus_space.integer_to_index(i)

    def index_to_integer(self, k: int) -> int:
        return self._torus_space.index_to_integer(k)

    def laplacian_eigenvalue(self, k: int) -> float:
        return self._torus_space.laplacian_eigenvalue(k)

    def laplacian_eigenvector_squared_norm(self, k: int) -> float:
        return self._torus_space.laplacian_eigenvector_squared_norm(k)

    def laplacian_eigenvectors_at_point(self, p: Tuple[float, float]) -> np.ndarray:
        th = self.point_to_angle(p)
        return self._torus_space.laplacian_eigenvectors_at_point(th)

    def random_point(self) -> Tuple[float, float]:
        """Returns a random point strictly within the unpadded bounds."""
        return (
            np.random.uniform(self._ax, self._bx),
            np.random.uniform(self._ay, self._by),
        )

    def geodesic_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """Euclidean distance on the plane (ignores Torus wrapping)."""
        return float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))

    def geodesic_quadrature(
        self, p1: Tuple[float, float], p2: Tuple[float, float], n_points: int
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """Standard straight-line Euclidean quadrature."""
        arc_length = self.geodesic_distance(p1, p2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        t, w = np.polynomial.legendre.leggauss(n_points)
        t_mapped = (t + 1) / 2.0

        x_pts = p1[0] + t_mapped * dx
        y_pts = p1[1] + t_mapped * dy

        scaled_weights = w * (arc_length / 2.0)
        points = list(zip(x_pts, y_pts))

        return points, scaled_weights

    def with_degree(self, degree: int) -> Lebesgue:
        return Lebesgue(
            degree,
            ax=self._ax,
            bx=self._bx,
            cx=self._cx,
            ay=self._ay,
            by=self._by,
            cy=self._cy,
        )

    def degree_transfer_operator(self, target_degree: int) -> LinearOperator:
        codomain = self.with_degree(target_degree)
        torus_transfer = self._torus_space.degree_transfer_operator(target_degree)

        return LinearOperator(
            self,
            codomain,
            lambda u: torus_transfer(u),
            adjoint_mapping=lambda v: torus_transfer.adjoint(v),
        )

    def invariant_covariance_function(
        self, spectral_variances: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        return self._torus_space.invariant_covariance_function(spectral_variances)

    def estimate_truncation_degree(self, *args, **kwargs) -> int:
        return self._torus_space.estimate_truncation_degree(*args, **kwargs)

    def degree_multiplicity(self, degree: int) -> int:
        return self._torus_space.degree_multiplicity(degree)

    def representative_index(self, degree: int) -> int:
        return self._torus_space.representative_index(degree)

    # ------------------------------------------------------ #
    #                 Methods for HilbertSpace               #
    # ------------------------------------------------------ #

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return self._torus_space.to_components(x)

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return self._torus_space.from_components(c)

    def is_element(self, x: Any) -> bool:
        return self._torus_space.is_element(x)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lebesgue):
            return NotImplemented
        return (
            self.kmax == other.kmax
            and self.bounds_x == other.bounds_x
            and self.bounds_y == other.bounds_y
        )

    # ------------------------------------------------------ #
    #                 Methods for HilbertModule              #
    # ------------------------------------------------------ #

    def vector_multiply(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return self._torus_space.vector_multiply(x1, x2)

    def vector_sqrt(self, x: np.ndarray) -> np.ndarray:
        return self._torus_space.vector_sqrt(x)


class Sobolev(SymmetricSobolevSpace):
    """Implementation of the Sobolev space Hˢ on a compact 2D plane."""

    def __init__(
        self,
        kmax: int,
        order: float,
        scale: float,
        /,
        *,
        ax: float = 0.0,
        bx: float = 1.0,
        cx: Optional[float] = None,
        ay: float = 0.0,
        by: float = 1.0,
        cy: Optional[float] = None,
        safe: bool = True,
    ):
        cx = 6 * scale if cx is None else cx
        cy = 6 * scale if cy is None else cy

        lebesgue_space = Lebesgue(kmax, ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy)
        SymmetricSobolevSpace.__init__(self, lebesgue_space, order, scale, safe=safe)

    @staticmethod
    def from_sobolev_parameters(order: float, scale: float, /, **kwargs) -> Sobolev:
        safe = kwargs.get("safe", True)
        if safe and order <= 1.0:
            raise ValueError(
                "Point evaluation on a 2D Plane requires Sobolev order > 1.0"
            )

        dummy_kwargs = kwargs.copy()
        dummy_kwargs["safe"] = False
        dummy = Sobolev(1, order, scale, **dummy_kwargs)

        kmax = dummy.estimate_truncation_degree(
            dummy.sobolev_function, rtol=kwargs.get("rtol", 1e-6)
        )

        if kwargs.get("power_of_two", False):
            n = int(np.log2(kmax))
            kmax = 2 ** (n + 1)

        return Sobolev(kmax, order, scale, **kwargs)

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        order: float,
        scale: float,
        /,
        **kwargs,
    ) -> Sobolev:
        safe = kwargs.pop("safe", True)
        min_deg = kwargs.pop("min_degree", 1)
        max_deg = kwargs.pop("max_degree", None)
        rtol = kwargs.pop("rtol", 1e-6)
        p2 = kwargs.pop("power_of_two", False)

        dummy = cls(max(1, min_deg), order, scale, safe=False, **kwargs)
        optimal_degree = dummy.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_deg, max_degree=max_deg
        )

        if p2:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(optimal_degree, order, scale, safe=safe, **kwargs)

    @classmethod
    def from_heat_kernel_prior(
        cls, kernel_scale: float, order: float, scale: float, /, **kwargs
    ) -> Sobolev:
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale), order, scale, **kwargs
        )

    @classmethod
    def from_sobolev_kernel_prior(
        cls,
        kernel_order: float,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        **kwargs,
    ) -> Sobolev:
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale), order, scale, **kwargs
        )

    # ---------------------------------------------- #
    #                   Properties                   #
    # -----------------------------------------------#

    @property
    def kmax(self) -> int:
        return self.underlying_space.kmax

    @property
    def bounds_x(self) -> Tuple[float, float, float]:
        return self.underlying_space.bounds_x

    @property
    def bounds_y(self) -> Tuple[float, float, float]:
        return self.underlying_space.bounds_y

    @property
    def torus_space(self) -> TorusSobolev:
        return TorusSobolev(
            self.kmax,
            self.order,
            self.scale,
            radius_x=self.underlying_space.torus_space.radius_x,
            radius_y=self.underlying_space.torus_space.radius_y,
        )

    # ---------------------------------------------- #
    #                 Delegations                    #
    # -----------------------------------------------#

    def with_order(self, order: float) -> Sobolev:
        ax, bx, cx = self.bounds_x
        ay, by, cy = self.bounds_y
        return Sobolev(
            self.kmax, order, self.scale, ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy
        )

    def with_degree(self, degree: int) -> Sobolev:
        ax, bx, cx = self.bounds_x
        ay, by, cy = self.bounds_y
        return Sobolev(
            degree, self.order, self.scale, ax=ax, bx=bx, cx=cx, ay=ay, by=by, cy=cy
        )

    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.underlying_space.points()

    def project_function(self, f: Callable[[Tuple[float, float]], float]) -> np.ndarray:
        return self.underlying_space.project_function(f)

    def to_coefficient_operator(self, kmax: int) -> LinearOperator:
        l2_op = self.underlying_space.to_coefficient_operator(kmax)
        return LinearOperator.from_formal_adjoint(self, l2_op.codomain, l2_op)

    def from_coefficient_operator(self, kmax: int) -> LinearOperator:
        l2_op = self.underlying_space.from_coefficient_operator(kmax)
        return LinearOperator.from_formal_adjoint(l2_op.domain, self, l2_op)

    def spectral_projection_operator(
        self, modes: List[Tuple[int, int]]
    ) -> LinearOperator:
        l2_op = self.underlying_space.spectral_projection_operator(modes)
        return LinearOperator.from_formal_adjoint(self, l2_op.codomain, l2_op)

    def estimate_truncation_degree(
        self,
        covariance_function: Callable[[float], float],
        /,
        *,
        rtol: float = 1e-6,
        min_degree: int = 1,
        max_degree: Optional[int] = None,
    ) -> int:
        """
        Delegates the energy truncation search to the underlying Plane Lebesgue space,
        ensuring it loops geometrically over (kx, ky) shells rather than flat 1D indices.
        """

        def sobolev_weighted_cov(eval_val: float) -> float:
            return covariance_function(eval_val) * self.sobolev_function(eval_val)

        return self.underlying_space.estimate_truncation_degree(
            sobolev_weighted_cov,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
        )


# ------------------------------------------------- #
#           Associated plotting functions           #
# ------------------------------------------------- #


def plot(
    space: Union[Lebesgue, Sobolev],
    u: np.ndarray,
    /,
    *,
    ax: Optional[Axes] = None,
    contour: bool = False,
    cmap: str = "RdBu",
    symmetric: bool = False,
    contour_lines: bool = False,
    contour_lines_kwargs: Optional[dict] = None,
    num_levels: int = 10,
    colorbar: bool = False,
    colorbar_kwargs: Optional[dict] = None,
    full: bool = False,
    **kwargs,
) -> Tuple[Axes, Any]:
    """
    Creates a high-quality map plot of a function on the 2D Plane.

    By default, this crops the view to the primary computational domain
    [ax, bx] x [ay, by]. Set `full=True` to view the tapered padding regions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    elif isinstance(ax, tuple):
        ax = ax[0]
        fig = ax.get_figure()
    else:
        fig = ax.get_figure()

    X_flat, Y_flat = space.points()
    x_1d = X_flat.reshape((2 * space.kmax, 2 * space.kmax))[:, 0]
    y_1d = Y_flat.reshape((2 * space.kmax, 2 * space.kmax))[0, :]

    kwargs.setdefault("cmap", cmap)
    if symmetric:
        data_max = 1.2 * np.nanmax(np.abs(u))
        kwargs.setdefault("vmin", -data_max)
        kwargs.setdefault("vmax", data_max)

    if "levels" in kwargs:
        levels = kwargs.pop("levels")
    else:
        vmin = kwargs.get("vmin", np.nanmin(u))
        vmax = kwargs.get("vmax", np.nanmax(u))
        levels = np.linspace(vmin, vmax, num_levels)

    u_plot = u.T

    im: Any
    if contour:
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)
        im = ax.contourf(x_1d, y_1d, u_plot, levels=levels, **kwargs)
    else:
        kwargs.setdefault("shading", "auto")
        im = ax.pcolormesh(x_1d, y_1d, u_plot, **kwargs)

    if contour_lines:
        cl_kwargs = contour_lines_kwargs if contour_lines_kwargs is not None else {}
        cl_kwargs.setdefault("colors", "k")
        cl_kwargs.setdefault("linewidths", 0.5)
        ax.contour(x_1d, y_1d, u_plot, levels=levels, **cl_kwargs)

    if colorbar and fig:
        cb_opts = colorbar_kwargs or {}
        cb_opts.setdefault("orientation", "horizontal")
        cb_opts.setdefault("shrink", 0.7)
        cb_opts.setdefault("pad", 0.05)
        fig.colorbar(im, ax=ax, **cb_opts)

    if not full:
        ax.set_xlim(space.bounds_x[0], space.bounds_x[1])
        ax.set_ylim(space.bounds_y[0], space.bounds_y[1])
    else:
        # Show the entire padded computational domain
        ax.set_xlim(
            space.bounds_x[0] - space.bounds_x[2], space.bounds_x[1] + space.bounds_x[2]
        )
        ax.set_ylim(
            space.bounds_y[0] - space.bounds_y[2], space.bounds_y[1] + space.bounds_y[2]
        )

    ax.set_aspect("equal", adjustable="box")

    return ax, im


def plot_geodesic_network(
    paths: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    /,
    *,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """
    Plots a network of intersecting straight-line paths on the 2D Plane.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    elif isinstance(ax, tuple):
        ax = ax[0]

    line_kwargs = kwargs.copy()
    src_style = line_kwargs.pop("source_kwargs", {})
    rec_style = line_kwargs.pop("receiver_kwargs", {})

    line_kwargs.setdefault("color", "black")
    line_kwargs.setdefault("linewidth", 0.8)
    line_kwargs.setdefault("alpha", 0.5)

    for p1, p2 in paths:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **line_kwargs)

    sources = list(set([tuple(p[0]) for p in paths]))
    receivers = list(set([tuple(p[1]) for p in paths]))

    if sources:
        src_x, src_y = zip(*sources)
        src_style.setdefault("marker", "*")
        src_style.setdefault("color", "gold")
        src_style.setdefault("s", 150)
        src_style.setdefault("edgecolor", "black")
        src_style.setdefault("zorder", 5)
        ax.scatter(src_x, src_y, **src_style)

    if receivers:
        rec_x, rec_y = zip(*receivers)
        rec_style.setdefault("marker", "o")
        rec_style.setdefault("color", "red")
        rec_style.setdefault("s", 50)
        rec_style.setdefault("edgecolor", "white")
        rec_style.setdefault("zorder", 5)
        ax.scatter(rec_x, rec_y, **rec_style)

    return ax
