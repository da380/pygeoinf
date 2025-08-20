"""
Module for Sobolev spaces on the two-sphere.
"""

from __future__ import annotations
from typing import Callable, Any, List, Optional, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np

from scipy.sparse import diags, coo_array

import pyshtools as sh

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.operators import LinearOperator
from pygeoinf.symmetric_space.symmetric_space import SymmetricSpaceSobolev
from pygeoinf.gaussian_measure import GaussianMeasure

# This block only runs for type checkers, not at runtime
if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy.crs import Projection
    from pygeoinf.forms import LinearForm


class Sobolev(SymmetricSpaceSobolev):
    """
    Implements Sobolev spaces H^s on a two-sphere.

    This class uses `pyshtools` for spherical harmonic transforms. Vectors in the
    space can be represented either as `pyshtools.SHGrid` objects (spatial
    domain) or `pyshtools.SHCoeffs` objects (spectral domain), controlled by a flag.
    """

    def __init__(
        self,
        lmax: int,
        order: float,
        scale: float,
        /,
        *,
        vector_as_SHGrid: bool = True,
        radius: float = 1.0,
        grid: str = "DH",
    ) -> None:
        """
        Args:
            lmax: The maximum spherical harmonic degree for truncation.
            order: The order of the Sobolev space, controlling smoothness.
            scale: The non-dimensional length-scale for the space.
            vector_as_SHGrid: If True (default), elements of the space are
                `pyshtools.SHGrid` objects. Otherwise, they are `pyshtools.SHCoeffs`.
            radius: The radius of the sphere. Defaults to 1.0.
            grid: The `pyshtools` grid type (e.g., 'DH', 'DH2', 'GLQ').
                Defaults to 'DH'.
        """

        self._lmax: int = lmax
        self._radius: float = radius
        self._grid: str = grid
        self._sampling: int = 2 if self.grid == "DH2" else 1
        self._extend: bool = True
        self._normalization: str = "ortho"
        self._csphase: int = 1
        self._sparse_coeffs_to_component: coo_array = (
            self._coefficient_to_component_mapping()
        )

        dim: int = (lmax + 1) ** 2

        self._vector_as_SHGrid: bool = vector_as_SHGrid
        if vector_as_SHGrid:
            super().__init__(
                order,
                scale,
                dim,
                self._to_components_from_SHGrid,
                self._from_components_to_SHGrid,
                self._inner_product_impl,
                self._to_dual_impl,
                self._from_dual_impl,
                ax=self._ax_impl,
                axpy=self._axpy_impl,
                vector_multiply=self._vector_multiply_impl,
            )
        else:
            super().__init__(
                order,
                scale,
                dim,
                self._to_components_from_SHCoeffs,
                self._from_components_to_SHCoeffs,
                self._inner_product_impl,
                self._to_dual_impl,
                self._from_dual_impl,
                ax=self._ax_impl,
                axpy=self._axpy_impl,
                vector_multiply=self._vector_multiply_impl,
            )

        self._metric_tensor: diags = self._degree_dependent_scaling_to_diagonal_matrix(
            self._sobolev_function
        )
        self._inverse_metric_tensor: diags = (
            self._degree_dependent_scaling_to_diagonal_matrix(
                lambda l: 1.0 / self._sobolev_function(l)
            )
        )

    # ===============================================#
    #                  Static methods                #
    # ===============================================#

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
        vector_as_SHGrid: bool = True,
        grid: str = "DH",
        rtol: float = 1e-8,
        power_of_two: bool = False,
    ) -> "Sobolev":
        """
        Creates an instance with `lmax` chosen based on the Sobolev parameters.

        This factory method estimates the spherical harmonic truncation degree
        (`lmax`) required to represent the space while meeting a specified
        relative tolerance for the truncation error. This is useful when the
        required `lmax` is not known a priori.

        Args:
            order: The order of the Sobolev space, controlling smoothness.
            scale: The non-dimensional length-scale for the space.
            radius: The radius of the sphere. Defaults to 1.0.
            vector_as_SHGrid: If True (default), elements are `SHGrid` objects.
            grid: The `pyshtools` grid type (e.g., 'DH'). Defaults to 'DH'.
            rtol: The relative tolerance used to determine the `lmax`.
            power_of_two: If True, `lmax` is set to the next power of two.

        Returns:
            An instance of the Sobolev class with a calculated `lmax`.
        """
        if order <= 1.0:
            raise ValueError("This method is only applicable for orders > 1.0")

        summation = 1.0
        l = 0
        err = 1.0
        sobolev_func = lambda deg: (1.0 + scale**2 * deg * (deg + 1)) ** order

        while err > rtol:
            l += 1
            term = (2 * l + 1) / sobolev_func(l)
            summation += term
            err = term / summation
            print(l, err)
            if l > 10000:
                raise RuntimeError("Failed to converge on a stable lmax.")

        if power_of_two:
            n = int(np.log2(l))
            l = 2 ** (n + 1)

        lmax = l
        return Sobolev(
            lmax,
            order,
            scale,
            vector_as_SHGrid=vector_as_SHGrid,
            radius=radius,
            grid=grid,
        )

    # ===============================================#
    #                   Properties                  #
    # ===============================================#

    @property
    def lmax(self) -> int:
        """The maximum spherical harmonic truncation degree."""
        return self._lmax

    @property
    def radius(self) -> float:
        """The radius of the sphere."""
        return self._radius

    @property
    def grid(self) -> str:
        """The `pyshtools` grid type used for spatial representations."""
        return self._grid

    @property
    def extend(self) -> bool:
        """True if the spatial grid includes both 0 and 360-degree longitudes."""
        return self._extend

    @property
    def normalization(self) -> str:
        """The spherical harmonic normalization convention used ('ortho')."""
        return self._normalization

    @property
    def csphase(self) -> int:
        """The Condon-Shortley phase convention used (1)."""
        return self._csphase

    # ==============================================#
    #                 Public methods                #
    # ==============================================#

    def project_function(self, f: Callable[[(float, float)], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        Args:
            f: A function that takes a point `(lat, lon)` and returns a value.
        """
        u = self.zero
        for j, lon in enumerate(u.lons()):
            for i, lat in enumerate(u.lats()):
                u.data[i, j] = f((lat, lon))
        return u

    def random_point(self) -> List[float]:
        """Returns a random point as `[latitude, longitude]`."""
        latitude = np.random.uniform(-90.0, 90.0)
        longitude = np.random.uniform(0.0, 360.0)
        return [latitude, longitude]

    def low_degree_projection(
        self,
        truncation_degree: int,
        /,
        *,
        smoother: Optional[Callable[[int], float]] = None,
    ) -> "LinearOperator":
        """
        Returns an operator that projects onto a lower-degree space.

        This can be used for truncating or smoothing a field in the spherical
        harmonic domain.

        Args:
            truncation_degree: The new maximum degree `lmax`.
            smoother: An optional callable `f(l)` that applies a degree-dependent
                weighting factor during projection.

        Returns:
            A `LinearOperator` that maps from this space to a new, lower-degree
            `Sobolev` space.
        """
        truncation_degree = (
            truncation_degree if truncation_degree <= self.lmax else self.lmax
        )
        # Default smoother is an identity mapping.
        f: Callable[[int], float] = smoother if smoother is not None else lambda l: 1.0

        row_dim = (truncation_degree + 1) ** 2
        col_dim = (self.lmax + 1) ** 2

        # Construct the sparse matrix that performs the coordinate projection.
        row, col = 0, 0
        rows, cols, data = [], [], []
        for l in range(self.lmax + 1):
            fac = f(l)
            for _ in range(l + 1):
                if l <= truncation_degree:
                    rows.append(row)
                    row += 1
                    cols.append(col)
                    data.append(fac)
                col += 1

        for l in range(truncation_degree + 1):
            fac = f(l)
            for _ in range(1, l + 1):
                rows.append(row)
                row += 1
                cols.append(col)
                data.append(fac)
                col += 1

        smat = coo_array(
            (data, (rows, cols)), shape=(row_dim, col_dim), dtype=float
        ).tocsc()

        codomain = Sobolev(
            truncation_degree,
            self.order,
            self.scale,
            vector_as_SHGrid=self._vector_as_SHGrid,
            radius=self.radius,
            grid=self._grid,
        )

        def mapping(u: Any) -> Any:
            uc = self.to_components(u)
            vc = smat @ uc
            return codomain.from_components(vc)

        def adjoint_mapping(v: Any) -> Any:
            vc = codomain.to_components(v)
            uc = smat.T @ vc
            return self.from_components(uc)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def dirac(self, point: List[float]) -> "LinearForm":
        """
        Returns the linear functional for point evaluation (Dirac measure).

        Args:
            point: A list containing `[latitude, longitude]`.
        """
        latitude, longitude = point
        colatitude = 90.0 - latitude

        coeffs = sh.expand.spharm(
            self.lmax,
            colatitude,
            longitude,
            normalization=self.normalization,
            degrees=True,
        )
        c = self._to_components_from_coeffs(coeffs)
        # Note: The user confirmed the logic for this return statement is correct
        # for their use case, despite the base class returning a LinearForm.
        return self.dual.from_components(c)

    def invariant_automorphism(self, f: Callable[[float], float]) -> "LinearOperator":
        matrix = self._degree_dependent_scaling_to_diagonal_matrix(f)

        def mapping(x: Any) -> Any:
            return self.from_components(matrix @ self.to_components(x))

        return LinearOperator.self_adjoint(self, mapping)

    def invariant_gaussian_measure(
        self, f: Callable[[float], float], /, *, expectation: Optional[Any] = None
    ) -> "GaussianMeasure":
        """
        Returns a Gaussian measure with a covariance of the form `f(Delta)`.

        Args:
            f: The scalar function `f(l(l+1))` defining the covariance.
            expectation: The mean of the measure. Defaults to zero.
        """

        def g(l: int) -> float:
            return np.sqrt(f(l) / self._sobolev_function(l))

        def h(l: int) -> float:
            return np.sqrt(self._sobolev_function(l) * f(l))

        matrix = self._degree_dependent_scaling_to_diagonal_matrix(g)
        adjoint_matrix = self._degree_dependent_scaling_to_diagonal_matrix(h)
        domain = EuclideanSpace(self.dim)

        def mapping(c: np.ndarray) -> Any:
            return self.from_components(matrix @ c)

        def adjoint_mapping(u: Any) -> np.ndarray:
            return adjoint_matrix @ self.to_components(u)

        inverse_matrix = self._degree_dependent_scaling_to_diagonal_matrix(
            lambda l: 1.0 / g(l)
        )
        inverse_adjoint_matrix = self._degree_dependent_scaling_to_diagonal_matrix(
            lambda l: 1.0 / h(l)
        )

        def inverse_mapping(u: Any) -> np.ndarray:
            return inverse_matrix @ self.to_components(u)

        def inverse_adjoint_mapping(c: np.ndarray) -> Any:
            # Note: This is a formal adjoint mapping.
            return self.from_components(inverse_adjoint_matrix @ c)

        covariance_factor = LinearOperator(
            domain, self, mapping, adjoint_mapping=adjoint_mapping
        )
        inverse_covariance_factor = LinearOperator(
            self, domain, inverse_mapping, adjoint_mapping=inverse_adjoint_mapping
        )

        return GaussianMeasure(
            covariance_factor=covariance_factor,
            inverse_covariance_factor=inverse_covariance_factor,
            expectation=expectation,
        )

    def plot(
        self,
        f: Any,
        /,
        *,
        projection: "Projection" = ccrs.PlateCarree(),
        contour: bool = False,
        cmap: str = "RdBu",
        coasts: bool = False,
        rivers: bool = False,
        borders: bool = False,
        map_extent: Optional[List[float]] = None,
        gridlines: bool = True,
        symmetric: bool = False,
        **kwargs,
    ) -> Tuple[Figure, "GeoAxes", Any]:
        """
        Creates a map plot of a function on the sphere using `cartopy`.

        Args:
            f: The function to be plotted (either an `SHGrid` or `SHCoeffs` object).
            projection: A `cartopy.crs` projection. Defaults to `PlateCarree`.
            contour: If True, creates a filled contour plot. Otherwise, a `pcolormesh` plot.
            cmap: The colormap name.
            coasts: If True, draws coastlines.
            rivers: If True, draws major rivers.
            borders: If True, draws country borders.
            map_extent: A list `[lon_min, lon_max, lat_min, lat_max]` to set map bounds.
            gridlines: If True, draws latitude/longitude gridlines.
            symmetric: If True, centers the color scale symmetrically around zero.
            **kwargs: Additional keyword arguments forwarded to the plotting function
                (`ax.contourf` or `ax.pcolormesh`).

        Returns:
            A tuple `(figure, axes, image)` containing the Matplotlib and Cartopy objects.
        """
        field: sh.SHGrid = (
            f
            if self._vector_as_SHGrid
            else f.expand(normalization=self.normalization, csphase=self.csphase)
        )

        lons = field.lons()
        lats = field.lats()

        figsize: Tuple[int, int] = kwargs.pop("figsize", (10, 8))
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})

        if map_extent is not None:
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        if coasts:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        if rivers:
            ax.add_feature(cfeature.RIVERS, linewidth=0.8)
        if borders:
            ax.add_feature(cfeature.BORDERS, linewidth=0.8)

        kwargs.setdefault("cmap", cmap)
        if symmetric:
            data_max = 1.2 * np.nanmax(np.abs(f.data))
            kwargs.setdefault("vmin", -data_max)
            kwargs.setdefault("vmax", data_max)

        levels = kwargs.pop("levels", 10)
        im: Any
        if contour:
            im = ax.contourf(
                lons,
                lats,
                field.data,
                transform=ccrs.PlateCarree(),
                levels=levels,
                **kwargs,
            )
        else:
            im = ax.pcolormesh(
                lons, lats, field.data, transform=ccrs.PlateCarree(), **kwargs
            )

        if gridlines:
            lat_interval = kwargs.pop("lat_interval", 30)
            lon_interval = kwargs.pop("lon_interval", 30)
            gl = ax.gridlines(
                linestyle="--",
                draw_labels=True,
                dms=True,
                x_inline=False,
                y_inline=False,
            )
            gl.xlocator = mticker.MultipleLocator(lon_interval)
            gl.ylocator = mticker.MultipleLocator(lat_interval)
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()

        return fig, ax, im

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a sphere.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters.
        """
        if not isinstance(other, Sobolev):
            return NotImplemented

        return (
            self.lmax == other.lmax
            and self.order == other.order
            and self.scale == other.scale
            and self.radius == other.radius
            and self.grid == other.grid
            and self._vector_as_SHGrid == other._vector_as_SHGrid
        )

    # ==============================================#
    #                Private methods                #
    # ==============================================#

    def _coefficient_to_component_mapping(self) -> coo_array:
        """Builds a sparse matrix to map `pyshtools` coeffs to component vectors."""
        row_dim = (self.lmax + 1) ** 2
        col_dim = 2 * (self.lmax + 1) ** 2

        row, col = 0, 0
        rows, cols = [], []
        for l in range(self.lmax + 1):
            col = l * (self.lmax + 1)
            for _ in range(l + 1):
                rows.append(row)
                row += 1
                cols.append(col)
                col += 1

        for l in range(self.lmax + 1):
            col = (self.lmax + 1) ** 2 + l * (self.lmax + 1) + 1
            for _ in range(1, l + 1):
                rows.append(row)
                row += 1
                cols.append(col)
                col += 1

        data = [1.0] * row_dim
        return coo_array(
            (data, (rows, cols)), shape=(row_dim, col_dim), dtype=float
        ).tocsc()

    def _to_components_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        """Returns a component vector from a `pyshtools` coefficient array."""
        f = coeffs.flatten(order="C")
        return self._sparse_coeffs_to_component @ f

    def _to_components_from_SHCoeffs(self, ulm: sh.SHCoeffs) -> np.ndarray:
        """Returns a component vector from an `SHCoeffs` object."""
        return self._to_components_from_coeffs(ulm.coeffs)

    def _to_components_from_SHGrid(self, u: sh.SHGrid) -> np.ndarray:
        """Returns a component vector from an `SHGrid` object."""
        ulm = u.expand(normalization=self.normalization, csphase=self.csphase)
        return self._to_components_from_SHCoeffs(ulm)

    def _from_components_to_SHCoeffs(self, c: np.ndarray) -> sh.SHCoeffs:
        """Returns an `SHCoeffs` object from its component vector."""
        f = self._sparse_coeffs_to_component.T @ c
        coeffs = f.reshape((2, self.lmax + 1, self.lmax + 1))
        return sh.SHCoeffs.from_array(
            coeffs, normalization=self.normalization, csphase=self.csphase
        )

    def _from_components_to_SHGrid(self, c: np.ndarray) -> sh.SHGrid:
        """Returns an `SHGrid` object from its component vector."""
        ulm = self._from_components_to_SHCoeffs(c)
        return ulm.expand(grid=self.grid, extend=self.extend)

    def _degree_dependent_scaling_to_diagonal_matrix(
        self, f: Callable[[int], float]
    ) -> diags:
        """Creates a diagonal sparse matrix from a function of degree `l`."""
        values = np.zeros(self.dim)
        i = 0
        for l in range(self.lmax + 1):
            j = i + l + 1
            values[i:j] = f(l)
            i = j
        for l in range(1, self.lmax + 1):
            j = i + l
            values[i:j] = f(l)
            i = j
        return diags([values], [0])

    def _sobolev_function(self, l: int) -> float:
        """The degree-dependent scaling that defines the Sobolev inner product."""
        return self.radius**2 * (1.0 + self.scale**2 * l * (l + 1)) ** self.order

    def _inner_product_impl(self, u: Any, v: Any) -> float:
        """Implements the Sobolev inner product in the spectral domain."""
        return np.dot(
            self._metric_tensor @ self.to_components(u), self.to_components(v)
        )

    def _to_dual_impl(self, u: Any) -> "LinearForm":
        """Implements the mapping to the dual space."""
        c = self._metric_tensor @ self.to_components(u)
        return self.dual.from_components(c)

    def _from_dual_impl(self, up: "LinearForm") -> Any:
        """Implements the mapping from the dual space."""
        c = self._inverse_metric_tensor @ self.dual.to_components(up)
        return self.from_components(c)

    def _ax_impl(self, a: float, x: Any) -> None:
        """
        Custom in-place ax implementation for pyshtools objects.
        x := a*x
        """
        if self._vector_as_SHGrid:
            # For SHGrid objects, modify the .data array
            x.data *= a
        else:
            # For SHCoeffs objects, modify the .coeffs array
            x.coeffs *= a

    def _axpy_impl(self, a: float, x: Any, y: Any) -> None:
        """
        Custom in-place axpy implementation for pyshtools objects.
        y := a*x + y
        """
        if self._vector_as_SHGrid:
            # For SHGrid objects, modify the .data array
            y.data += a * x.data
        else:
            # For SHCoeffs objects, modify the .coeffs array
            y.coeffs += a * x.coeffs

    def _vector_multiply_impl(self, u1: Any, u2: Any) -> Any:
        """Implements element-wise multiplication of two fields."""
        if self._vector_as_SHGrid:
            return u1 * u2
        else:
            u1_field = u1.expand(grid=self.grid, extend=self.extend)
            u2_field = u2.expand(grid=self.grid, extend=self.extend)
            u3_field = u1_field * u2_field
            return u3_field.expand(
                normalization=self.normalization, csphase=self.csphase
            )


class Lebesgue(Sobolev):
    """
    Implements the L^2 space on the two-sphere.

    This is a special case of the `Sobolev` class with `order = 0`.
    """

    def __init__(
        self,
        lmax: int,
        /,
        *,
        vector_as_SHGrid: bool = True,
        radius: float = 1.0,
        grid: str = "DH",
    ) -> None:
        """
        Args:
            lmax: The maximum spherical harmonic degree for truncation.
            vector_as_SHGrid: If True, elements are `SHGrid` objects. Otherwise,
                they are `SHCoeffs` objects.
            radius: The radius of the sphere.
            grid: The `pyshtools` grid type.
        """
        super().__init__(
            lmax,
            0.0,
            1.0,
            vector_as_SHGrid=vector_as_SHGrid,
            radius=radius,
            grid=grid,
        )


class LowPassFilter:
    """
    Implements a simple Hann-type low-pass filter in the spherical harmonic domain.
    """

    def __init__(self, lower_degree: int, upper_degree: int) -> None:
        """
        Args:
            lower_degree: Below this degree `l`, the filter gain is 1.
            upper_degree: Above this degree `l`, the filter gain is 0.
        """
        self._lower_degree: int = lower_degree
        self._upper_degree: int = upper_degree

    def __call__(self, l: int) -> float:
        if l <= self._lower_degree:
            return 1.0
        elif self._lower_degree <= l <= self._upper_degree:
            return 0.5 * (
                1.0
                - np.cos(
                    np.pi
                    * (self._upper_degree - l)
                    / (self._upper_degree - self._lower_degree)
                )
            )
        else:
            return 0.0
