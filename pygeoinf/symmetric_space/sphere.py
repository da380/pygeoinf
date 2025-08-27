"""
Provides concrete implementations of function spaces on the two-sphere (S²).

This module uses the abstract framework from the symmetric space module to create
fully-featured `Lebesgue` (L²) and `Sobolev` (Hˢ) Hilbert spaces for functions
defined on the surface of a sphere.

It utilizes the `pyshtools` library for highly efficient and accurate spherical
harmonic transforms. Following a compositional design, this module first
defines a base `Lebesgue` space and then constructs the `Sobolev` space as a
`MassWeightedHilbertSpace` over it. The module also includes powerful plotting
utilities built on `cartopy` for professional-quality geospatial visualization.

Key Classes
-----------
- `SphereHelper`: A mixin class providing the core geometry, spherical harmonic
  transform machinery, and `cartopy`-based plotting utilities.
- `Lebesgue`: A concrete implementation of the L²(S²) space of square-integrable
  functions on the sphere.
- `Sobolev`: A concrete implementation of the Hˢ(S²) space of functions with a
  specified degree of smoothness.
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

from pygeoinf.hilbert_space import (
    HilbertModule,
    MassWeightedHilbertModule,
)
from pygeoinf.operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from .symmetric_space import (
    AbstractInvariantLebesgueSpace,
    AbstractInvariantSobolevSpace,
)


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy.crs import Projection


class SphereHelper:
    """
    A mixin providing common functionality for function spaces on the sphere.

    This helper is not intended for direct instantiation. It provides the core
    geometry (radius, grid type), the spherical harmonic transform machinery via
    `pyshtools`, and `cartopy`-based plotting utilities that are shared by the
    `Lebesgue` and `Sobolev` space classes.
    """

    def __init__(
        self,
        lmax: int,
        radius: float,
        grid: str,
        extend: bool,
    ):
        """
        Args:
            lmax: The maximum spherical harmonic degree to be represented.
            radius: Radius of the sphere.
            grid: The `pyshtools` grid type.
            extend: If True, the spatial grid includes both 0 and 360-degree longitudes.
        """
        self._lmax: int = lmax
        self._radius: float = radius

        if grid == "DH2":
            self._grid = "DH"
            self._sampling = 2
        else:
            self._grid = grid
            self._sampling = 1

        self._extend: bool = extend

        # SH coefficient options fixed internally
        self._normalization: str = "ortho"
        self._csphase: int = 1

        # Set up sparse matrix that maps SHCoeff data arrrays into reduced form
        self._sparse_coeffs_to_component: coo_array = (
            self._coefficient_to_component_mapping()
        )

    def orthonormalised(self) -> bool:
        """The space is orthonormalised."""
        return True

    def _space(self):
        return self

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
    def sampling(self) -> int:
        """The sampling factor used for spatial representations."""
        return self._sampling

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

    @property
    def spatial_dimension(self) -> int:
        """The dimension of the space."""
        return 2

    def random_point(self) -> List[float]:
        """Returns a random point as `[latitude, longitude]`."""
        latitude = np.rad2deg(np.arcsin(np.random.uniform(-1.0, 1.0)))
        longitude = np.random.uniform(0.0, 360.0)
        return [latitude, longitude]

    def laplacian_eigenvalue(self, k: [int, int]) -> float:
        """
        Returns the (l.m)-th eigenvalue of the Laplacian.

        Args:
            k = (l,m): The index of the eigenvalue to return.
        """
        l = k[0]
        return l * (l + 1) / self.radius**2

    def degree_from_laplacian_eigenvalue(self, eig: float) -> float:
        """
        Returns the degree corresponding to a given eigenvalue.

        Note that the value is returned as a float

        Args:
            eig: The eigenvalue.
        """
        return np.sqrt(self.radius**2 * eig + 0.25)

    def trace_of_invariant_automorphism(self, f: Callable[[float], float]) -> float:
        """
        Returns the trace of the automorphism of the form f(Δ) with f a function
        that is well-defined on the spectrum of the Laplacian.

        Args:
            f: A real-valued function that is well-defined on the spectrum
               of the Laplacian.
        """
        trace = 0
        for l in range(self.lmax + 1):
            for m in range(-l, l + 1):
                trace += f(self.laplacian_eigenvalue((l, m)))
        return trace

    def project_function(self, f: Callable[[(float, float)], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        Args:
            f: A function that takes a point `(lat, lon)` and returns a value.
        """
        u = sh.SHGrid.from_zeros(
            self.lmax, grid=self.grid, extend=self.extend, sampling=self._sampling
        )
        for j, lon in enumerate(u.lons()):
            for i, lat in enumerate(u.lats()):
                u.data[i, j] = f((lat, lon))

        return u

    def to_coefficient(self, u: sh.SHGrid) -> sh.SHCoeffs:
        """Maps a function vector to its spherical harmonic coefficients."""
        return u.expand(normalization=self.normalization, csphase=self.csphase)

    def from_coefficient(self, ulm: sh.SHCoeffs) -> sh.SHGrid:
        """Maps spherical harmonic coefficients to a function vector."""
        grid = self.grid if self._sampling == 1 else "DH2"
        return ulm.expand(grid=grid, extend=self.extend)

    def plot(
        self,
        u: sh.SHGrid,
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
            u: The element to be plotted.
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

        lons = u.lons()
        lats = u.lats()

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
            data_max = 1.2 * np.nanmax(np.abs(u.data))
            kwargs.setdefault("vmin", -data_max)
            kwargs.setdefault("vmax", data_max)

        levels = kwargs.pop("levels", 10)
        im: Any
        if contour:
            im = ax.contourf(
                lons,
                lats,
                u.data,
                transform=ccrs.PlateCarree(),
                levels=levels,
                **kwargs,
            )
        else:
            im = ax.pcolormesh(
                lons, lats, u.data, transform=ccrs.PlateCarree(), **kwargs
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

    # --------------------------------------------------------------- #
    #                         private methods                         #
    # ----------------------------------------------------------------#

    def _grid_name(self):
        return self.grid if self._sampling == 1 else "DH2"

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

    def _degree_dependent_scaling_values(self, f: Callable[[int], float]) -> diags:
        """Creates a diagonal sparse matrix from a function of degree `l`."""
        dim = (self.lmax + 1) ** 2
        values = np.zeros(dim)
        i = 0
        for l in range(self.lmax + 1):
            j = i + l + 1
            values[i:j] = f(l)
            i = j
        for l in range(1, self.lmax + 1):
            j = i + l
            values[i:j] = f(l)
            i = j
        return values

    def _coefficient_to_component(self, ulm: sh.SHCoeffs) -> np.ndarray:
        """Maps spherical harmonic coefficients to a component vector."""
        flat_coeffs = ulm.coeffs.flatten(order="C")
        return self._sparse_coeffs_to_component @ flat_coeffs

    def _component_to_coefficient(self, c: np.ndarray) -> sh.SHCoeffs:
        """Maps a component vector to spherical harmonic coefficients."""
        flat_coeffs = self._sparse_coeffs_to_component.T @ c
        coeffs = flat_coeffs.reshape((2, self.lmax + 1, self.lmax + 1))
        return sh.SHCoeffs.from_array(
            coeffs, normalization=self.normalization, csphase=self.csphase
        )


class Lebesgue(SphereHelper, HilbertModule, AbstractInvariantLebesgueSpace):
    """
    Implementation of the Lebesgue space L² on the sphere.

    This class represents square-integrable functions on a sphere. A function is
    represented by a `pyshtools.SHGrid` object, which stores its values on a
    regular grid in latitude and longitude. The L² inner product is defined
    in the spherical harmonic domain.
    """

    def __init__(
        self,
        lmax: int,
        /,
        *,
        radius: float = 1,
        grid: str = "DH",
        extend: bool = True,
    ):

        if lmax < 0:
            raise ValueError("lmax must be non-negative")

        self._dim = (lmax + 1) ** 2

        SphereHelper.__init__(self, lmax, radius, grid, extend)

    @property
    def dim(self) -> int:
        """The dimension of the space."""
        return self._dim

    def to_components(self, u: sh.SHGrid) -> np.ndarray:
        coeff = self.to_coefficient(u)
        return self._coefficient_to_component(coeff)

    def from_components(self, c: np.ndarray) -> sh.SHGrid:
        coeff = self._component_to_coefficient(c)
        return self.from_coefficient(coeff)

    def to_dual(self, u: sh.SHGrid) -> LinearForm:
        coeff = self.to_coefficient(u)
        cp = self._coefficient_to_component(coeff) * self.radius**2
        return self.dual.from_components(cp)

    def from_dual(self, up: LinearForm) -> sh.SHGrid:
        cp = self.dual.to_components(up) / self.radius**2
        coeff = self._component_to_coefficient(cp)
        return self.from_coefficient(coeff)

    def ax(self, a: float, x: sh.SHGrid) -> None:
        """
        Custom in-place ax implementation for pyshtools objects.
        x := a*x
        """
        x.data *= a

    def axpy(self, a: float, x: sh.SHGrid, y: sh.SHGrid) -> None:
        """
        Custom in-place axpy implementation for pyshtools objects.
        y := a*x + y
        """
        y.data += a * x.data

    def vector_multiply(self, x1: sh.SHGrid, x2: sh.SHGrid) -> sh.SHGrid:
        """
        Computes the pointwise product of two functions.
        """
        return x1 * x2

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a sphere.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Lebesgue):
            return NotImplemented

        return self.lmax == other.lmax and self.radius == other.radius

    def eigenfunction_norms(self) -> np.ndarray:
        """Returns a list of the norms of the eigenfunctions."""
        return np.fromiter(
            [self.radius for i in range(self.dim)],
            dtype=float,
        )

    def invariant_automorphism_from_index_function(
        self, g: Callable[[(int, int)], float]
    ) -> LinearOperator:
        values = self._degree_dependent_scaling_values(lambda l: g((l, 0)))
        matrix = diags([values], [0])

        def mapping(u):
            c = matrix @ (self.to_components(u))
            coeff = self._component_to_coefficient(c)
            return self.from_coefficient(coeff)

        return LinearOperator.self_adjoint(self, mapping)

    def __str__(self) -> str:
        """Returns a human-readable string summary of the space."""
        return (
            f"Lebesgue space on sphere:\n"
            f"lmax={self.lmax}\n"
            f"radius={self.radius}\n"
            f"grid={self.grid}\n"
            f"extend={self.extend}"
        )


class Sobolev(SphereHelper, MassWeightedHilbertModule, AbstractInvariantSobolevSpace):
    """
    Implementation of the Sobolev space Hˢ on the sphere.

    This class represents functions with a specified degree of smoothness. It is
    constructed as a `MassWeightedHilbertModule` over the `Lebesgue` space, where
    the mass operator weights the spherical harmonic coefficients to enforce
    smoothness. This is the primary class for defining smooth, random function
    fields (e.g., for geophysics or climate science).
    """

    def __init__(
        self,
        lmax: int,
        order: float,
        scale: float,
        /,
        radius: float = 1,
        grid: str = "DH",
        extend: bool = True,
    ):

        if lmax < 0:
            raise ValueError("lmax must be non-negative")

        SphereHelper.__init__(self, lmax, radius, grid, extend)
        AbstractInvariantSobolevSpace.__init__(self, order, scale)

        lebesgue = Lebesgue(lmax, radius=radius, grid=grid, extend=extend)

        mass_operator = lebesgue.invariant_automorphism(self.sobolev_function)
        inverse_mass_operator = lebesgue.invariant_automorphism(
            lambda k: 1.0 / self.sobolev_function(k)
        )

        MassWeightedHilbertModule.__init__(
            self, lebesgue, mass_operator, inverse_mass_operator
        )

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
            term = 1 / sobolev_func(l)
            summation += term
            err = term / summation
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
            radius=radius,
            grid=grid,
        )

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a sphere.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Sobolev):
            return NotImplemented

        return (
            self.lmax == other.lmax
            and self.radius == other.radius
            and self.order == other.order
            and self.scale == other.scale
        )

    def eigenfunction_norms(self) -> np.ndarray:
        """Returns a list of the norms of the eigenfunctions."""
        values = self._degree_dependent_scaling_values(
            lambda l: np.sqrt(self.sobolev_function(self.laplacian_eigenvalue((l, 0))))
        )
        return self.radius * np.fromiter(values, dtype=float)

    def dirac(self, point: (float, float)) -> LinearForm:
        """
        Returns the linear functional for point evaluation (Dirac measure).

        Args:
            point: A tuple containing `(latitude, longitude)`.
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
        ulm = sh.SHCoeffs.from_array(
            coeffs,
            normalization=self.normalization,
            csphase=self.csphase,
        )

        c = self._coefficient_to_component(ulm)
        return self.dual.from_components(c)

    def __str__(self) -> str:
        """Returns a human-readable string summary of the space."""
        return (
            f"Lebesgue space on sphere:\n"
            f"lmax={self.lmax}\n"
            f"order={self.order}\n"
            f"scale={self.scale}\n"
            f"radius={self.radius}\n"
            f"grid={self.grid}\n"
            f"extend={self.extend}"
        )
