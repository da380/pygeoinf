"""
Provides concrete implementations of function spaces on the two-sphere (S²).

This module uses the abstract framework from the symmetric space module to create
fully-featured `Lebesgue` (L²) and `Sobolev` (Hˢ) Hilbert spaces for functions
defined on the surface of a sphere.

It utilizes the `pyshtools` library for highly efficient and accurate spherical
harmonic transforms. 

Following a compositional design, this module first defines a base `Lebesgue`
space with the standard L² inner product. The `Sobolev` space, which encodes
smoothness, is then constructed as a `MassWeightedHilbertSpace` over this
Lebesgue space. The module also includes powerful plotting utilities built on
`cartopy` for professional-quality geospatial visualization.

Key Classes
-----------
SphereHelper
    A mixin class providing the core geometry, spherical harmonic transform
    machinery using `pyshtools`, and `cartopy`-based plotting utilities.
Lebesgue
    A concrete implementation of the L²(S²) space of square-integrable
    functions on the sphere.
Sobolev
    A concrete implementation of the Hˢ(S²) space, which represents functions
    with a specified degree of smoothness.
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
    HilbertSpace,
    MassWeightedHilbertSpace,
)
from pygeoinf.operators import LinearOperator
from pygeoinf.linear_forms import LinearForm
from pygeoinf.symmetric_space_new.symmetric_space import (
    AbstractInvariantLebesgueSpace,
    AbstractInvariantSobolevSpace,
)


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from cartopy.mpl.geoaxes import GeoAxes
    from cartopy.crs import Projection


class SphereHelper:
    """
    Helper class for function spaces on the sphere.
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


class Lebesgue(SphereHelper, HilbertSpace, AbstractInvariantLebesgueSpace):
    """
    Implements L²(S²) as an instance of HilbertSpace based on spherical harmonic
    transformations.
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

        SphereHelper.__init__(self, lmax, radius, grid, extend)

        HilbertSpace.__init__(
            self,
            (lmax + 1) ** 2,
            self._to_components_impl,
            self._from_components_impl,
            self._to_dual_impl,
            self._from_dual_impl,
            ax=self._ax_impl,
            axpy=self._axpy_impl,
            vector_multiply=lambda u1, u2: u1 * u2,
        )

    def eigenfunction_norms(self) -> np.ndarray:
        """Returns a list of the norms of the eigenfunctions."""
        return np.fromiter(
            [self.radius for i in range(self.dim)],
            dtype=float,
        )

    def invariant_automorphism(self, f: Callable[[float], float]) -> LinearOperator:
        values = self._degree_dependent_scaling_values(
            lambda l: f(self.laplacian_eigenvalue((l, 0)))
        )
        matrix = diags([values], [0])

        def mapping(u):
            c = matrix @ (self.to_components(u))
            coeff = self._component_to_coefficient(c)
            return self.from_coefficient(coeff)

        return LinearOperator.self_adjoint(self, mapping)

    # ================================================================ #
    #                         Private methods                          #
    # ================================================================ #

    def _to_components_impl(self, u: sh.SHGrid) -> np.ndarray:
        coeff = self.to_coefficient(u)
        return self._coefficient_to_component(coeff)

    def _from_components_impl(self, c: np.ndarray) -> sh.SHGrid:
        coeff = self._component_to_coefficient(c)
        return self.from_coefficient(coeff)

    def _to_dual_impl(self, u: sh.SHGrid) -> LinearForm:
        coeff = self.to_coefficient(u)
        cp = self._coefficient_to_component(coeff) * self.radius**2
        return self.dual.from_components(cp)

    def _from_dual_impl(self, up: LinearForm) -> sh.SHGrid:
        cp = self.dual.to_components(up) / self.radius**2
        coeff = self._component_to_coefficient(cp)
        return self.from_coefficient(coeff)

    def _ax_impl(self, a: float, x: Any) -> None:
        """
        Custom in-place ax implementation for pyshtools objects.
        x := a*x
        """
        x.data *= a

    def _axpy_impl(self, a: float, x: Any, y: Any) -> None:
        """
        Custom in-place axpy implementation for pyshtools objects.
        y := a*x + y
        """
        y.data += a * x.data


class Sobolev(SphereHelper, MassWeightedHilbertSpace, AbstractInvariantSobolevSpace):
    """
    Implements Hˢ(S²) as an instance of MassWeightedHilbertSpace.
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

        MassWeightedHilbertSpace.__init__(
            self, lebesgue, mass_operator, inverse_mass_operator
        )

    def eigenfunction_norms(self) -> np.ndarray:
        """Returns a list of the norms of the eigenfunctions."""
        values = self._degree_dependent_scaling_values(
            lambda l: np.sqrt(self.sobolev_function(self.laplacian_eigenvalue((l, 0))))
        )
        return self.radius * np.fromiter(values, dtype=float)

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a circle.

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
