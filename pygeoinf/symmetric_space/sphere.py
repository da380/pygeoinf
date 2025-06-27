"""
Module for Sobolev spaces on the two-sphere.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, coo_array
import pyshtools as sh

import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


from pygeoinf.hilbert_space import LinearOperator, EuclideanSpace
from pygeoinf.symmetric_space.symmetric_space import SymmetricSpaceSobolev
from pygeoinf.gaussian_measure import GaussianMeasure


class Sobolev(SymmetricSpaceSobolev):
    """Sobolev spaces on a two-sphere as an instance of HilbertSpace."""

    def __init__(
        self, lmax, order, scale, /, *, vector_as_SHGrid=True, radius=1, grid="DH"
    ):
        """
        Args:
            lmax (int): Truncation degree for spherical harmoincs.
            order (float): Order of the Sobolev space.
            scale (float): Non-dimensional length-scale for the space.
            vector_as_SHGrid (bool): If true, elements of the space are
                instances of the SHGrid class, otherwise they are SHCoeffs.
            radius (float): Radius of the two sphere.
            grid (str): pyshtools grid type.
        """

        self._lmax = lmax
        self._radius = radius
        self._grid = grid
        if self.grid == "DH2":
            self._sampling = 2
        else:
            self._sampling = 1
        self._extend = True
        self._normalization = "ortho"
        self._csphase = 1
        self._sparse_coeffs_to_component = self._coefficient_to_component_mapping()

        dim = (lmax + 1) ** 2

        self._vector_as_SHGrid = vector_as_SHGrid
        if vector_as_SHGrid:
            SymmetricSpaceSobolev.__init__(
                self,
                order,
                scale,
                dim,
                self._to_components_from_SHGrid,
                self._from_components_to_SHGrid,
                self._inner_product_impl,
                self._to_dual_impl,
                self._from_dual_impl,
                vector_multiply=self._vector_multiply_impl,
            )
        else:
            SymmetricSpaceSobolev.__init__(
                self,
                order,
                scale,
                dim,
                self._to_components_from_SHCoeffs,
                self._from_components_to_SHCoeffs,
                self._inner_product_impl,
                self._to_dual_impl,
                self._from_dual_impl,
                vector_multiply=self._vector_multiply_impl,
            )

        self._metric_tensor = self._degree_dependent_scaling_to_diagonal_matrix(
            self._sobolev_function
        )
        self._inverse_metric_tensor = self._degree_dependent_scaling_to_diagonal_matrix(
            lambda l: 1 / self._sobolev_function(l)
        )

    # ===============================================#
    #                   Properties                  #
    # ===============================================#

    @property
    def lmax(self):
        """Returns truncation degree."""
        return self._lmax

    @property
    def radius(self):
        """Returns radius of the sphere."""
        return self._radius

    @property
    def grid(self):
        """Returns spatial grid option."""
        return self._grid

    @property
    def extend(self):
        """True is spatial grid contains longitudes 0 and 360."""
        return self._extend

    @property
    def normalization(self):
        """Spherical harmonic normalisation option."""
        return self._normalization

    @property
    def csphase(self):
        """Condon Shortley phase option."""
        return self._csphase

    # ==============================================#
    #                 Public methods                #
    # ==============================================#

    def random_point(self):
        latitude = np.random.uniform(-90, 90)
        longitude = np.random.uniform(0, 360)
        return [latitude, longitude]

    def low_degree_projection(self, truncation_degree, /, *, smoother=None):
        """
        Returns a LinearOperator that maps the space onto a Sobolev space with
        the same parameters but based on a lower truncation degree.
        """
        truncation_degree = (
            truncation_degree if truncation_degree <= self.lmax else self.lmax
        )
        f = smoother if smoother is not None else lambda l: l

        # construct the spare matrix that performs the coordinate projection.
        row_dim = (truncation_degree + 1) ** 2
        col_dim = (self.lmax + 1) ** 2

        row = 0
        col = 0
        rows = []
        cols = []
        data = []
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
            (data, (rows, cols)), shape=(row_dim, col_dim), dtype=int
        ).tocsc()

        codomain = Sobolev(
            truncation_degree,
            self.order,
            self.scale,
            vector_as_SHGrid=self._vector_as_SHGrid,
            radius=self.radius,
            grid=self._grid,
        )

        def mapping(u):
            uc = self.to_components(u)
            vc = smat @ uc
            return codomain.from_components(vc)

        def adjoint_mapping(v):
            vc = codomain.to_components(v)
            uc = smat.T @ vc
            return self.from_components(uc)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def dirac(self, point):

        latitude, longitude = point
        colatitude = 90 - latitude

        coeffs = sh.expand.spharm(
            self.lmax,
            colatitude,
            longitude,
            normalization="ortho",
            degrees=True,
        )
        c = self._to_components_from_coeffs(coeffs)
        return self.dual.from_components(c)

    def invariant_automorphism(self, f):
        matrix = self._degree_dependent_scaling_to_diagonal_matrix(f)

        def mapping(x):
            return self.from_components(matrix @ self.to_components(x))

        return LinearOperator.self_adjoint(self, mapping)

    def invariant_gaussian_measure(self, f, /, *, expectation=None):

        def g(l):
            return np.sqrt(f(l) / (self.radius**2 * self._sobolev_function(l)))

        def h(l):
            return np.sqrt(self.radius**2 * self._sobolev_function(l) * f(l))

        matrix = self._degree_dependent_scaling_to_diagonal_matrix(g)
        adjoint_matrix = self._degree_dependent_scaling_to_diagonal_matrix(h)
        domain = EuclideanSpace(self.dim)

        def mapping(c):
            return self.from_components(matrix @ c)

        def adjoint_mapping(u):
            return adjoint_matrix @ self.to_components(u)

        inverse_matrix = self._degree_dependent_scaling_to_diagonal_matrix(
            lambda l: 1 / g(l)
        )

        inverse_adjoint_matrix = self._degree_dependent_scaling_to_diagonal_matrix(
            lambda l: 1 / h(l)
        )

        def inverse_mapping(u):
            return inverse_matrix @ self.to_components(u)

        def inverse_adjoint_mapping(c):
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
        f,
        /,
        *,
        projection=ccrs.PlateCarree(),
        contour=False,
        cmap="RdBu",
        coasts=False,
        rivers=False,
        borders=False,
        map_extent=None,
        gridlines=True,
        symmetric=False,
        **kwargs,
    ):
        """
        Return a plot of a function.

        Args:
            f (vector): Function to be plotted.
            projection: cartopy projection to be used. Default is Robinson.
            contour (bool): If True, a contour plot is made, otherwise a pcolor plot.
            cmap (string): colormap. Default is RdBu.
            coasts (bool): If True, coast lines plotted. Default is True.
            rivers (bool): If True, major rivers plotted. Default is False.
            borders (bool): If True, country borders are plotted. Default is False.
            map_extent ([float]): Sets the (lon, lat) range for plotting.
            Tuple of [lon_min, lon_max, lat_min, lat_max]. Default is None.
            gridlines (bool): If True, gridlines are included. Default is True.
            symmetric (bool): If True, clim values set symmetrically based on the fields maximum absolute value.
                Option overridden if vmin or vmax are set.
            kwargs: Keyword arguments for forwarding to the plotting functions.
        """

        if self._vector_as_SHGrid:
            field = f
        else:
            field = f.expand(normalization=self.normalization, csphase=self.csphase)

        lons = field.lons()
        lats = field.lats()

        figsize = kwargs.pop("figsize", (10, 8))
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

        lat_interval = kwargs.pop("lat_interval", 30)
        lon_interval = kwargs.pop("lon_interval", 30)

        if symmetric:
            data_max = 1.2 * np.nanmax(np.abs(f.data))
            kwargs.setdefault("vmin", -data_max)
            kwargs.setdefault("vmax", data_max)

        levels = kwargs.pop("levels", 10)

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
                lons,
                lats,
                field.data,
                transform=ccrs.PlateCarree(),
                **kwargs,
            )

        if gridlines:
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

    # ==============================================#
    #                Private methods                #
    # ==============================================#

    def _coefficient_to_component_mapping(self):
        # returns a sparse matrix that maps from flattended pyshtools
        # coefficients to component vectors.

        row_dim = (self.lmax + 1) ** 2
        col_dim = 2 * (self.lmax + 1) ** 2

        row = 0
        rows = []
        cols = []
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

        data = [1] * row_dim
        return coo_array(
            (data, (rows, cols)), shape=(row_dim, col_dim), dtype=int
        ).tocsc()

    def spherical_harmonic_index(self, l, m):
        """Return the component index for given spherical harmonic degree and order."""
        if m >= 0:
            return int(l * (l + 1) / 2) + m
        else:
            offset = int((self.lmax + 1) * (self.lmax + 2) / 2)
            return offset + int((l - 1) * l / 2) - m - 1

    def _to_components_from_coeffs(self, coeffs):
        """Return component vector from coefficient array."""
        f = coeffs.flatten(order="C")
        return self._sparse_coeffs_to_component @ f

    def _to_components_from_SHCoeffs(self, ulm):
        """Return component vector from SHCoeffs object."""
        return self._to_components_from_coeffs(ulm.coeffs)

    def _to_components_from_SHGrid(self, u):
        """Return component vector from SHGrid object."""
        ulm = u.expand(normalization=self.normalization, csphase=self.csphase)
        return self._to_components_from_SHCoeffs(ulm)

    def _from_components_to_SHCoeffs(self, c):
        """Return SHCoeffs object from its component vector."""
        f = self._sparse_coeffs_to_component.T @ c
        coeffs = f.reshape((2, self.lmax + 1, self.lmax + 1))
        return sh.SHCoeffs.from_array(
            coeffs, normalization=self.normalization, csphase=self.csphase
        )

    def _from_components_to_SHGrid(self, c):
        """Return SHGrid object from its component vector."""
        ulm = self._from_components_to_SHCoeffs(c)
        return ulm.expand(grid=self.grid, extend=self.extend)

    def _degree_dependent_scaling_to_diagonal_matrix(self, f):
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

    def _sobolev_function(self, l):
        # Degree-dependent scaling that defines the Sobolev inner product.
        return (1 + self.scale**2 * l * (l + 1)) ** self.order

    def _inner_product_impl(self, u, v):
        # Implementation of the inner product.
        return self.radius**2 * np.dot(
            self._metric_tensor @ self.to_components(u), self.to_components(v)
        )

    def _to_dual_impl(self, u):
        # Implementation of the mapping to the dual space.
        c = self._metric_tensor @ self.to_components(u) * self.radius**2
        return self.dual.from_components(c)

    def _from_dual_impl(self, up):
        # Implementation of the mapping from the dual space.
        c = self._inverse_metric_tensor @ self.dual.to_components(up) / self.radius**2
        return self.from_components(c)

    def _normalise_covariance_function(self, f, amplitude):
        # Normalise a degree-dependent scaling function, f, so that
        # the associated invariant Gaussian measure has standard deviation
        # for point values equal to amplitude.
        norm = 0
        for l in range(self.lmax + 1):
            norm += (
                f(l)
                * (2 * l + 1)
                / (4 * np.pi * self.radius**2 * self._sobolev_function(l))
            )
        return lambda l: amplitude**2 * f(l) / norm

    def _vector_multiply_impl(self, u1, u2):

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
    L2 on the two-sphere as an instance of HilbertSpace.

    Implemented as a special case of the Sobolev class with order = 0.
    """

    def __init__(self, lmax, /, *, vector_as_SHGrid=True, radius=1, grid="DH"):
        """
        Args:
            lmax (int): Truncation degree for spherical harmoincs.
            vector_as_SHGrid (bool): If true, elements of the space are
                instances of the SHGrid class, otherwise they are SHCoeffs.
            radius (float): Radius of the two sphere.
            grid (str): pyshtools grid type.
        """
        super().__init__(
            lmax,
            0,
            1,
            vector_as_SHGrid=vector_as_SHGrid,
            radius=radius,
            grid=grid,
        )


###############################################################
#                      Utility classes                        #
###############################################################


class LowPassFilter:
    """
    Class implementing a simple Hann-type low-pass filter in
    the spherical harmonic domain
    """

    def __init__(self, lower_degree, upper_degree):
        """
        Args:
            lower_degree (int): Below this degree, the filter returns one.
            upper_degree (int): Above this degree, the filter returns zero.
                Its value between the lower and upper degrees decreases smoothly.
        """
        self._lower_degree = lower_degree
        self._upper_degree = upper_degree

    def __call__(self, l):
        if l <= self._lower_degree:
            return 1
        elif self._lower_degree <= l <= self._upper_degree:
            return 0.5 * (
                1
                - np.cos(
                    np.pi
                    * (self._upper_degree - l)
                    / (self._upper_degree - self._lower_degree)
                )
            )
        else:
            return 0
