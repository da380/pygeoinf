"""
Module for Sobolev spaces on the two-sphere.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, coo_array
import pyshtools as sh


from pygeoinf.hilbert_space import (
    HilbertSpace,
    LinearOperator,
    EuclideanSpace,
    LinearForm,
)

from pygeoinf.gaussian_measure import GaussianMeasure


class SHToolsHelper:
    """Helper class for working with pyshtool grid functions and coefficients."""

    def __init__(self, lmax, /, *, radius=1, grid="DH"):
        """
        Args:
            lmax (int): Maximum degree for spherical harmonic expansions.
            radius (float): radius of the sphere.
            grid (str): Grid type for spatial functions.
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

    @property
    def lmax(self):
        """Returns truncation degree."""
        return self._lmax

    @property
    def dim(self):
        """Returns dimension of vector space."""
        return (self.lmax + 1) ** 2

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


class Sobolev(SHToolsHelper, HilbertSpace):
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
        self._order = order
        self._scale = scale
        SHToolsHelper.__init__(self, lmax, radius=radius, grid=grid)
        self._metric_tensor = self._degree_dependent_scaling_to_diagonal_matrix(
            self._sobolev_function
        )
        self._inverse_metric_tensor = self._degree_dependent_scaling_to_diagonal_matrix(
            lambda l: 1 / self._sobolev_function(l)
        )
        self._vector_as_SHGrid = vector_as_SHGrid
        if vector_as_SHGrid:
            HilbertSpace.__init__(
                self,
                self.dim,
                self._to_components_from_SHGrid,
                self._from_components_to_SHGrid,
                self._inner_product_impl,
                self._to_dual_impl,
                self._from_dual_impl,
            )
        else:
            HilbertSpace.__init__(
                self,
                self.dim,
                self._to_components_from_SHCoeffs,
                self._from_components_to_SHCoeffs,
                self._inner_product_impl,
                self._to_dual_impl,
                self._from_dual_impl,
            )

    # =============================================#
    #                   Properties                 #
    # =============================================#

    @property
    def order(self):
        """Order of the Sobolev space."""
        return self._order

    @property
    def scale(self):
        """Non-dimensional length-scale."""
        return self._scale

    @property
    def laplace_beltrami_operator(self):
        """
        Returns the Laplace Beltrami operator on the space.
        """
        codomain = Sobolev(self.lmax, self.order - 2, self.scale)
        return self.invariant_operator(codomain, lambda l: l * (l + 1) / self.radius**2)

    # ==============================================#
    #                 Public methods                #
    # ==============================================#

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

    def dirac(self, latitude, longitude, /, *, degrees=True):
        """
        Returns the Dirac measure at the given point as an
        instance of LinearForm.

        Args:
            latitude (float): Latitude for the Dirac measure.
            longitude (float): Longitude for the Dirac measure.
            degrees (bool): If true, angles in degrees, otherwise
                they are in radians.
        Returns:
            LinearForm: The Dirac measure as a linear form on the space.

        Notes:
            The form is only well-defined mathematically if the order of the
            space is greater than 1.
        """
        if degrees:
            colatitude = 90 - latitude
        else:
            colatitude = np.pi / 2 - latitude
        coeffs = sh.expand.spharm(
            self.lmax,
            colatitude,
            longitude,
            normalization="ortho",
            degrees=degrees,
        )
        c = self._to_components_from_coeffs(coeffs)
        return self.dual.from_components(c)

    def dirac_representation(self, latitude, longitude, /, *, degrees=True):
        """
        Returns representation of the Dirac measure at the given point
        within the Sobolev space.

        Args:
            latitude (float): Latitude for the Dirac measure.
            longitude (float): Longitude for the Dirac measure.
            degrees (bool): If true, angles in degrees, otherwise
                they are in radians.
        Returns:
            LinearForm: The Dirac measure as a linear form on the space.

        Notes:
            The form is only well-defined mathematically if the order of the
            space is greater than 1.
        """
        up = self.dirac(latitude, longitude, degrees=degrees)
        return self.from_dual(up)

    def point_evaluation_operator(self, lats, lons, /, *, degrees=True):
        """
        Returns a LinearOperator instance that maps an element of the
        space to its values at the given set of latitudes and longitudes.

        Args:
            lats (ArrayLike): Array of latitudes.
            lons (ArrayLike): Array of longitudes.
            degrees (bool): If true, angles input as degrees, else in radians.

        Returns:
            LinearOperator: The operator on the space.

        Raises:
            ValueError: If the number of latitudes and longitudes are not equal.

        Notes:
            The operator is only well-defined mathematically if the order of the
            space is greater than 1.
        """

        if lats.size != lons.size:
            raise ValueError("Must have the same number of latitudes and longitudes.")
        codomain = EuclideanSpace(lats.size)

        def mapping(u):
            ulm = u.expand(normalization=self.normalization, csphase=self.csphase)
            return ulm.expand(lat=lats, lon=lons, degrees=degrees)

        def dual_mapping(vp):
            cvp = codomain.dual.to_components(vp)
            cup = np.zeros(self.dim)
            for c, lat, lon in zip(cvp, lats, lons):
                dirac = self.dirac(lat, lon, degrees=degrees)
                cup += c * self.dual.to_components(dirac)
            return LinearForm(self, components=cup)

        return LinearOperator(self, codomain, mapping, dual_mapping=dual_mapping)

    def invariant_operator(self, codomain, f):
        """
        Returns a rotationally invariant linear operator from the
        Sobolev space to another one.

        Args:
            codoaim (Sobolev): The codomain of the operator.
            f (callable): The degree-dependent scaling-function.

        Returns:
            LinearOperator: The linear operator.

        Raises:
            ValueError: If the codomain is not another Sobolev on a two-sphere.
        """
        if not isinstance(codomain, Sobolev):
            raise ValueError("Codomain must be another Sobolev space on a sphere.")
        matrix = self._degree_dependent_scaling_to_diagonal_matrix(f)
        if codomain == self:

            def mapping(x):
                return self.from_components(matrix @ self.to_components(x))

            return LinearOperator.self_adjoint(self, mapping)
        else:

            def mapping(x):
                return codomain.from_components(matrix @ self.to_components(x))

            def dual_mapping(yp):
                return self.dual.from_components(
                    matrix @ codomain.dual.to_components(yp)
                )

            return LinearOperator(self, codomain, mapping, dual_mapping=dual_mapping)

    def invariant_gaussian_measure(self, f, /, *, expectation=None):
        """
        Returns an invariant Gaussian measure on the space.

        Args:
            f (callable): Degree-dependent scaling function that defines the
                covariance operator.
            expectation (Sobolev vector | None): The expected value of the measure.

        Returns:
            GaussianMeasure: The Gaussian measure on the space.

        """

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

    def sobolev_gaussian_measure(self, order, scale, amplitude, /, *, expectation=None):
        """
        Returns an invariant Gaussian measure on the space whose covariance
        operator takes the Sobolev form:

        Args:
            order (float): The Sobolev order for the covariance.
            scale (float): The non-dimensional length-scale for the covariance.
            amplitude (float): The standard deviation of point values.
            expectation (Sobolev vector | None): The expected value of the measure.

        Returns:
            GaussianMeasures: The Gaussian measure on the space.
        """

        def f(l):
            return (1 + scale**2 * l * (l + 1)) ** (-order)

        return self.invariant_gaussian_measure(
            self._normalise_covariance_function(f, amplitude), expectation=expectation
        )

    def heat_kernel_gaussian_measure(self, scale, amplitude, /, *, expectation=None):
        """
        Returns an invariant Gaussian measure on the space whose covariance
        operator takes the form of a heat kernel.

        Args:
            scale (float): The non-dimensional length-scale for the covariance.
            amplitude (float): The standard deviation of point values.
            expectation (Sobolev vector | None): The expected value of the measure.

        Returns:
            GaussianMeasures: The Gaussian measure on the space.
        """

        def f(l):
            return np.exp(-0.5 * l * (l + 1) * scale**2)

        return self.invariant_gaussian_measure(
            self._normalise_covariance_function(f, amplitude), expectation=expectation
        )

    def sample_variance(self, vectors, /, *, expectation=None):
        """
        Given a list of elements in the space, forms a field
        of point-wise sample variances.

        Args:
            vectors ([vector]): A list of vectors.
            expectation (vector): The expected value that can be provided.
                If not, the sample expectation is used.
        """
        n = len(vectors)
        if expectation is None:
            assert n > 1
            fac = 1 / (n - 1)
            ubar = self.sample_expectation(vectors)
        else:
            assert n > 0
            fac = 1 / n
            ubar = expectation
        uvar = self.zero
        for u in vectors:
            u = (u - ubar) * (u - ubar)
            uvar = self.axpy(fac, u, uvar)
        return uvar

    def sample_std(self, vectors, /, *, expectation=None):
        """
        Given a list of elements in the space, forms a field
        of point-wise sample standard deviations.

        Args:
            vectors ([vector]): A list of vectors.
            expectation (vector): The expected value that can be provided.
                If not, the sample expectation is used.
        """
        ustd = self.sample_variance(vectors, expectation=expectation)
        ustd.data = np.sqrt(ustd.data)
        return ustd

    def plot(self, u, *args, **kwargs):
        """
        Make a simple plot of an element of the space.
        """
        if self._vector_as_SHGrid:
            plt.pcolormesh(u.lons(), u.lats(), u.data, *args, **kwargs)
        else:
            self.plot(u.expand(grid=self.grid, extend=self.extend), *args, **kwargs)

    # ==============================================#
    #                Private methods               #
    # ==============================================#

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
            0,
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
