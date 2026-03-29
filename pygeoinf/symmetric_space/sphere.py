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
"""

from __future__ import annotations
from typing import Callable, Any, List, Optional, Tuple, TYPE_CHECKING

import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from scipy.spatial import cKDTree


try:
    import pyshtools as sh
    from pyshtools.shio import SHCilmToVector
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from cartopy.mpl.geoaxes import GeoAxes
except ImportError:
    raise ImportError(
        "pyshtools and cartopy are required for the sphere module. "
        "Please install them with 'pip install pygeoinf[sphere]'"
    ) from None


from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_operators import LinearOperator


from .symmetric_space import AbstractSymmetricLebesgueSpace, SymmetricSobolevSpace


if TYPE_CHECKING:
    from cartopy.crs import Projection
    from pyshtools import SHGrid


class Lebesgue(AbstractSymmetricLebesgueSpace):
    """
    Implementation of the Lebesgue space L² on a sphere.

    This class represents square-integrable functions on a sphere. A function is
    represented by its values on an evenly spaced grid. The co-ordinate basis for
    the space is through spherical harmonic expansions.
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
        """
        Args:
            lmax: Maximum degree for the expansions.
            radius: Radius of the sphere. Defaults to 1.
            grid: pyshtools grid type. Defaults to "DH"
            extend: If true longitudes wrap fully. Defaults to True.
        """

        if lmax < 0:
            raise ValueError("lmax must be non-negative")

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

        AbstractSymmetricLebesgueSpace.__init__(self, 2, lmax, (lmax + 1) ** 2, False)

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        /,
        *,
        radius: float = 1.0,
        grid: str = "DH",
        extend: bool = True,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space with a spherical harmonic truncation degree (`lmax`)
        automatically chosen to capture the expected energy of functions drawn from
        a specified prior measure.

        This factory method calculates the expected squared norm (energy) of a random field
        whose spectral variances are defined by the provided `covariance_function`. It iteratively
        adds higher spherical harmonic degrees until the relative contribution of the next degree
        drops below the specified relative tolerance.

        Args:
            covariance_function: A callable mapping a Laplacian eigenvalue to its spectral variance.
            radius: The radius of the sphere. Defaults to 1.0.
            grid: The pyshtools spatial grid format to use. Defaults to "DH".
            extend: If True, the spatial grid includes both 0 and 360-degree longitudes.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree. If convergence
                is not reached by this degree, the search terminates and returns this value.
            power_of_two: If True, rounds the resulting `lmax` up to the nearest power of two.

        Returns:
            Lebesgue: A fully instantiated L² space on the sphere with the optimal `lmax`.

        Raises:
            RuntimeError: If the energy sequence fails to converge within 100,000 iterations
                and no `max_degree` is specified.
        """
        dummy_space = cls(max(1, min_degree), radius=radius, grid=grid, extend=extend)

        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )

        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(optimal_degree, radius=radius, grid=grid, extend=extend)

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        /,
        *,
        radius: float = 1.0,
        grid: str = "DH",
        extend: bool = True,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space on the sphere, tuned to the expected
        energy of a Heat Kernel prior measure.

        Args:
            kernel_scale: The length-scale parameter of the heat kernel covariance.
            radius: The radius of the sphere. Defaults to 1.0.
            grid: The pyshtools spatial grid format to use. Defaults to "DH".
            extend: If True, the spatial grid includes both 0 and 360-degree longitudes.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `lmax` up to the nearest power of two.
        """
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale),
            radius=radius,
            grid=grid,
            extend=extend,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
        )

    @classmethod
    def from_sobolev_kernel_prior(
        cls,
        kernel_order: float,
        kernel_scale: float,
        /,
        *,
        radius: float = 1.0,
        grid: str = "DH",
        extend: bool = True,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
    ) -> Lebesgue:
        """
        Creates an instance of the L² space on the sphere, tuned to the expected
        energy of a Sobolev-type prior measure.

        Args:
            kernel_order: The smoothness order of the Sobolev prior measure.
            kernel_scale: The length-scale parameter of the Sobolev prior measure.
            radius: The radius of the sphere. Defaults to 1.0.
            grid: The pyshtools spatial grid format to use. Defaults to "DH".
            extend: If True, the spatial grid includes both 0 and 360-degree longitudes.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `lmax` up to the nearest power of two.
        """
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale),
            radius=radius,
            grid=grid,
            extend=extend,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
        )

    # ------------------------------------------------------ #
    #                       Properties                       #
    # ------------------------------------------------------ #

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
    def grid_type(self) -> str:
        """
        Returns the pyshtools grid type.
        """
        return self.grid if self._sampling == 1 else "DH2"

    # ------------------------------------------------------ #
    #                    Public methods                      #
    # ------------------------------------------------------ #

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

    def to_coefficients(self, u: sh.SHGrid) -> sh.SHCoeffs:
        """Maps a function vector to its spherical harmonic coefficients."""
        return u.expand(normalization=self.normalization, csphase=self.csphase)

    def from_coefficients(self, ulm: sh.SHCoeffs) -> sh.SHGrid:
        """Maps spherical harmonic coefficients to a function vector."""
        grid = self.grid if self._sampling == 1 else "DH2"
        return ulm.expand(grid=grid, extend=self.extend)

    def sample_power_measure(
        self,
        measure,
        n_samples,
        /,
        *,
        lmin=None,
        lmax=None,
        parallel: bool = False,
        n_jobs: int = -1,
    ):
        """
        Takes in a Gaussian measure on the space, draws n_samples from
        and returns samples for the spherical harmonic power at degrees in
        the indicated range.
        """

        lmin = 0 if lmin is None else lmin
        lmax = self.lmax if lmax is None else min(self.lmax, lmax)

        samples = measure.samples(n_samples, parallel=parallel, n_jobs=n_jobs)

        powers = []
        for u in samples:
            ulm = self.to_coefficients(u)
            powers.append(ulm.spectrum(lmax=lmax, convention="power")[lmin:])

        return powers

    # ------------------------------------------------------ #
    #           Methods for SymmetricHilbertSpace            #
    # ------------------------------------------------------ #

    def index_to_integer(self, k: Tuple[int, int]) -> int:
        l, m = k
        if abs(m) > l or l < 0:
            raise ValueError("Invalid spherical harmonic: |m| must be <= l, and l >= 0")

        # Pure Python is much faster for scalars
        return l**2 + m if m >= 0 else l**2 + l + abs(m)

    def integer_to_index(self, i: int) -> Tuple[int, int]:
        if i < 0:
            raise ValueError("Index cannot be negative.")

        # math.isqrt is vastly faster than np.floor(np.sqrt(i))
        l = math.isqrt(i)
        r = i - l**2
        m = r if r <= l else l - r

        return l, m

    def laplacian_eigenvalue(self, k: Tuple[int, int]) -> float:
        """
        Returns the (l.m)-th eigenvalue of the Laplacian.

        Args:
            k = (l,m): The index of the eigenvalue to return.
        """
        l = k[0]
        return l * (l + 1) / self.radius**2

    def laplacian_eigenvector_squared_norm(self, k: Tuple[int, int]) -> float:
        return self.radius**2

    def laplacian_eigenvectors_at_point(self, point: Tuple[float, float]) -> np.ndarray:
        latitude, longitude = point
        colatitude = 90.0 - latitude

        coeffs = sh.expand.spharm(
            self.lmax,
            colatitude,
            longitude,
            normalization=self.normalization,
            degrees=True,
        )
        return SHCilmToVector(coeffs)

    def random_point(self) -> Tuple[float, float]:
        """Returns a random point as `[latitude, longitude]`."""
        latitude = np.rad2deg(np.arcsin(np.random.uniform(-1.0, 1.0)))
        longitude = np.random.uniform(0.0, 360.0)
        return (latitude, longitude)

    def geodesic_distance(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """Returns the great-circle distance between two points on the sphere."""
        v1, v2 = self._to_vector(*p1), self._to_vector(*p2)
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        omega = np.arccos(dot_product)
        return float(self.radius * omega)

    def geodesic_quadrature(
        self, p1: Tuple[float, float], p2: Tuple[float, float], n_points: int
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """Generates Gauss-Legendre quadrature points and weights along a great-circle arc."""

        arc_length = self.geodesic_distance(p1, p2)
        omega = arc_length / self.radius

        if omega < 1e-10:
            return [p1] * n_points, np.zeros(n_points)

        if np.abs(omega - np.pi) < 1e-10:
            raise ValueError(
                "Points are antipodal; the great circle path is not unique."
            )

        v1, v2 = self._to_vector(*p1), self._to_vector(*p2)
        x, w = np.polynomial.legendre.leggauss(n_points)

        t_vals = (x + 1) / 2.0
        scaled_weights = w * (arc_length / 2.0)

        sin_omega = np.sin(omega)
        points = []

        for t in t_vals:
            coeff1 = np.sin((1 - t) * omega) / sin_omega
            coeff2 = np.sin(t * omega) / sin_omega
            v_interp = coeff1 * v1 + coeff2 * v2
            points.append(self._to_latlon(v_interp))

        return points, scaled_weights

    def pairs_within_distance(
        self, points: List[Tuple[float, float]], max_distance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # 1. Convert all (lat, lon) points to 3D Cartesian vectors
        vecs = (
            np.array([self._to_vector(lat, lon) for lat, lon in points]) * self.radius
        )

        # 2. Build the K-D Tree
        tree = cKDTree(vecs)

        # 3. Map geodesic max distance to Euclidean chord distance
        # d_chord = 2 * R * sin(d_geodesic / (2R))
        max_chord = 2.0 * self.radius * np.sin(max_distance / (2.0 * self.radius))

        # 4. Query the tree (returns a sparse DOK matrix)
        dok = tree.sparse_distance_matrix(tree, max_chord)
        coo = dok.tocoo()

        # 5. Convert chord distances back to geodesic distances
        # d_geodesic = 2 * R * arcsin(d_chord / (2R))
        ratio = np.clip(coo.data / (2.0 * self.radius), -1.0, 1.0)
        geo_dists = 2.0 * self.radius * np.arcsin(ratio)

        # Note: sparse_distance_matrix explicitly drops zero-distance pairs (the diagonal!)
        # We must manually inject the diagonal back in for distance = 0.0
        n = len(points)
        rows = np.concatenate([coo.row, np.arange(n)])
        cols = np.concatenate([coo.col, np.arange(n)])
        final_dists = np.concatenate([geo_dists, np.zeros(n)])

        return rows, cols, final_dists

    def with_degree(self, degree: int) -> Lebesgue:
        return Lebesgue(degree, radius=self.radius, grid=self.grid, extend=self.extend)

    def degree_transfer_operator(self, target_degree: int) -> LinearOperator:
        """
        Returns the transfer operator from this space to one with a different degree.

        This operator leverages the hierarchical nature of the 1D SH vector to
        efficiently truncate or zero-pad the coefficients.
        """
        codomain = self.with_degree(target_degree)

        def mapping(u: sh.SHGrid) -> sh.SHGrid:
            vec_in = self.to_components(u)
            target_size = (target_degree + 1) ** 2

            if target_size > vec_in.size:
                vec_out = np.pad(vec_in, (0, target_size - vec_in.size))
            else:
                vec_out = vec_in[:target_size]

            return codomain.from_components(vec_out)

        def adjoint_mapping(v: sh.SHGrid) -> sh.SHGrid:
            vec_in = codomain.to_components(v)
            target_size = (self.lmax + 1) ** 2

            if target_size > vec_in.size:
                vec_out = np.pad(vec_in, (0, target_size - vec_in.size))
            else:
                vec_out = vec_in[:target_size]

            return self.from_components(vec_out)

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def invariant_covariance_function(
        self, spectral_variances: np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:

        degree_variances = np.zeros(self.lmax + 1)
        for l in range(self.lmax + 1):
            idx = self.index_to_integer((l, 0))
            degree_variances[l] = spectral_variances[idx]

        # Spherical harmonic addition theorem coefficients:
        # (2l + 1) / (4 * pi * R^2)
        coeffs = (
            degree_variances
            * (2 * np.arange(self.lmax + 1) + 1)
            / (4 * np.pi * self.radius**2)
        )

        def cov_evaluator(distances: np.ndarray) -> np.ndarray:
            cos_theta = np.cos(distances / self.radius)
            return np.polynomial.legendre.legval(cos_theta, coeffs)

        return cov_evaluator

    def degree_multiplicity(self, degree: int) -> int:
        return 2 * degree + 1

    def representative_index(self, degree: int) -> Tuple[int, int]:
        return (degree, 0)

    # ------------------------------------------------------ #
    #                 Methods for HilbertSpace               #
    # ------------------------------------------------------ #

    def to_components(self, u: sh.SHGrid) -> np.ndarray:
        coeff = self.to_coefficients(u)
        return self._coefficient_to_component(coeff)

    def from_components(self, c: np.ndarray) -> sh.SHGrid:
        coeff = self._component_to_coefficients(c)
        return self.from_coefficients(coeff)

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

    def __eq__(self, other: object) -> bool:
        """
        Checks for mathematical equality with another Sobolev space on a sphere.

        Two spaces are considered equal if they are of the same type and have
        the same defining parameters (kmax, order, scale, and radius).
        """
        if not isinstance(other, Lebesgue):
            return NotImplemented

        return (
            self.lmax == other.lmax
            and self.radius == other.radius
            and self.grid == other.grid
            and self.extend == other.extend
        )

    def is_element(self, x: Any) -> bool:
        """
        Checks if an object is a valid element of the space.
        """
        if not isinstance(x, sh.SHGrid):
            return False
        if not x.lmax == self.lmax:
            return False
        if not x.grid == self.grid_type:
            return False
        if not x.extend == self.extend:
            return False
        return True

    # ------------------------------------------------------ #
    #                 Methods for HilbertModule              #
    # ------------------------------------------------------ #

    def vector_multiply(self, x1: sh.SHGrid, x2: sh.SHGrid) -> sh.SHGrid:
        """
        Computes the pointwise product of two functions.
        """
        return x1 * x2

    def vector_sqrt(self, x: sh.SHGrid) -> sh.SHGrid:
        """
        Returns the pointwise square root of a function.
        """
        y = x.copy()
        y.data = np.sqrt(x.data)
        return y

    # ------------------------------------------------------ #
    #                   Additional methods                   #
    # ------------------------------------------------------ #

    def to_coefficient_operator(self, lmax: int, /, *, lmin: int = 0):
        r"""
        Returns a LinearOperator mapping a function to its spherical harmonic coefficients.

        The operator maps an element of the Hilbert space to a vector in \(\mathbb{R}^k\).
        The coefficients in the output vector follow the native pyshtools ordering:
        ordered by degree $l$ (major), and for each degree, the order $m$ is sorted
        as $0, 1, \dots, l, -1, \dots, -l$.

        Args:
            lmax: The maximum spherical harmonic degree to include in the output.
            lmin: The minimum spherical harmonic degree to include. Defaults to 0.

        Returns:
            A LinearOperator mapping `SHGrid` -> `numpy.ndarray`.
        """
        vector_size = (lmax + 1) ** 2 - lmin**2
        codomain = EuclideanSpace(vector_size)

        def mapping(u: sh.SHGrid) -> np.ndarray:
            vec = self.to_components(u)
            target_size = (lmax + 1) ** 2

            if target_size > vec.size:
                vec = np.pad(vec, (0, target_size - vec.size))
            else:
                vec = vec[:target_size]

            # Truncate lower degrees if lmin > 0
            return vec[lmin**2 :] if lmin > 0 else vec

        def adjoint_mapping(data: np.ndarray) -> sh.SHGrid:
            # Pad missing lower degrees if lmin > 0
            vec = np.concatenate((np.zeros(lmin**2), data)) if lmin > 0 else data
            target_size = (self.lmax + 1) ** 2

            if target_size > vec.size:
                vec = np.pad(vec, (0, target_size - vec.size))
            else:
                vec = vec[:target_size]

            return self.from_components(vec) / self.radius**2

        return LinearOperator(self, codomain, mapping, adjoint_mapping=adjoint_mapping)

    def from_coefficient_operator(self, lmax: int, /, *, lmin: int = 0):
        r"""
        Returns a LinearOperator mapping a vector of coefficients to a function.

        The operator maps a vector in \(\mathbb{R}^k\) to an element of the Hilbert space.
        The input vector must follow the native pyshtools ordering: ordered by
        degree $l$ (major), and for each degree, the order $m$ is sorted as
        $0, 1, \dots, l, -1, \dots, -l$.

        Args:
            lmax: The maximum spherical harmonic degree expected in the input.
            lmin: The minimum spherical harmonic degree expected. Defaults to 0.

        Returns:
            A LinearOperator mapping `numpy.ndarray` -> `SHGrid`.
        """
        vector_size = (lmax + 1) ** 2 - lmin**2
        domain = EuclideanSpace(vector_size)

        def mapping(data: np.ndarray) -> sh.SHGrid:
            # Pad missing lower degrees if lmin > 0
            vec = np.concatenate((np.zeros(lmin**2), data)) if lmin > 0 else data
            target_size = (self.lmax + 1) ** 2

            if target_size > vec.size:
                vec = np.pad(vec, (0, target_size - vec.size))
            else:
                vec = vec[:target_size]

            return self.from_components(vec)

        def adjoint_mapping(u: sh.SHGrid) -> np.ndarray:
            vec = self.to_components(u)
            target_size = (lmax + 1) ** 2

            if target_size > vec.size:
                vec = np.pad(vec, (0, target_size - vec.size))
            else:
                vec = vec[:target_size]

            # Truncate lower degrees if lmin > 0
            vec = vec[lmin**2 :] if lmin > 0 else vec
            return vec * self.radius**2

        return LinearOperator(domain, self, mapping, adjoint_mapping=adjoint_mapping)

    # ------------------------------------------------------ #
    #                      Private methods                   #
    # ------------------------------------------------------ #

    def _coefficient_to_component(self, ulm: sh.SHCoeffs) -> np.ndarray:
        """Maps spherical harmonic coefficients to a component vector."""
        return sh.shio.SHCilmToVector(ulm.coeffs)

    def _component_to_coefficients(self, c: np.ndarray) -> sh.SHCoeffs:
        """Maps a component vector to spherical harmonic coefficients."""
        coeffs = sh.shio.SHVectorToCilm(c)
        return sh.SHCoeffs.from_array(
            coeffs, normalization=self.normalization, csphase=self.csphase
        )

    @staticmethod
    def _to_vector(lat: float, lon: float) -> np.ndarray:
        """Converts a latitude/longitude pair (in degrees) to a 3D unit vector."""
        lat_rad, lon_rad = np.radians(lat), np.radians(lon)
        return np.array(
            [
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ]
        )

    @staticmethod
    def _to_latlon(vec: np.ndarray) -> Tuple[float, float]:
        """Converts a 3D vector back to a latitude/longitude pair (in degrees)."""
        vec = vec / np.linalg.norm(vec)
        lat_rad = np.arcsin(vec[2])
        lon_rad = np.arctan2(vec[1], vec[0])
        return (np.degrees(lat_rad), np.degrees(lon_rad))


class Sobolev(SymmetricSobolevSpace):
    """
    Implementation of the Sobolev space Hˢ on a circle.
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
        safe: bool = True,
    ):
        """
        Args:
        lmax: Maximum degree for the expansions.
        order: The Sobolev order, controlling the smoothness of functions.
        scale: The Sobolev length-scale.
        radius: Radius of the sphere. Defaults to 1.
        grid: pyshtools grid type. Defaults to "DH"
        extend: If true longitudes wrap fully. Defaults to True.
        safe: If true, the class checks for mathematical correctness of operations
                  where possible.
        """

        lebesgue_space = Lebesgue(lmax, radius=radius, grid=grid, extend=extend)
        SymmetricSobolevSpace.__init__(self, lebesgue_space, order, scale, safe=safe)

    @staticmethod
    def from_sobolev_parameters(
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
        grid: str = "DH",
        rtol: float = 1e-8,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
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
            safe: If true, the class checks for mathematical correctness of operations
                  where possible.

        Returns:
            An instance of the Sobolev class with a calculated `lmax`.
        """
        if order <= 1.0:
            raise ValueError("This method is only applicable for orders > 1.0")

        summation = 1.0
        l = 0
        err = 1.0

        def sobolev_func(deg):
            return (1.0 + (scale / radius) ** 2 * deg * (deg + 1)) ** order

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
        return Sobolev(lmax, order, scale, radius=radius, grid=grid, safe=safe)

    @classmethod
    def from_covariance(
        cls,
        covariance_function: Callable[[float], float],
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
        grid: str = "DH",
        extend: bool = True,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space with a spherical harmonic truncation degree
        (`lmax`) automatically chosen to capture the expected energy of functions drawn from
        a specified prior measure.

        This factory method calculates the expected squared norm (energy) of a random field
        whose spectral variances are defined by the provided `covariance_function`, accounting
        for the Sobolev mass-weighting factor. It iteratively adds higher spherical harmonic
        degrees until the relative contribution of the next degree drops below the tolerance.

        Args:
            covariance_function: A callable mapping a Laplacian eigenvalue to its spectral variance.
            order: The Sobolev order, controlling the smoothness of functions.
            scale: The Sobolev length-scale.
            radius: The radius of the sphere. Defaults to 1.0.
            grid: The pyshtools spatial grid format to use. Defaults to "DH".
            extend: If True, the spatial grid includes both 0 and 360-degree longitudes.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree. If convergence
                is not reached by this degree, the search terminates and returns this value.
            power_of_two: If True, rounds the resulting `lmax` up to the nearest power of two.
            safe: If True, enables mathematical correctness checks during operations.

        Returns:
            Sobolev: A fully instantiated Sobolev space on the sphere with the optimal `lmax`.

        Raises:
            RuntimeError: If the energy sequence fails to converge within 100,000 iterations
                and no `max_degree` is specified.
        """
        dummy_space = cls(
            max(1, min_degree),
            order,
            scale,
            radius=radius,
            grid=grid,
            extend=extend,
            safe=safe,
        )

        optimal_degree = dummy_space.estimate_truncation_degree(
            covariance_function, rtol=rtol, min_degree=min_degree, max_degree=max_degree
        )

        if power_of_two:
            n = int(np.log2(optimal_degree))
            optimal_degree = 2 ** (n + 1)

        return cls(
            optimal_degree,
            order,
            scale,
            radius=radius,
            grid=grid,
            extend=extend,
            safe=safe,
        )

    @classmethod
    def from_heat_kernel_prior(
        cls,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
        grid: str = "DH",
        extend: bool = True,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space on the sphere, tuned to the expected
        energy of a Heat Kernel prior measure.

        Args:
            kernel_scale: The length-scale parameter of the heat kernel covariance.
            order: The Sobolev order defining the function space.
            scale: The Sobolev length-scale defining the function space.
            radius: The radius of the sphere. Defaults to 1.0.
            grid: The pyshtools spatial grid format to use. Defaults to "DH".
            extend: If True, the spatial grid includes both 0 and 360-degree longitudes.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `lmax` up to the nearest power of two.
            safe: If True, enables mathematical correctness checks during operations.
        """
        return cls.from_covariance(
            cls.heat_kernel(kernel_scale),
            order,
            scale,
            radius=radius,
            grid=grid,
            extend=extend,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
            safe=safe,
        )

    @classmethod
    def from_sobolev_kernel_prior(
        cls,
        kernel_order: float,
        kernel_scale: float,
        order: float,
        scale: float,
        /,
        *,
        radius: float = 1.0,
        grid: str = "DH",
        extend: bool = True,
        rtol: float = 1e-6,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        power_of_two: bool = False,
        safe: bool = True,
    ) -> Sobolev:
        """
        Creates an instance of the Sobolev space on the sphere, tuned to the expected
        energy of a Sobolev-type prior measure.

        Args:
            kernel_order: The smoothness order of the Sobolev prior measure.
            kernel_scale: The length-scale parameter of the Sobolev prior measure.
            order: The Sobolev order defining the function space.
            scale: The Sobolev length-scale defining the function space.
            radius: The radius of the sphere. Defaults to 1.0.
            grid: The pyshtools spatial grid format to use. Defaults to "DH".
            extend: If True, the spatial grid includes both 0 and 360-degree longitudes.
            rtol: The relative tolerance for the energy truncation. Defaults to 1e-6.
            min_degree: The absolute minimum truncation degree to return. Defaults to 0.
            max_degree: An optional safety ceiling for the truncation degree.
            power_of_two: If True, rounds the resulting `lmax` up to the nearest power of two.
            safe: If True, enables mathematical correctness checks during operations.
        """
        return cls.from_covariance(
            cls.sobolev_kernel(kernel_order, kernel_scale),
            order,
            scale,
            radius=radius,
            grid=grid,
            extend=extend,
            rtol=rtol,
            min_degree=min_degree,
            max_degree=max_degree,
            power_of_two=power_of_two,
            safe=safe,
        )

    # ----------------------------------------- #
    #                 Properties                #
    # ----------------------------------------- #

    @property
    def lmax(self) -> int:
        """The maximum spherical harmonic truncation degree."""
        return self.underlying_space.lmax

    @property
    def radius(self) -> float:
        """The radius of the sphere."""
        return self.underlying_space.radius

    @property
    def grid(self) -> str:
        """The `pyshtools` grid type used for spatial representations."""
        return self.underlying_space.grid

    @property
    def sampling(self) -> int:
        """The sampling factor used for spatial representations."""
        return self.underlying_space.sampling

    @property
    def extend(self) -> bool:
        """True if the spatial grid includes both 0 and 360-degree longitudes."""
        return self.underlying_space.extend

    @property
    def normalization(self) -> str:
        """The spherical harmonic normalization convention used ('ortho')."""
        return self.underlying_space.normalization

    @property
    def csphase(self) -> int:
        """The Condon-Shortley phase convention used (1)."""
        return self.underlying_space.csphase

    @property
    def grid_type(self) -> str:
        """
        Returns the pyshtools grid type.
        """
        return self.underlying_space.grid_type

    # -------------------------------------------------- #
    #                   Public methods                   #
    # -------------------------------------------------- #

    def with_order(self, order: float) -> Sobolev:
        return Sobolev(
            self.lmax,
            order,
            self.scale,
            radius=self.radius,
            grid=self.grid,
            extend=self.extend,
        )

    def with_degree(self, degree: int) -> Sobolev:
        return Sobolev(
            degree,
            self.order,
            self.scale,
            radius=self.radius,
            grid=self.grid,
            extend=self.extend,
        )

    def project_function(self, f: Callable[[(float, float)], float]) -> np.ndarray:
        """
        Returns an element of the space by projecting a given function.

        Args:
            f: A function that takes a point `(lat, lon)` and returns a value.
        """
        return self.underlying_space.project_function(f)

    def to_coefficients(self, u: sh.SHGrid) -> sh.SHCoeffs:
        """Maps a function vector to its spherical harmonic coefficients."""
        return self.underlying_space.to_coefficients(u)

    def from_coefficients(self, ulm: sh.SHCoeffs) -> sh.SHGrid:
        """Maps spherical harmonic coefficients to a function vector."""
        return self.underlying_space.from_coefficients(ulm)

    def sample_power_measure(
        self,
        measure,
        n_samples,
        /,
        *,
        lmin=None,
        lmax=None,
        parallel: bool = False,
        n_jobs: int = -1,
    ):
        """
        Takes in a Gaussian measure on the space, draws n_samples from
        and returns samples for the spherical harmonic power at degrees in
        the indicated range.
        """

        return self.underlying_space.sample_power_measure(
            measure, n_samples, lmin=lmin, lmax=lmax, parallel=parallel, n_jobs=n_jobs
        )

    def ax(self, a: float, x: sh.SHGrid) -> None:
        self.underlying_space.ax(a, x)

    def axpy(self, a: float, x: sh.SHGrid, y: sh.SHGrid) -> None:
        self.underlying_space.axpy(a, x, y)

    def to_coefficient_operator(self, lmax: int, /, *, lmin: int = 0):
        r"""
        Returns a LinearOperator mapping a function to its spherical harmonic coefficients.

        The operator maps an element of the Hilbert space to a vector in $\mathbb{R}^k$.
        The coefficients in the output vector are ordered by degree $l$ (major)
        and order $m$ (minor), from $-l$ to $+l$.

        **Ordering:**

        .. math::
            u = [u_{0,0}, \quad u_{1,-1}, u_{1,0}, u_{1,1}, \quad u_{2,-2}, \dots, u_{2,2}, \quad \dots]

        (assuming `lmin=0`).

        Args:
            lmax: The maximum spherical harmonic degree to include in the output.
            lmin: The minimum spherical harmonic degree to include. Defaults to 0.

        Returns:
            A LinearOperator mapping `SHGrid` -> `numpy.ndarray`.
        """

        l2_operator = self.underlying_space.to_coefficient_operator(lmax, lmin=lmin)

        return LinearOperator.from_formal_adjoint(
            self, l2_operator.codomain, l2_operator
        )

    def from_coefficient_operator(self, lmax: int, /, *, lmin: int = 0):
        r"""
        Returns a LinearOperator mapping a vector of coefficients to a function.

        The operator maps a vector in $\mathbb{R}^k$ to an element of the Hilbert space.
        The input vector must follow the standard $l$-major, $m$-minor ordering.

        **Ordering:**

        .. math::
            v = [u_{0,0}, \quad u_{1,-1}, u_{1,0}, u_{1,1}, \quad u_{2,-2}, \dots, u_{2,2}, \quad \dots]

        (assuming `lmin=0`).

        Args:
            lmax: The maximum spherical harmonic degree expected in the input.
            lmin: The minimum spherical harmonic degree expected. Defaults to 0.

        Returns:
            A LinearOperator mapping `numpy.ndarray` -> `SHGrid`.
        """

        l2_operator = self.underlying_space.from_coefficient_operator(lmax, lmin=lmin)

        return LinearOperator.from_formal_adjoint(l2_operator.domain, self, l2_operator)


# -------------------------------------------------- #
#             Associated plotting methods            #
# -------------------------------------------------- #


def _get_or_create_geoaxes(
    projection: Optional[Projection] = None,
) -> Tuple[GeoAxes, bool]:
    """
    Safely retrieves the current GeoAxes if one exists, or creates a fresh figure
    and GeoAxes to prevent overlapping with incompatible Cartesian plots.

    Returns:
        A tuple of (ax, is_new) where `is_new` is True if a fresh canvas was created.
    """
    if projection is None:
        projection = ccrs.PlateCarree()

    # 1. Check if any figures exist, and if the current figure has any axes
    if plt.get_fignums() and plt.gcf().axes:
        current_ax = plt.gca()
        # 2. Test if the active axis is a Cartopy GeoAxes
        if isinstance(current_ax, GeoAxes):
            return current_ax, False

    # 3. If no valid GeoAxes was found, safely spin up a fresh figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    return ax, True


def plot(
    u: SHGrid,
    /,
    *,
    ax: Optional[GeoAxes] = None,
    projection: Optional[Projection] = None,
    contour: bool = False,
    cmap: str = "RdBu",
    coasts: bool = False,
    rivers: bool = False,
    borders: bool = False,
    map_extent: Optional[List[float]] = None,
    gridlines: bool = True,
    symmetric: bool = False,
    contour_lines: bool = False,
    contour_lines_kwargs: Optional[dict] = None,
    num_levels: int = 10,
    **kwargs,
) -> Tuple[GeoAxes, Any]:
    """
    Creates a high-quality map plot of a spherical harmonic function using Cartopy.

    Args:
        u: The scalar field to be plotted, evaluated on a spatial grid.
        ax: An existing Cartopy GeoAxes object. If None, creates a new one.
        projection: A `cartopy.crs` projection instance defining the map view.
            Defaults to `ccrs.PlateCarree()`.
        contour: If True, renders the field as a filled contour plot (`contourf`).
            If False, renders it as a pseudo-color mesh (`pcolormesh`). Defaults to False.
        cmap: The Matplotlib colormap string or object to use. Defaults to "RdBu".
        coasts: If True, overlays high-resolution coastlines. Defaults to False.
        rivers: If True, overlays major river systems. Defaults to False.
        borders: If True, overlays international country borders. Defaults to False.
        map_extent: A list `[lon_min, lon_max, lat_min, lat_max]` limiting the view
            extent of the map. Defaults to None (global view).
        gridlines: If True, draws latitude and longitude gridlines with labels.
            Defaults to True.
        symmetric: If True, dynamically centers the color scale symmetrically around
            zero (e.g., from -max to +max). Defaults to False.
        contour_lines: If True, overlays solid contour lines on top of the base plot.
            Defaults to False.
        contour_lines_kwargs: A dictionary of keyword arguments passed to `ax.contour`.
        num_levels: The number of color levels to generate automatically. Defaults to 10.
        **kwargs: Additional keyword arguments forwarded to `ax.contourf` or `ax.pcolormesh`.

    Returns:
        A tuple `(ax, im)` containing:
            - ax: The Cartopy GeoAxes object.
            - im: The rendered image object (either a QuadMesh or ContourSet).
    """
    if ax is None:
        ax, _ = _get_or_create_geoaxes(projection)

    lons = u.lons()
    lats = u.lats()

    if map_extent is not None:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    if coasts:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    if rivers:
        ax.add_feature(cfeature.RIVERS, linewidth=0.8)
    if borders:
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    # Pop gridline intervals safely BEFORE passing kwargs to plot functions
    lat_interval = kwargs.pop("lat_interval", 30)
    lon_interval = kwargs.pop("lon_interval", 30)

    kwargs.setdefault("cmap", cmap)
    if symmetric:
        data_max = 1.2 * np.nanmax(np.abs(u.data))
        kwargs.setdefault("vmin", -data_max)
        kwargs.setdefault("vmax", data_max)

    if "levels" in kwargs:
        levels = kwargs.pop("levels")
    else:
        vmin = kwargs.get("vmin", np.nanmin(u.data))
        vmax = kwargs.get("vmax", np.nanmax(u.data))
        levels = np.linspace(vmin, vmax, num_levels)

    im: Any
    if contour:
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)
        im = ax.contourf(
            lons, lats, u.data, transform=ccrs.PlateCarree(), levels=levels, **kwargs
        )
    else:
        im = ax.pcolormesh(lons, lats, u.data, transform=ccrs.PlateCarree(), **kwargs)

    if contour_lines:
        cl_kwargs = contour_lines_kwargs if contour_lines_kwargs is not None else {}
        cl_kwargs.setdefault("colors", "k")
        cl_kwargs.setdefault("linewidths", 0.5)
        ax.contour(
            lons, lats, u.data, transform=ccrs.PlateCarree(), levels=levels, **cl_kwargs
        )

    if gridlines:
        gl = ax.gridlines(
            linestyle="--", draw_labels=True, dms=True, x_inline=False, y_inline=False
        )
        gl.xlocator = mticker.MultipleLocator(lon_interval)
        gl.ylocator = mticker.MultipleLocator(lat_interval)
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

    return ax, im


def plot_geodesic(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    /,
    *,
    ax: Optional[GeoAxes] = None,
    n_points: int = 50,
    **kwargs,
) -> GeoAxes:
    """
    Plots a geodesic (great-circle) curve between two points on the sphere.

    Args:
        p1: The starting coordinate as a tuple `(latitude, longitude)` in degrees.
        p2: The ending coordinate as a tuple `(latitude, longitude)` in degrees.
        ax: An existing Cartopy GeoAxes object. If None, creates a new one.
        n_points: The number of points to use for interpolation (kept for compatibility).
        **kwargs: Keyword arguments passed directly to `ax.plot`.

    Returns:
        The Cartopy GeoAxes object.
    """
    if ax is None:
        ax, _ = _get_or_create_geoaxes(ccrs.PlateCarree())

    kwargs.setdefault("color", "black")
    kwargs.setdefault("linewidth", 2)

    lat1, lon1 = p1
    lat2, lon2 = p2
    ax.plot([lon1, lon2], [lat1, lat2], transform=ccrs.Geodetic(), **kwargs)

    return ax


def plot_geodesic_network(
    paths: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    /,
    *,
    ax: Optional[GeoAxes] = None,
    n_points: int = 50,
    **kwargs,
) -> GeoAxes:
    """
    Plots a network of intersecting geodesic paths onto a Cartopy map.

    Args:
        paths: A list of point pairs defining the network.
        ax: An existing Cartopy GeoAxes object. If None, creates a new global view.
        n_points: The number of interpolation points per curve.
        **kwargs: Default styling arguments applied to the geodesic lines.

    Returns:
        The Cartopy GeoAxes object.
    """
    if ax is None:
        ax, is_new = _get_or_create_geoaxes(ccrs.PlateCarree())
        # Only apply global extent and coastlines if we spawned a brand-new map canvas
        if is_new:
            ax.set_global()
            ax.coastlines()

    kwargs.setdefault("color", "black")
    kwargs.setdefault("linewidth", 0.8)
    kwargs.setdefault("alpha", 0.5)

    for p1, p2 in paths:
        plot_geodesic(p1, p2, ax=ax, n_points=n_points, **kwargs)

    sources = list(set([tuple(p[0]) for p in paths]))
    receivers = list(set([tuple(p[1]) for p in paths]))

    src_lats, src_lons = zip(*sources)
    rec_lats, rec_lons = zip(*receivers)

    src_style = kwargs.pop("source_kwargs", {})
    src_style.setdefault("marker", "*")
    src_style.setdefault("color", "gold")
    src_style.setdefault("s", 150)
    src_style.setdefault("edgecolor", "black")
    src_style.setdefault("zorder", 5)
    ax.scatter(src_lons, src_lats, transform=ccrs.Geodetic(), **src_style)

    rec_style = kwargs.pop("receiver_kwargs", {})
    rec_style.setdefault("marker", "o")
    rec_style.setdefault("color", "red")
    rec_style.setdefault("s", 50)
    rec_style.setdefault("edgecolor", "white")
    rec_style.setdefault("zorder", 5)
    ax.scatter(rec_lons, rec_lats, transform=ccrs.Geodetic(), **rec_style)

    return ax
