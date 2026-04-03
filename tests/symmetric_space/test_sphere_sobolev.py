"""
Tests for the Sobolev function space Hˢ on a sphere.
"""

import pytest
import numpy as np
import pyshtools as sh

from pygeoinf.symmetric_space.sphere import Sobolev


@pytest.mark.parametrize("lmax, radius, grid", [(8, 1.0, "DH"), (16, 6371.0, "GLQ")])
def test_sobolev_axioms(lmax: int, radius: float, grid: str):
    """
    Verifies that the Sobolev space instance satisfies all Hilbert space axioms
    by calling its internal self-check method.
    """
    # Sobolev parameters can be fixed for the generic axiom checks
    space = Sobolev(lmax, 2.0, 0.5, radius=radius, grid=grid)
    space.check(n_checks=5)


class TestSphereSobolevSpecifics:
    """
    Tests functionalities that are specific to the Sobolev nature of the space,
    particularly those dependent on order and scale.
    """

    @pytest.fixture(params=[(1.5, 0.5), (2.5, 0.8)])
    def sobolev_space(self, request) -> Sobolev:
        """Provides a Sobolev space with varying order and scale."""
        order, scale = request.param
        return Sobolev(16, order, scale, radius=1.0, grid="DH")

    def test_dirac_functional_property(self, sobolev_space: Sobolev):
        """
        Tests that applying the Dirac functional δ_p to a function f
        correctly evaluates the function at point p, i.e., <δ_p, f> = f(p).
        """
        space = sobolev_space
        test_point = (30.0, 60.0)  # 30°N, 60°E

        # A combination of Y(1,0) and Y(2,2), which is analytic everywhere.
        def test_func(point: tuple[float, float]) -> float:
            lat_rad, lon_rad = np.deg2rad(point[0]), np.deg2rad(point[1])
            return 2 * np.sin(lat_rad) + np.cos(lat_rad) ** 2 * np.cos(2 * lon_rad)

        dirac_functional = space.dirac(test_point)
        func_vector = space.project_function(test_func)

        functional_evaluation = space.duality_product(dirac_functional, func_vector)
        direct_evaluation = test_func(test_point)

        assert np.isclose(functional_evaluation, direct_evaluation, rtol=1e-2)

    def test_eigenfunction_norm(self, sobolev_space: Sobolev):
        """
        Tests that the Sobolev norm of a spherical harmonic eigenfunction
        matches its analytically calculated value.
        """
        space = sobolev_space
        l, m = 5, 2  # A non-trivial spherical harmonic

        coeffs = sh.SHCoeffs.from_zeros(space.lmax, normalization="ortho", csphase=1)
        coeffs.set_coeffs(1.0, l, m)
        harmonic_grid = space.from_coefficients(coeffs)

        numerical_norm = space.norm(harmonic_grid)

        eigenvalue = space.laplacian_eigenvalue((l, m))
        scaling_factor = np.sqrt(space.sobolev_function(eigenvalue))
        analytical_norm = space.radius * scaling_factor

        assert np.isclose(numerical_norm, analytical_norm)

    def test_sobolev_coefficient_operators_axioms(self, sobolev_space: Sobolev):
        """
        Verifies that the Sobolev coefficient operators satisfy all
        LinearOperator axioms using the standard check() method.
        """
        lmax = sobolev_space.lmax

        op_to = sobolev_space.to_coefficient_operator(lmax)
        op_to.check(n_checks=5)

        op_from = sobolev_space.from_coefficient_operator(lmax)
        op_from.check(n_checks=5)

    def test_sobolev_coefficient_mapping(self, sobolev_space: Sobolev):
        """
        Tests the coefficient operators specifically within the Sobolev context
        to ensure they map to the correct underlying coefficients using the
        native pyshtools vector ordering.
        """
        lmax = sobolev_space.lmax

        # Generate a known coefficient vector
        vector_size = (lmax + 1) ** 2
        vec_in = np.zeros(vector_size)

        # Set a specific coefficient (e.g., l=2, m=0) to 1.0
        # pyshtools ordering: degree l major; order m is [0, 1, ..., l, -1, ..., -l]
        target_idx = 4
        vec_in[target_idx] = 1.0

        op_from = sobolev_space.from_coefficient_operator(lmax)
        op_to = sobolev_space.to_coefficient_operator(lmax)

        # Map vector to Sobolev field and back
        u = op_from(vec_in)
        vec_out = op_to(u)

        assert np.allclose(vec_in, vec_out)

        # Verify directly against pyshtools to ensure semantic correctness
        coeffs = sobolev_space.to_coefficients(u)
        assert np.isclose(coeffs.coeffs[0, 2, 0], 1.0)

    def test_degree_and_with_degree(self, sobolev_space: Sobolev):
        """Tests the unified degree property and the with_degree factory."""
        space = sobolev_space
        assert space.degree == space.lmax

        target_degree = space.lmax + 4
        new_space = space.with_degree(target_degree)

        assert new_space.degree == target_degree
        assert new_space.radius == space.radius
        assert new_space.order == space.order
        assert new_space.scale == space.scale

    def test_degree_transfer_operator(self, sobolev_space: Sobolev):
        """Tests the degree transfer operator's axioms in a mass-weighted space."""
        op_up = sobolev_space.degree_transfer_operator(sobolev_space.lmax + 4)
        op_up.check(n_checks=5)

    def test_point_evaluation_matrix_free_equivalence(self, sobolev_space: Sobolev):
        """
        Tests that the dense, matrix-free serial, and matrix-free parallel
        implementations of the point evaluation operator produce mathematically
        identical results and satisfy the LinearOperator axioms.
        """
        points = [(10.0, 20.0), (-45.0, 100.0), (80.0, 300.0), (0.0, 0.0), (5.0, 5.0)]

        op_dense = sobolev_space.point_evaluation_operator(points, matrix_free=False)
        op_free_serial = sobolev_space.point_evaluation_operator(
            points, matrix_free=True, parallel=False
        )
        op_free_parallel = sobolev_space.point_evaluation_operator(
            points, matrix_free=True, parallel=True, n_jobs=2
        )

        # 1. Verify axioms for all implementations
        op_dense.check(n_checks=3)
        op_free_serial.check(n_checks=3)
        op_free_parallel.check(n_checks=3)

        # 2. Check forward mapping equivalence
        u = sobolev_space.random()
        val_dense = op_dense(u)

        assert np.allclose(val_dense, op_free_serial(u))
        assert np.allclose(val_dense, op_free_parallel(u))

        # 3. Check adjoint mapping equivalence
        y = np.random.randn(len(points))
        u_adj_dense = op_dense.adjoint(y)

        assert np.allclose(u_adj_dense.data, op_free_serial.adjoint(y).data)
        assert np.allclose(u_adj_dense.data, op_free_parallel.adjoint(y).data)

    def test_path_average_matrix_free_equivalence(self, sobolev_space: Sobolev):
        """
        Tests that the dense, matrix-free serial, matrix-free parallel, and
        lazy quadrature implementations of the path average operator produce
        mathematically identical results and satisfy the LinearOperator axioms.
        """
        paths = [
            ((0.0, 0.0), (10.0, 10.0)),
            ((-20.0, 50.0), (40.0, -50.0)),
            ((5.0, 100.0), (-5.0, 120.0)),
        ]

        op_dense = sobolev_space.path_average_operator(
            paths, n_points=5, matrix_free=False
        )
        op_free_serial = sobolev_space.path_average_operator(
            paths, n_points=5, matrix_free=True, parallel=False, lazy_quadrature=False
        )
        op_free_parallel = sobolev_space.path_average_operator(
            paths,
            n_points=5,
            matrix_free=True,
            parallel=True,
            n_jobs=2,
            lazy_quadrature=False,
        )
        op_lazy_serial = sobolev_space.path_average_operator(
            paths, n_points=5, matrix_free=True, parallel=False, lazy_quadrature=True
        )
        op_lazy_parallel = sobolev_space.path_average_operator(
            paths,
            n_points=5,
            matrix_free=True,
            parallel=True,
            n_jobs=2,
            lazy_quadrature=True,
        )

        operators = [
            op_dense,
            op_free_serial,
            op_free_parallel,
            op_lazy_serial,
            op_lazy_parallel,
        ]

        # 1. Verify axioms for all implementations
        for op in operators:
            op.check(n_checks=3)

        # 2. Check forward mapping equivalence
        u = sobolev_space.random()
        val_dense = op_dense(u)

        for op in operators[1:]:
            assert np.allclose(val_dense, op(u))

        # 3. Check adjoint mapping equivalence
        y = np.random.randn(len(paths))
        u_adj_dense = op_dense.adjoint(y)

        for op in operators[1:]:
            assert np.allclose(u_adj_dense.data, op.adjoint(y).data)


def test_factory_methods():
    """Tests the automatic truncation degree factories for Sobolev spaces on a sphere."""
    space = Sobolev.from_sobolev_kernel_prior(
        4.0, 0.1, 1.0, 0.5, radius=1.0, min_degree=4, power_of_two=True
    )
    assert isinstance(space, Sobolev)
    assert space.order == 1.0
    assert space.scale == 0.5
    assert space.radius == 1.0
    assert space.lmax >= 4
    assert (space.lmax & (space.lmax - 1)) == 0
