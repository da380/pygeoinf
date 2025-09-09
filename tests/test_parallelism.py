"""
Tests for parallel computing features across different modules.

This suite verifies that parallel implementations produce results identical to
their serial counterparts.
"""

import pytest
import numpy as np
import pygeoinf as inf


class TestLinearOperatorParallelism:
    """Tests parallel dense matrix construction for LinearOperator."""

    @pytest.fixture(scope="class")
    def operator(self) -> inf.LinearOperator:
        """Provides a non-trivial operator for parallelism tests."""
        space = inf.symmetric_space.sphere.Sobolev(8, 2.0, 0.2)
        return space.invariant_automorphism(lambda eig: 1.0 / (1.0 + 0.1 * eig))

    @pytest.mark.parametrize("galerkin", [True, False])
    def test_matrix_correctness(self, operator: inf.LinearOperator, galerkin: bool):
        """
        Ensures parallel matrix computation gives the same result as serial.
        """
        # 1. Compute the matrix using the trusted serial implementation.
        serial_matrix = operator.matrix(dense=True, galerkin=galerkin, parallel=False)

        # 2. Compute the matrix using the parallel implementation.
        parallel_matrix = operator.matrix(dense=True, galerkin=galerkin, parallel=True)

        # 3. Assert that the two matrices are numerically identical.
        assert np.allclose(serial_matrix, parallel_matrix)


class TestLinearFormParallelism:
    """Tests parallel component computation for LinearForm."""

    @pytest.fixture(scope="class")
    def space(self) -> inf.HilbertSpace:
        """Provides a non-trivial Hilbert space for the form."""
        return inf.symmetric_space.sphere.Lebesgue(10)

    @pytest.fixture(scope="class")
    def mapping(self, space: inf.HilbertSpace) -> callable:
        """Defines a mapping for the LinearForm."""
        # A non-trivial mapping: inner product with a fixed random vector.
        fixed_vec = space.random()
        return lambda x: space.inner_product(x, fixed_vec)

    def test_component_computation(self, space: inf.HilbertSpace, mapping: callable):
        """
        Ensures parallel component computation gives the same result as serial.
        """
        # 1. Compute components using the serial implementation.
        form_serial = inf.LinearForm(space, mapping=mapping, parallel=False)

        # 2. Compute components using the parallel implementation.
        form_parallel = inf.LinearForm(space, mapping=mapping, parallel=True)

        # 3. Assert that the two component vectors are numerically identical.
        assert np.allclose(form_serial.components, form_parallel.components)


class TestGaussianMeasureParallelism:
    """Tests parallel low-rank approximation for GaussianMeasure."""

    @pytest.fixture(scope="class")
    def gaussian_measure(self) -> inf.GaussianMeasure:
        """Provides a Gaussian measure with a full-rank covariance."""
        space = inf.EuclideanSpace(dim=30)
        # Create a random symmetric positive-definite matrix
        A = np.random.randn(30, 30)
        cov_matrix = A.T @ A + 0.1 * np.eye(30)
        return inf.GaussianMeasure.from_covariance_matrix(space, cov_matrix)

    def test_low_rank_approximation(self, gaussian_measure: inf.GaussianMeasure):
        """
        Ensures parallel low-rank approximation gives the same result as serial.
        """
        rank_estimate = 15
        # Set a seed because the randomized algorithm's result depends on it.
        # This ensures both serial and parallel runs use the same random numbers.
        np.random.seed(42)

        # 1. Compute the low-rank approximation serially.
        approx_serial = gaussian_measure.low_rank_approximation(
            rank_estimate, method="fixed", parallel=False
        )

        # Reset the seed to ensure the parallel run starts from the same state.
        np.random.seed(42)

        # 2. Compute the low-rank approximation in parallel.
        approx_parallel = gaussian_measure.low_rank_approximation(
            rank_estimate, method="fixed", parallel=True
        )

        # 3. Assert that the resulting covariance operators are identical.
        cov_serial = approx_serial.covariance.matrix(dense=True, galerkin=True)
        cov_parallel = approx_parallel.covariance.matrix(dense=True, galerkin=True)

        assert np.allclose(cov_serial, cov_parallel)
