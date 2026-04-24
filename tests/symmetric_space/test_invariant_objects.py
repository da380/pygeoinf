"""
Tests for the invariant operators and measures on symmetric spaces.
"""

import pytest
import numpy as np

from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.linear_solvers import MinResSolver

from pygeoinf.symmetric_space.circle import Lebesgue
from pygeoinf.symmetric_space import (
    InvariantLinearAutomorphism,
    InvariantGaussianMeasure,
)
from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.linear_operators import LinearOperator



@pytest.fixture
def space() -> Lebesgue:
    """Provides a simple symmetric space for testing invariant objects."""
    # A small circle space is fast and perfectly symmetric
    return Lebesgue(8, radius=1.0)


class TestInvariantLinearAutomorphism:
    """Tests the type preservation and math of invariant operators."""

    def test_instantiation_and_properties(self, space: Lebesgue):
        op = space.invariant_automorphism(lambda k: 1.0 / (1.0 + k))

        assert isinstance(op, InvariantLinearAutomorphism)
        assert len(op.eigenvalues) == space.dim
        assert op.trace > 0

    def test_algebra_type_preservation(self, space: Lebesgue):
        op1 = space.invariant_automorphism(lambda k: 2.0)
        op2 = space.invariant_automorphism(lambda k: 3.0)

        # Operations between invariant operators must return invariant operators
        assert isinstance(op1 + op2, InvariantLinearAutomorphism)
        assert isinstance(op1 - op2, InvariantLinearAutomorphism)
        assert isinstance(op1 @ op2, InvariantLinearAutomorphism)
        assert isinstance(op1 * 2.0, InvariantLinearAutomorphism)
        assert isinstance(-op1, InvariantLinearAutomorphism)
        assert isinstance(op1.inverse, InvariantLinearAutomorphism)

        # Values should be exact (diagonal algebra)
        assert np.allclose((op1 @ op2).eigenvalues, 6.0)
        assert np.allclose((op1 + op2).eigenvalues, 5.0)

    def test_fallback_to_linear_operator(self, space: Lebesgue):
        op_inv = space.invariant_automorphism(lambda k: 1.0)
        op_generic = LinearOperator.from_matrix(
            space, space, np.eye(space.dim), galerkin=True
        )

        # Mixing with a generic LinearOperator should fall back gracefully
        assert type(op_inv + op_generic) is LinearOperator
        assert type(op_inv @ op_generic) is LinearOperator


class TestInvariantGaussianMeasure:
    """Tests the type preservation, optimization, and scaling of invariant measures."""

    def test_instantiation(self, space: Lebesgue):
        measure = space.invariant_gaussian_measure(lambda k: 1.0 / (1.0 + k))
        assert isinstance(measure, InvariantGaussianMeasure)
        assert isinstance(measure.covariance, InvariantLinearAutomorphism)

        # Internally, zero expectation should be None
        assert measure.has_zero_expectation is True
        assert measure._expectation is None

    def test_automatic_inverse_covariance(self, space: Lebesgue):
        """Tests that the inverse covariance is correctly set or skipped."""
        # 1. Strictly positive variances -> Inverse should be set automatically
        measure_pos = space.invariant_gaussian_measure(lambda k: 1.0 / (1.0 + k))
        assert measure_pos.inverse_covariance_set is True
        assert isinstance(measure_pos.inverse_covariance, InvariantLinearAutomorphism)

        expected_inv_evals = 1.0 / measure_pos.spectral_variances
        assert np.allclose(
            measure_pos.inverse_covariance.eigenvalues, expected_inv_evals
        )

        # 2. Contains a zero variance -> Inverse should NOT be set
        zero_vars = np.ones(space.dim)
        zero_vars[0] = 0.0
        measure_zero = InvariantGaussianMeasure(space, zero_vars)

        assert measure_zero.inverse_covariance_set is False
        with pytest.raises(AttributeError):
            _ = measure_zero.inverse_covariance

    def test_algebra_type_preservation(self, space: Lebesgue):
        m1 = space.invariant_gaussian_measure(lambda k: 1.0)
        m2 = space.invariant_gaussian_measure(lambda k: 2.0)

        assert isinstance(m1 + m2, InvariantGaussianMeasure)
        assert isinstance(m1 - m2, InvariantGaussianMeasure)
        assert isinstance(m1 * 2.0, InvariantGaussianMeasure)
        assert isinstance(-m1, InvariantGaussianMeasure)

    def test_zero_expectation_type_preservation(self, space: Lebesgue):
        exp = space.random()
        m1 = space.invariant_gaussian_measure(lambda k: 1.0, expectation=exp)

        assert not m1.has_zero_expectation
        m1_zeroed = m1.zero_expectation()

        assert isinstance(m1_zeroed, InvariantGaussianMeasure)
        assert m1_zeroed.has_zero_expectation

    def test_rescale_and_factories(self, space: Lebesgue):
        """Specifically tests the factory methods that caused the previous bug."""
        m_scaled = space.norm_scaled_heat_kernel_gaussian_measure(0.1, std=2.0)

        assert isinstance(m_scaled, InvariantGaussianMeasure)
        # Verify the scaling actually worked
        samples = m_scaled.samples(2000)
        sample_norms = [space.norm(s) for s in samples]
        # E[||x||^2] = std^2 = 4.0. The mean of squared norms should be close to 4.0
        mean_sq_norm = np.mean([n**2 for n in sample_norms])
        assert np.isclose(mean_sq_norm, 4.0, rtol=0.1)

    def test_affine_mapping_type_preservation(self, space: Lebesgue):
        measure = space.invariant_gaussian_measure(lambda k: 1.0)

        # 1. Map with Invariant Automorphism -> Should stay Invariant
        op_inv = space.invariant_automorphism(lambda k: 2.0)
        m_mapped_inv = measure.affine_mapping(operator=op_inv)
        assert isinstance(m_mapped_inv, InvariantGaussianMeasure)

        # 2. Map with generic LinearOperator -> Must fall back to GaussianMeasure
        op_gen = LinearOperator.from_matrix(
            space, space, np.eye(space.dim) * 2.0, galerkin=True
        )
        m_mapped_gen = measure.affine_mapping(operator=op_gen)
        assert type(m_mapped_gen) is GaussianMeasure

    def test_kl_divergence_fast_vs_dense(self, space: Lebesgue):
        """
        Ensures the O(N) invariant KL divergence calculation exactly matches
        the dense matrix fallback from the base GaussianMeasure class.
        """
        # P = N(mu_p, C_p)
        mu_p = space.random()
        measure_p = space.invariant_gaussian_measure(
            lambda k: 1.0 / (1.0 + k), expectation=mu_p
        )

        # Q = N(mu_q, C_q)
        mu_q = space.random()
        measure_q = space.invariant_gaussian_measure(
            lambda k: 2.0 / (1.0 + k), expectation=mu_q
        )

        # 1. Fast O(N) path
        kl_fast = measure_p.kl_divergence(measure_q)

        # 2. Force the slow dense path by downcasting Q to a base GaussianMeasure
        measure_q_base = GaussianMeasure(
            covariance=measure_q.covariance, expectation=measure_q.expectation
        )
        kl_slow = measure_p.kl_divergence(measure_q_base)

        assert np.isclose(kl_fast, kl_slow, rtol=1e-8)

    def test_affine_mapping_kkt_spectral_preconditioner(self, space: Lebesgue):
        """
        Tests that pushing an invariant measure forward with a generic operator
        and an inverse solver successfully triggers the spectrally-optimized
        KKT preconditioner (Water-filling + Jacobi Schur) and resolves the exact inverse.
        """

        # 1. Create a highly ill-conditioned prior measure (e.g., heat kernel)
        measure = space.heat_kernel_gaussian_measure(0.1)

        # 2. Create a generic forward operator to a smaller Euclidean space
        dim_out = max(1, space.dim // 2)
        target_space = EuclideanSpace(dim_out)
        P_matrix = np.random.randn(dim_out, space.dim)
        op_gen = LinearOperator.from_matrix(
            space, target_space, P_matrix, galerkin=True
        )

        # 3. Push forward with MINRES. Because 'op_gen' is generic, this drops
        # down to the base class but injects our smart invariant preconditioner first.
        solver = MinResSolver(rtol=1e-10)
        transformed_measure = measure.affine_mapping(
            operator=op_gen, inverse_solver=solver
        )

        # 4. Verify type fallback and KKT success
        assert type(transformed_measure) is GaussianMeasure
        assert transformed_measure.inverse_covariance_set

        # 5. Verify mathematical exactness: C_inv @ C = I
        C = transformed_measure.covariance.matrix(dense=True, galerkin=True)
        C_inv = transformed_measure.inverse_covariance.matrix(dense=True, galerkin=True)
        identity = np.eye(dim_out)

        assert np.allclose(C_inv @ C, identity, atol=1e-5)
