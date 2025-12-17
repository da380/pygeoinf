"""
Tests for the spherical harmonic tools module.
"""

import pytest
import numpy as np
import pyshtools as sh
from pygeoinf.symmetric_space.sh_tools import SHVectorConverter


class TestSHVectorConverter:

    @pytest.fixture
    def lmax(self):
        return 10

    @pytest.fixture
    def converter(self, lmax):
        return SHVectorConverter(lmax=lmax, lmin=0)

    def test_vector_size(self, converter, lmax):
        """Test that the calculated vector size is correct."""
        expected_size = (lmax + 1) ** 2
        assert converter.vector_size == expected_size

    def test_round_trip_identity(self, converter, lmax):
        """Test that to_vector and from_vector are inverses."""
        # Create random coefficients
        coeffs = sh.SHCoeffs.from_random(np.ones(lmax + 1), normalization="ortho")

        # Convert to vector
        vec = converter.to_vector(coeffs.coeffs)

        # Convert back to array
        reconstructed_coeffs_array = converter.from_vector(vec)

        # Check equality
        np.testing.assert_allclose(coeffs.coeffs, reconstructed_coeffs_array)

    def test_specific_coefficient_mapping(self, lmax):
        """
        Test that a specific coefficient (l, m) maps to the expected index
        in the vector.
        """
        # Create a converter with lmin=0
        converter = SHVectorConverter(lmax=lmax, lmin=0)

        # We want to track the coefficient for l=2, m=1
        # In pyshtools 'ortho', this is at index [0, 2, 1] for cosine (positive m)
        # or [1, 2, 1] for sine (negative m).

        coeffs = np.zeros((2, lmax + 1, lmax + 1))

        # Set term for l=2, m=0 (Center of the l=2 block)
        coeffs[0, 2, 0] = 1.0

        vec = converter.to_vector(coeffs)

        # Index calculation:
        # l=0 takes 1 spot
        # l=1 takes 3 spots
        # l=2 starts at index 4. m goes -2, -1, 0. So 0 is at index 4+2=6.
        assert vec[6] == 1.0
        assert np.sum(vec) == 1.0

    def test_truncation(self, converter, lmax):
        """Test handling of input larger than converter's lmax."""
        larger_lmax = lmax + 5
        coeffs = sh.SHCoeffs.from_random(
            np.ones(larger_lmax + 1), normalization="ortho"
        )

        # Convert using the smaller converter (should truncate)
        vec = converter.to_vector(coeffs.coeffs)

        assert len(vec) == converter.vector_size

        # Verify the data matches the lower degrees
        reconstructed = converter.from_vector(vec)
        np.testing.assert_allclose(
            reconstructed[:, : lmax + 1, : lmax + 1],
            coeffs.coeffs[:, : lmax + 1, : lmax + 1],
        )

    def test_padding(self, converter, lmax):
        """Test handling of input smaller than converter's lmax."""
        smaller_lmax = lmax - 5
        coeffs = sh.SHCoeffs.from_random(
            np.ones(smaller_lmax + 1), normalization="ortho"
        )

        # Convert using the larger converter (should zero pad)
        vec = converter.to_vector(coeffs.coeffs)

        assert len(vec) == converter.vector_size

        # Check that high degrees are zero
        reconstructed = converter.from_vector(vec)
        assert np.all(reconstructed[:, smaller_lmax + 1 :, :] == 0)
