"""
Unit tests for discontinuity functionality.

Tests for the new API that allows creating spaces with discontinuities:
- IntervalDomain.split_at_discontinuities()
- Lebesgue.restrict_to_subinterval()
- Lebesgue.with_discontinuities()
"""

import pytest
from pygeoinf.interval import IntervalDomain, Lebesgue, LebesgueSpaceDirectSum


class TestIntervalDomainSplit:
    """Tests for IntervalDomain.split_at_discontinuities()"""

    def test_split_single_discontinuity(self):
        """Test splitting at a single discontinuity point."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        subdomains = domain.split_at_discontinuities([0.5])

        assert len(subdomains) == 2
        assert subdomains[0].a == 0
        assert subdomains[0].b == 0.5
        assert subdomains[1].a == 0.5
        assert subdomains[1].b == 1

    def test_split_multiple_discontinuities(self):
        """Test splitting at multiple discontinuity points."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        subdomains = domain.split_at_discontinuities([0.25, 0.5, 0.75])

        assert len(subdomains) == 4
        assert subdomains[0].a == 0
        assert subdomains[0].b == 0.25
        assert subdomains[1].a == 0.25
        assert subdomains[1].b == 0.5
        assert subdomains[2].a == 0.5
        assert subdomains[2].b == 0.75
        assert subdomains[3].a == 0.75
        assert subdomains[3].b == 1

    def test_split_empty_list(self):
        """Test that empty list returns original domain."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        subdomains = domain.split_at_discontinuities([])

        assert len(subdomains) == 1
        assert subdomains[0] == domain

    def test_split_unsorted_points(self):
        """Test that unsorted points are automatically sorted."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        subdomains = domain.split_at_discontinuities([0.75, 0.25, 0.5])

        assert len(subdomains) == 4
        assert subdomains[0].b == 0.25
        assert subdomains[1].b == 0.5
        assert subdomains[2].b == 0.75

    def test_split_point_outside_domain_raises(self):
        """Test that points outside domain raise ValueError."""
        domain = IntervalDomain(0, 1, boundary_type='closed')

        with pytest.raises(ValueError, match="must be in interior"):
            domain.split_at_discontinuities([1.5])

        with pytest.raises(ValueError, match="must be in interior"):
            domain.split_at_discontinuities([-0.5])

    def test_split_point_at_boundary_raises(self):
        """Test that points at boundary raise ValueError."""
        domain = IntervalDomain(0, 1, boundary_type='closed')

        with pytest.raises(ValueError, match="must be in interior"):
            domain.split_at_discontinuities([0])

        with pytest.raises(ValueError, match="must be in interior"):
            domain.split_at_discontinuities([1])

    def test_split_duplicate_points_raises(self):
        """Test that duplicate points raise ValueError."""
        domain = IntervalDomain(0, 1, boundary_type='closed')

        with pytest.raises(ValueError, match="unique"):
            domain.split_at_discontinuities([0.5, 0.5])


class TestLebesgueRestriction:
    """Tests for Lebesgue.restrict_to_subinterval()"""

    def test_restrict_to_subinterval(self):
        """Test basic restriction to subinterval."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue(100, domain, basis='none')

        subdomain = IntervalDomain(0, 0.5, boundary_type='closed')
        restricted = space.restrict_to_subinterval(subdomain)

        assert restricted.dim == 100
        assert restricted.function_domain == subdomain
        assert restricted.function_domain.a == 0
        assert restricted.function_domain.b == 0.5

    def test_restrict_subdomain_outside_raises(self):
        """Test that subdomain outside domain raises ValueError."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue(100, domain, basis='none')

        subdomain = IntervalDomain(-0.5, 0.5, boundary_type='closed')

        with pytest.raises(ValueError, match="must be contained"):
            space.restrict_to_subinterval(subdomain)

    def test_restrict_inherits_settings(self):
        """Test that restricted space inherits integration settings."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue(100, domain, basis='none')

        # Set custom integration settings
        space.integration_method = 'trapz'
        space.integration_npoints = 2000

        subdomain = IntervalDomain(0, 0.5, boundary_type='closed')
        restricted = space.restrict_to_subinterval(subdomain)

        assert restricted.integration_method == 'trapz'
        assert restricted.integration_npoints == 2000


class TestLebesgueWithDiscontinuities:
    """Tests for Lebesgue.with_discontinuities()"""

    def test_with_single_discontinuity(self):
        """Test creating space with single discontinuity."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(200, domain, [0.5])

        assert isinstance(space, LebesgueSpaceDirectSum)
        assert space.number_of_subspaces == 2
        assert space.dim == 200

        # Check subspaces (cast to Lebesgue for type checker)
        sub0 = space.subspace(0)
        sub1 = space.subspace(1)
        assert isinstance(sub0, Lebesgue)
        assert isinstance(sub1, Lebesgue)

        assert sub0.function_domain.a == 0
        assert sub0.function_domain.b == 0.5
        assert sub1.function_domain.a == 0.5
        assert sub1.function_domain.b == 1

        # Dimensions should be roughly equal (proportional to length)
        assert sub0.dim + sub1.dim == 200

    def test_with_multiple_discontinuities(self):
        """Test creating space with multiple discontinuities."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(400, domain, [0.25, 0.5, 0.75])

        assert isinstance(space, LebesgueSpaceDirectSum)
        assert space.number_of_subspaces == 4
        assert space.dim == 400

    def test_with_custom_dimensions(self):
        """Test creating space with custom dimension allocation."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(
            200, domain, [0.5],
            dim_per_subspace=[80, 120]
        )

        assert space.number_of_subspaces == 2
        assert space.dim == 200
        assert space.subspace(0).dim == 80
        assert space.subspace(1).dim == 120

    def test_with_custom_dimensions_wrong_length_raises(self):
        """Test that wrong number of custom dimensions raises."""
        domain = IntervalDomain(0, 1, boundary_type='closed')

        with pytest.raises(ValueError, match="must have length"):
            Lebesgue.with_discontinuities(
                200, domain, [0.5],
                dim_per_subspace=[100]  # Should be 2 elements
            )

    def test_with_custom_dimensions_wrong_sum_raises(self):
        """Test that custom dimensions not summing to total raises."""
        domain = IntervalDomain(0, 1, boundary_type='closed')

        with pytest.raises(ValueError, match="must sum to"):
            Lebesgue.with_discontinuities(
                200, domain, [0.5],
                dim_per_subspace=[80, 100]  # Sums to 180, not 200
            )

    def test_proportional_allocation(self):
        """Test that default allocation is proportional to lengths."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(100, domain, [0.3])

        # First subdomain is 30% of total, second is 70%
        # So we expect roughly 30 and 70 dimensions
        dim0 = space.subspace(0).dim
        dim1 = space.subspace(1).dim

        assert dim0 + dim1 == 100
        # Allow some tolerance due to rounding
        assert 25 <= dim0 <= 35
        assert 65 <= dim1 <= 75

    def test_with_no_discontinuities(self):
        """Test creating space with no discontinuities."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(100, domain, [])

        # Should still return a DirectSum with one subspace
        assert isinstance(space, LebesgueSpaceDirectSum)
        assert space.number_of_subspaces == 1
        assert space.dim == 100
        sub0 = space.subspace(0)
        assert isinstance(sub0, Lebesgue)
        assert sub0.function_domain == domain


class TestBasisHandling:
    """Tests for basis handling in discontinuous spaces"""

    def test_basis_string_propagates_to_all_subspaces(self):
        """Test that string basis type is applied to all subspaces."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(
            200, domain, [0.5],
            basis='sine'
        )

        # Check both subspaces have sine basis
        sub0 = space.subspace(0)
        sub1 = space.subspace(1)
        assert isinstance(sub0, Lebesgue)
        assert isinstance(sub1, Lebesgue)

        # Both should have basis type 'sine'
        assert sub0._basis_type == 'sine'
        assert sub1._basis_type == 'sine'

    def test_basis_none_creates_baseless_subspaces(self):
        """Test that basis='none' creates baseless subspaces."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(
            200, domain, [0.5],
            basis='none'
        )

        # Check both subspaces are baseless
        sub0 = space.subspace(0)
        sub1 = space.subspace(1)
        assert isinstance(sub0, Lebesgue)
        assert isinstance(sub1, Lebesgue)

        assert sub0._basis_type == 'none'
        assert sub1._basis_type == 'none'

    def test_basis_per_subspace_different_bases(self):
        """Test setting different basis for each subspace."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(
            200, domain, [0.5],
            basis_per_subspace=['fourier', 'sine']
        )

        # Check each subspace has its specified basis
        sub0 = space.subspace(0)
        sub1 = space.subspace(1)
        assert isinstance(sub0, Lebesgue)
        assert isinstance(sub1, Lebesgue)

        assert sub0._basis_type == 'fourier'
        assert sub1._basis_type == 'sine'

    def test_basis_per_subspace_overrides_basis(self):
        """Test that basis_per_subspace overrides basis parameter."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(
            200, domain, [0.5],
            basis='fourier',  # This should be ignored
            basis_per_subspace=['sine', 'cosine']
        )

        # Check basis_per_subspace was used, not basis
        sub0 = space.subspace(0)
        sub1 = space.subspace(1)
        assert isinstance(sub0, Lebesgue)
        assert isinstance(sub1, Lebesgue)

        assert sub0._basis_type == 'sine'
        assert sub1._basis_type == 'cosine'

    def test_basis_per_subspace_wrong_length_raises(self):
        """Test that wrong length basis_per_subspace raises error."""
        domain = IntervalDomain(0, 1, boundary_type='closed')

        with pytest.raises(ValueError, match="must have length"):
            Lebesgue.with_discontinuities(
                200, domain, [0.5],
                basis_per_subspace=['fourier']  # Should be 2 elements
            )

    def test_basis_list_raises_error(self):
        """Test that providing list of basis functions raises error."""
        domain = IntervalDomain(0, 1, boundary_type='closed')

        with pytest.raises(ValueError, match="not supported"):
            Lebesgue.with_discontinuities(
                200, domain, [0.5],
                basis=[lambda x: 1, lambda x: x]  # List not supported
            )

    def test_multiple_discontinuities_with_bases(self):
        """Test basis handling with multiple discontinuities."""
        domain = IntervalDomain(0, 1, boundary_type='closed')
        space = Lebesgue.with_discontinuities(
            400, domain, [0.25, 0.5, 0.75],
            basis_per_subspace=['fourier', 'sine', 'cosine', 'fourier']
        )

        assert space.number_of_subspaces == 4
        for i, expected_basis in enumerate(['fourier', 'sine', 'cosine',
                                            'fourier']):
            sub = space.subspace(i)
            assert isinstance(sub, Lebesgue)
            assert sub._basis_type == expected_basis


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
