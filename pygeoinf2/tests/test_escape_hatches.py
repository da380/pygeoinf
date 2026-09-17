"""The escape hatches the audit found closed (FUNCTIONALITY_AUDIT.md §0.3).

Four of the five are back: the incomplete LU on the banded and block
preconditioners, a factor route on the ellipsoid's support function, a
coefficient operator that zero-pads past the space's own truncation, and
a walk that proposes a truncation degree beyond the space it is asked on.
The fifth, recomputing a path operator's quadrature on every application,
was dropped on purpose; DESIGN §76 says why.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ellipsoid
from pygeoinf2.numerics.preconditioners import (
    BandedPreconditioner,
    BlockPreconditioner,
)
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import check_operator
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space


def spd(space, rng, *, banded=False):
    root = rng.normal(size=(space.dim, space.dim))
    matrix = root @ root.T + space.dim * np.identity(space.dim)
    if banded:
        matrix = np.diag(np.diag(matrix)) + np.diag(np.diag(matrix, 1), 1)
        matrix = matrix + np.diag(np.diag(matrix, 1), -1)
    return LinearOperator.from_matrix(
        space,
        space,
        matrix,
        form="galerkin",
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
    )


class TestIncompleteFactorization:
    """``incomplete=True`` on all three sparse preconditioners, not one."""

    @pytest.mark.parametrize(
        "build",
        [
            lambda **kw: BandedPreconditioner(1, form="galerkin", **kw),
            lambda **kw: BlockPreconditioner(
                [[0, 1, 2], [3, 4, 5], [6, 7]], form="galerkin", **kw
            ),
        ],
        ids=["banded", "block"],
    )
    def test_a_lossless_incomplete_factorization_is_the_exact_one(self, build, rng):
        space = make_dense_metric_space(8)
        operator = spd(space, rng)
        exact = build()(operator)
        lossless = build(incomplete=True, drop_tol=0.0, fill_factor=100.0)(operator)
        for _ in range(5):
            vector = space.random(rng=rng)
            assert space.norm(
                space.subtract(exact(vector), lossless(vector))
            ) == pytest.approx(0.0, abs=1e-8 * space.norm(exact(vector)))

    def test_a_lossy_one_is_still_a_usable_preconditioner(self, rng):
        space = make_dense_metric_space(8)
        operator = spd(space, rng)
        lossy = BandedPreconditioner(2, incomplete=True, drop_tol=0.5)(operator)
        vector = space.random(rng=rng)
        assert np.isfinite(space.norm(lossy(vector)))


class TestEllipsoidFactor:
    """The support value through a factor is the one through the covariance."""

    @pytest.fixture
    def pieces(self, rng):
        space = make_dense_metric_space(5)
        coefficients = EuclideanSpace(5)
        root = rng.normal(size=(5, 5)) + 3.0 * np.identity(5)
        factor = LinearOperator.from_matrix(
            coefficients, space, root, form="components"
        )
        covariance = (factor @ factor.adjoint).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        precision = LinearOperator.from_matrix(
            space,
            space,
            np.linalg.inv(covariance.matrix(form="components")),
            form="components",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        return space, precision, covariance, factor

    def test_the_two_routes_agree(self, pieces, rng):
        space, precision, covariance, factor = pieces
        center = space.random(rng=rng)
        plain = Ellipsoid(space, precision, center=center, covariance=covariance)
        factored = Ellipsoid(
            space, precision, center=center, covariance=covariance, factor=factor
        )
        assert factored.factor is factor
        assert factored.translate(center).factor is factor
        for _ in range(5):
            direction = space.random(rng=rng)
            assert factored.support_function()(direction) == pytest.approx(
                plain.support_function()(direction)
            )
            assert space.norm(
                space.subtract(
                    factored.support_maximizer(direction),
                    plain.support_maximizer(direction),
                )
            ) == pytest.approx(0.0, abs=1e-10)

    def test_a_factor_into_the_wrong_space_is_refused(self, pieces):
        space, precision, covariance, factor = pieces
        wrong = LinearOperator.from_matrix(
            EuclideanSpace(5), EuclideanSpace(5), np.identity(5), form="components"
        )
        with pytest.raises(ValueError, match="factor"):
            Ellipsoid(space, precision, covariance=covariance, factor=wrong)

    def test_a_measure_passes_its_factor_to_the_credible_ellipsoid(self, pieces, rng):
        space, precision, covariance, factor = pieces
        measure = GaussianMeasure(
            space, covariance=covariance, covariance_factor=factor, precision=precision
        )
        ellipsoid = measure.credible_set(level=0.9)
        assert ellipsoid.factor is not None
        without = Ellipsoid(
            space,
            ellipsoid.precision,
            center=ellipsoid.center,
            covariance=ellipsoid.covariance,
        )
        for _ in range(5):
            direction = space.random(rng=rng)
            assert ellipsoid.support_function()(direction) == pytest.approx(
                without.support_function()(direction)
            )


finufft = pytest.importorskip("finufft")

from pygeoinf2.symmetric_space import Sobolev as BoxSobolev  # noqa: E402
from pygeoinf2.symmetric_space.fourier import PeriodicBox  # noqa: E402
from pygeoinf2.symmetric_space.sphere import Sobolev  # noqa: E402


class TestCoefficientPadding:
    """A band past the space's truncation reports zeros there."""

    def test_the_sphere_pads_with_zeros(self, rng):
        space = Sobolev(4, 2.0, 0.2)
        within = space.coefficient_operator(lmax=4)
        padded = space.coefficient_operator(lmax=6)
        assert padded.codomain.dim == 7**2
        field = space.random(rng=rng)
        out = padded(field)
        assert np.allclose(out[: within.codomain.dim], within(field))
        assert np.allclose(out[within.codomain.dim :], 0.0)
        check_operator(padded, rng=rng)

        synthesis = space.from_coefficient_operator(lmax=6)
        assert synthesis.domain.dim == 7**2
        coefficients = rng.normal(size=7**2)
        assert np.allclose(
            space.to_components(synthesis(coefficients)),
            space.to_components(
                space.from_coefficient_operator(lmax=4)(coefficients[: 5**2])
            ),
        )
        check_operator(synthesis, rng=rng)

    def test_a_band_starting_past_the_space_is_all_zeros(self, rng):
        space = Sobolev(4, 2.0, 0.2)
        above = space.coefficient_operator(lmin=5, lmax=6)
        assert above.codomain.dim == 7**2 - 5**2
        assert np.allclose(above(space.random(rng=rng)), 0.0)

    def test_the_box_pads_with_zeros(self, rng):
        space = BoxSobolev((8, 8), 1.0, 0.3)
        highest = int(space.degrees.max())
        padded = space.coefficient_operator(lmax=highest + 2)
        larger = space.with_degree(highest + 3)
        assert (
            padded.codomain.dim
            == larger.coefficient_operator(lmax=highest + 2).codomain.dim
        )
        check_operator(padded, rng=rng)
        check_operator(space.from_coefficient_operator(lmax=highest + 2), rng=rng)

    def test_an_inverted_band_is_still_refused(self):
        space = Sobolev(4, 2.0, 0.2)
        with pytest.raises(ValueError, match="Degrees must satisfy"):
            space.coefficient_operator(lmin=7, lmax=6)


class TestSufficientDegree:
    """The walk goes past the space, and counts what each degree would hold."""

    def test_the_sphere_walk_is_the_weighted_sum(self):
        space = Sobolev(4, 2.0, 0.2)

        def symbol(eigenvalues):
            return (1.0 + 0.2**2 * eigenvalues) ** -2.0

        degree = space.sufficient_degree(symbol, rtol=1e-4)
        assert degree > 4
        # The same walk by hand.
        total, by_hand = 0.0, 0
        while True:
            term = (2 * by_hand + 1) * float(symbol(by_hand * (by_hand + 1.0)))
            total += term
            if term / total <= 1e-4:
                break
            by_hand += 1
        assert degree == by_hand
        assert space.with_degree(degree).lmax == degree

    def test_the_floor_and_the_ceiling(self):
        space = Sobolev(4, 2.0, 0.2)

        def symbol(eigenvalues):
            return (1.0 + 0.2**2 * eigenvalues) ** -2.0

        free = space.sufficient_degree(symbol, rtol=1e-4)
        assert (
            space.sufficient_degree(symbol, rtol=1e-4, min_degree=free + 5) == free + 5
        )
        assert space.sufficient_degree(symbol, rtol=1e-4, max_degree=3) == 3
        with pytest.raises(ValueError):
            space.sufficient_degree(symbol, rtol=2.0)

    def test_the_box_shells_are_the_packings_own(self):
        """Each degree's enumerated modes are the ones a box holding that
        degree has, eigenvalue for eigenvalue."""
        box = PeriodicBox((16, 16), lengths=(1.0, 2.0))
        for degree in range(0, 5):
            enumerated = np.sort(box._degree_eigenvalues(degree))
            present = np.sort(box.laplacian_eigenvalues[box.degrees == degree])
            assert enumerated.size == present.size
            assert np.allclose(enumerated, present)

    def test_the_box_walk_goes_past_the_space(self):
        box = PeriodicBox((8, 8))
        degree = box.sufficient_degree(lambda lam: (1.0 + lam) ** -2.0, rtol=1e-3)
        assert degree > int(box.degrees.max())
        assert int(box.with_degree(degree).degrees.max()) >= degree
