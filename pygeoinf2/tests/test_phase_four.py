"""
The catalogue's remaining algebra, numerics, probability and geometry.

Nothing here blocks an example, which is why it comes last; but each row was
marked as used in practice, so each one is ported rather than dropped. See
DESIGN.md section 21.16.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.subspaces import AffineSubspace, OrthogonalProjector
from pygeoinf2.numerics.solvers import (
    CGSolver,
    CholeskySolver,
    FlexibleCGSolver,
    GMRESSolver,
)
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import check_operator
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space


def nonsymmetric(space, rng):
    """A well-conditioned operator with no symmetry to exploit."""
    size = space.dim
    matrix = rng.normal(size=(size, size)) + 4.0 * np.identity(size)
    return LinearOperator.from_component_matrix(space, space, matrix), matrix


def positive_definite(space, rng):
    size = space.dim
    root = rng.normal(size=(size, size))
    matrix = root @ root.T + size * np.identity(size)
    return LinearOperator.self_adjoint(
        space,
        lambda x: space.solve_gram(matrix @ space.to_components(x)),
        traits=Traits.POSITIVE_DEFINITE,
    )


class TestSparseMatrices:
    def test_a_sparse_matrix_needs_no_separate_constructor(self, rng):
        """Both support ``@`` and ``.T``, which is all the constructors use."""
        X = make_weighted_space()
        Y = EuclideanSpace(X.dim)
        dense = np.diag(np.arange(1.0, X.dim + 1))
        sparse = LinearOperator.from_derivative_matrix(X, Y, csr_matrix(dense))
        check_operator(sparse, rng=rng)
        reference = LinearOperator.from_derivative_matrix(X, Y, dense)
        x, y = X.random(rng=rng), Y.random(rng=rng)
        assert np.allclose(sparse(x), reference(x))
        assert np.allclose(
            X.to_components(sparse.adjoint(y)),
            X.to_components(reference.adjoint(y)),
        )

    def test_a_wrong_shape_is_still_refused(self):
        X = make_weighted_space()
        with pytest.raises(ValueError, match="expected"):
            LinearOperator.from_component_matrix(X, X, csr_matrix(np.zeros((2, 2))))


class TestCoordinateOperators:
    def test_they_invert_one_another(self, rng):
        X = make_weighted_space()
        projection, inclusion = X.coordinate_projection(), X.coordinate_inclusion()
        check_operator(projection, rng=rng)
        check_operator(inclusion, rng=rng)
        x = X.random(rng=rng)
        assert np.allclose(inclusion(projection(x)), x)

    def test_the_adjoint_of_inclusion_is_not_the_projection(self, rng):
        """Which is section 5.6 in its smallest possible setting.

        The adjoint gives *derivative* components, ``G c_x``; the projection
        gives the components themselves. They coincide only when the metric is
        the identity.
        """
        X = make_weighted_space()
        x = X.random(rng=rng)
        assert not np.allclose(
            X.coordinate_inclusion().adjoint(x), X.coordinate_projection()(x)
        )
        assert np.allclose(
            X.coordinate_inclusion().adjoint(x),
            X.apply_gram(X.to_components(x)),
        )


class TestDiagonals:
    @pytest.fixture
    def banded(self, rng):
        X = make_weighted_space()
        full = rng.normal(size=(X.dim, X.dim))
        matrix = np.triu(np.tril(full, 1), -1)
        return X, matrix, LinearOperator.from_component_matrix(X, X, matrix)

    def test_the_exact_route_reads_the_matrix(self, banded):
        X, matrix, operator = banded
        got = operator.diagonals(offsets=[-1, 0, 1], form="components")
        for index, offset in enumerate([-1, 0, 1]):
            diagonal = np.diag(matrix, offset)
            if offset >= 0:
                assert np.allclose(
                    got[index, offset : offset + diagonal.size], diagonal
                )
            else:
                assert np.allclose(got[index, : diagonal.size], diagonal)

    def test_the_banded_probe_agrees_on_a_banded_operator(self, banded):
        X, matrix, operator = banded
        assert np.allclose(
            operator.diagonals(offsets=[-1, 0, 1], form="components"),
            operator.diagonals(offsets=[-1, 0, 1], form="components", probe="banded"),
        )

    def test_the_banded_probe_is_wrong_on_a_full_one(self, rng):
        """The negative control. It sums in the out-of-band entries, and says so."""
        X = make_weighted_space()
        operator = LinearOperator.from_component_matrix(
            X, X, rng.normal(size=(X.dim, X.dim))
        )
        assert not np.allclose(
            operator.diagonals(offsets=[-1, 0, 1], form="components"),
            operator.diagonals(offsets=[-1, 0, 1], form="components", probe="banded"),
        )

    def test_no_offsets_is_refused(self, banded):
        _, _, operator = banded
        with pytest.raises(ValueError, match="At least one offset"):
            operator.diagonals(offsets=[])


class TestGMRES:
    def test_it_solves_a_nonsymmetric_system_exactly(self, rng):
        X = EuclideanSpace(12)
        operator, matrix = nonsymmetric(X, rng)
        b = X.random(rng=rng)
        result = GMRESSolver(rtol=1e-12, restart=X.dim)(operator).solve(b)
        assert result.converged
        assert result.iterations <= X.dim
        assert np.allclose(result.solution, np.linalg.solve(matrix, b))

    def test_it_works_on_a_weighted_space(self, rng):
        """Coordinate-free: the Krylov basis is orthonormal in *this* metric."""
        X = make_weighted_space()
        operator, matrix = nonsymmetric(X, rng)
        b = X.random(rng=rng)
        result = GMRESSolver(rtol=1e-12, restart=X.dim)(operator).solve(b)
        assert np.allclose(
            X.to_components(result.solution),
            np.linalg.solve(matrix, X.to_components(b)),
        )

    def test_restarting_still_converges(self, rng):
        X = EuclideanSpace(12)
        operator, matrix = nonsymmetric(X, rng)
        b = X.random(rng=rng)
        result = GMRESSolver(rtol=1e-12, restart=3, maxiter=400)(operator).solve(b)
        assert result.converged
        assert result.iterations > X.dim  # restarting costs iterations
        assert np.allclose(result.solution, np.linalg.solve(matrix, b))

    def test_it_asks_nothing_of_its_operator(self):
        assert GMRESSolver.requires == Traits.NONE

    def test_a_nonsense_restart_is_refused(self):
        with pytest.raises(ValueError, match="restart length"):
            GMRESSolver(restart=0)


class TestFlexibleCG:
    def test_it_agrees_with_cg_on_a_fixed_preconditioner(self, rng):
        """Polak-Ribiere and Fletcher-Reeves coincide exactly there.

        Which is the point: the extra term measures the failure of an
        assumption that a fixed preconditioner does not break.
        """
        X = make_weighted_space()
        operator = positive_definite(X, rng)
        b = X.random(rng=rng)
        strict = CGSolver(rtol=1e-12)(operator).solve(b)
        flexible = FlexibleCGSolver(rtol=1e-12)(operator).solve(b)
        assert np.allclose(
            X.to_components(strict.solution), X.to_components(flexible.solution)
        )
        assert strict.iterations == flexible.iterations

    def test_it_converges_with_a_changing_preconditioner(self, rng):
        X = make_weighted_space()
        operator = positive_definite(X, rng)
        inner = CGSolver(rtol=0.3, maxiter=1, strict=False)(operator)
        preconditioner = LinearOperator.from_callables(X, X, inner, adjoint=inner)
        b = X.random(rng=rng)
        result = FlexibleCGSolver(
            rtol=1e-10, preconditioner=preconditioner, strict=False
        )(operator).solve(b)
        assert X.norm(X.subtract(operator(result.solution), b)) < 1e-8 * X.norm(b)


class TestSolverDiagnostics:
    def test_the_callback_sees_every_step(self, rng):
        X = make_weighted_space()
        operator = positive_definite(X, rng)
        seen = []
        result = CGSolver(
            rtol=1e-12, callback=lambda step, residual: seen.append(step)
        )(operator).solve(X.random(rng=rng))
        assert seen == list(range(result.iterations + 1))

    def test_the_history_records_the_residuals(self, rng):
        X = make_weighted_space()
        operator = positive_definite(X, rng)
        result = CGSolver(rtol=1e-12)(operator).solve(X.random(rng=rng))
        assert len(result.history) == result.iterations + 1
        assert result.history[-1] == pytest.approx(result.residual_norm)

    def test_a_stalled_solve_says_where_it_stalled(self, rng):
        """Which the final residual alone cannot."""
        X = make_weighted_space()
        operator = positive_definite(X, rng)
        result = CGSolver(rtol=1e-16, maxiter=2, strict=False)(operator).solve(
            X.random(rng=rng)
        )
        assert not result.converged
        assert len(result.history) >= 2


class TestMeasureAdjustments:
    @pytest.fixture
    def measure(self, rng):
        X = EuclideanSpace(6)
        root = rng.normal(size=(6, 6))
        return (
            X,
            root @ root.T,
            GaussianMeasure.from_covariance_matrix(X, root @ root.T),
        )

    def test_a_regularised_inverse_supplies_a_precision(self, measure, rng):
        space, matrix, without = measure
        with pytest.raises(NotImplementedError):
            without.mahalanobis_squared(space.random(rng=rng))
        with_precision = without.with_regularized_inverse(
            CholeskySolver(), damping=1e-6
        )
        x = space.random(rng=rng)
        assert with_precision.mahalanobis_squared(x) == pytest.approx(
            float(x @ np.linalg.solve(matrix + 1e-6 * np.identity(6), x))
        )

    def test_the_covariance_is_left_alone(self, measure):
        """The two are deliberately not inverses; the measure says so."""
        space, matrix, without = measure
        with_precision = without.with_regularized_inverse(
            CholeskySolver(), damping=1e-3
        )
        assert np.allclose(
            with_precision.covariance.matrix(form="galerkin"),
            without.covariance.matrix(form="galerkin"),
        )

    def test_negative_damping_is_refused(self, measure):
        _, _, without = measure
        with pytest.raises(ValueError, match="non-negative"):
            without.with_regularized_inverse(CholeskySolver(), damping=-1.0)

    def test_rescaling_hits_the_requested_deviation(self, measure, rng):
        space, _, original = measure
        direction = space.random(rng=rng)
        rescaled = original.rescale_directional_variance(direction, 2.5)
        assert np.sqrt(rescaled.directional_variance(direction)) == pytest.approx(2.5)

    def test_rescaling_moves_every_direction_together(self, measure, rng):
        """A recalibration, not a change of shape."""
        space, _, original = measure
        first, second = space.random(rng=rng), space.random(rng=rng)
        rescaled = original.rescale_directional_variance(first, 2.5)
        ratio = rescaled.directional_variance(first) / original.directional_variance(
            first
        )
        assert rescaled.directional_variance(second) == pytest.approx(
            ratio * original.directional_variance(second)
        )

    def test_a_zero_variance_direction_is_refused(self, measure):
        _, _, original = measure
        with pytest.raises(ValueError, match="not positive"):
            original.rescale_directional_variance(original.domain.zero(), 1.0)

    def test_thresholding_that_breaks_positivity_is_refused(self):
        """It is not a covariance any more, so it is not returned as one.

        Dropping entries from a positive semidefinite matrix usually leaves one
        — which is why it needs checking rather than assuming. This particular
        covariance goes to a smallest eigenvalue of ``-0.41`` at a tenth of its
        largest entry.
        """
        space = EuclideanSpace(4)
        covariance = np.array(
            [
                [0.4544, 0.8193, -0.3161, -1.1381],
                [0.8193, 3.0151, -0.8542, -1.1519],
                [-0.3161, -0.8542, 2.4867, 2.6594],
                [-1.1381, -1.1519, 2.6594, 7.5421],
            ]
        )
        measure = GaussianMeasure.from_covariance_matrix(space, covariance)
        measure.with_sparse_approximation(threshold=0.01)
        with pytest.raises(ValueError, match="positive semidefinite"):
            measure.with_sparse_approximation(threshold=0.1)

    def test_a_gentle_threshold_keeps_a_covariance(self, measure, rng):
        space, _, original = measure
        sparse = original.with_sparse_approximation(threshold=1e-12)
        assert np.allclose(
            sparse.covariance.matrix(form="galerkin"),
            original.covariance.matrix(form="galerkin"),
        )


class TestSubspaceConstructions:
    @pytest.fixture
    def constrained(self, rng):
        X = make_weighted_space()
        operator = LinearOperator.from_derivative_matrix(
            X, EuclideanSpace(2), rng.normal(size=(2, X.dim))
        )
        value = np.array([1.0, -0.5])
        return X, operator, value, AffineSubspace.from_linear_equation(operator, value)

    def test_it_remembers_the_equation_it_was_built_from(self, constrained):
        space, operator, value, subspace = constrained
        assert subspace.has_explicit_equation
        assert subspace.constraint_operator is operator
        assert np.allclose(subspace.constraint_value, value)

    def test_a_subspace_from_a_basis_has_no_equation(self, rng):
        X = make_weighted_space()
        subspace = AffineSubspace.from_tangent_basis(X, [X.random(rng=rng)])
        assert not subspace.has_explicit_equation
        with pytest.raises(AttributeError, match="explicit equation"):
            subspace.constraint_operator

    def test_the_pseudo_inverse_gives_the_translation(self, constrained):
        space, _, value, subspace = constrained
        assert np.allclose(
            space.to_components(subspace.pseudo_inverse()(value)),
            space.to_components(subspace.translation),
        )

    def test_a_different_constraint_value_moves_only_the_translation(
        self, constrained, rng
    ):
        space, operator, _, subspace = constrained
        moved = subspace.with_constraint_value(np.array([2.0, 3.0]))
        assert np.allclose(operator(moved.translation), [2.0, 3.0])
        x = space.random(rng=rng)
        assert np.allclose(
            space.to_components(subspace.projector(x)),
            space.to_components(moved.projector(x)),
        )

    def test_the_projection_operator_lands_on_the_subspace(self, constrained, rng):
        space, operator, value, subspace = constrained
        projection = subspace.projection_operator()
        assert np.allclose(operator(projection(space.random(rng=rng))), value)

    def test_hyperplanes_round_trip(self, constrained, rng):
        space, _, _, subspace = constrained
        planes = subspace.to_hyperplanes()
        assert len(planes) == 2
        rebuilt = AffineSubspace.from_hyperplanes(planes)
        x = space.random(rng=rng)
        assert np.allclose(
            space.to_components(rebuilt.project(x)),
            space.to_components(subspace.project(x)),
        )

    def test_a_point_of_the_subspace_is_on_every_hyperplane(self, constrained, rng):
        space, _, _, subspace = constrained
        point = subspace.project(space.random(rng=rng))
        assert all(plane.contains(point) for plane in subspace.to_hyperplanes())

    def test_tangent_and_complement_partition_the_space(self, rng):
        X = make_weighted_space()
        vectors = [X.random(rng=rng) for _ in range(2)]
        tangent = AffineSubspace.from_tangent_basis(X, vectors)
        complement = AffineSubspace.from_complement_basis(X, vectors)
        assert tangent.dimension() + complement.dimension() == X.dim

    def test_a_projector_reports_a_basis_of_its_range(self, rng):
        X = make_weighted_space()
        vectors = [X.random(rng=rng) for _ in range(2)]
        projector = OrthogonalProjector.from_basis(X, vectors)
        basis = projector.basis()
        assert len(basis) == 2
        for vector in basis:
            assert np.allclose(projector(vector), vector)

    def test_an_affine_subspace_is_its_own_boundary(self, constrained):
        _, _, _, subspace = constrained
        assert subspace.boundary() is subspace

    def test_no_hyperplanes_is_refused(self):
        with pytest.raises(ValueError, match="At least one hyperplane"):
            AffineSubspace.from_hyperplanes([])


class TestDerivativeOperator:
    def test_it_differentiates(self):
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        X = BoxLebesgue((256,), lengths=(2.0 * np.pi,))
        axis = X.grid_axes[0]
        assert np.allclose(
            X.derivative_operator()(X.project_function(np.sin)),
            np.cos(axis),
            atol=1e-9,
        )

    def test_it_is_anti_self_adjoint_in_l2(self, rng):
        """``int f' g == -int f g'``, which is what its adjoint must reflect."""
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        X = BoxLebesgue((64,), lengths=(1.0,))
        derivative = X.derivative_operator()
        check_operator(derivative, rng=rng)
        u, v = X.random(rng=rng), X.random(rng=rng)
        assert X.inner_product(derivative(u), v) == pytest.approx(
            -X.inner_product(u, derivative(v))
        )

    def test_it_picks_the_right_axis(self):
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        X = BoxLebesgue((32, 32), lengths=(2.0 * np.pi, 2.0 * np.pi))
        first, second = np.meshgrid(*X.grid_axes, indexing="ij")
        field = X.project_function(lambda p: np.sin(p[0]) * np.cos(3.0 * p[1]))
        assert np.allclose(
            X.derivative_operator(axis=1)(field),
            -3.0 * np.sin(first) * np.sin(3.0 * second),
            atol=1e-9,
        )

    def test_the_sobolev_version_has_a_working_adjoint(self, rng):
        from pygeoinf2.symmetric_space import Sobolev as BoxSobolev

        X = BoxSobolev((64,), 2.0, 0.1, lengths=(1.0,))
        check_operator(X.derivative_operator(), rng=rng)

    def test_a_missing_axis_is_refused(self):
        from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue

        with pytest.raises(ValueError, match="axes"):
            BoxLebesgue((16,), lengths=(1.0,)).derivative_operator(axis=2)


class TestSurfaces:
    def test_a_projection_lands_on_the_surface(self, rng):
        from pygeoinf2.geometry.convex import BallSurface

        X = make_weighted_space()
        surface = BallSurface(X, radius=2.0)
        point = surface.project(X.random(rng=rng))
        assert surface.contains(point)
        assert X.norm(point) == pytest.approx(2.0)

    def test_the_centre_has_no_nearest_point(self):
        from pygeoinf2.geometry.convex import BallSurface

        X = make_weighted_space()
        with pytest.raises(ValueError, match="equidistant"):
            BallSurface(X).project(X.zero())

    def test_samples_are_on_the_surface_and_isotropic(self, rng):
        from pygeoinf2.geometry.convex import BallSurface

        X = make_weighted_space()
        surface = BallSurface(X, radius=2.0)
        draws = [surface.sample(rng=rng) for _ in range(300)]
        assert all(surface.contains(draw) for draw in draws)
        # white noise, not `random`: only the former is isotropic in a metric
        assert X.norm(X.mean(draws)) < 0.3 * surface.radius

    def test_an_ellipsoid_surface_recognises_its_own_points(self, rng):
        from pygeoinf2.geometry.convex import EllipsoidSurface

        X = make_weighted_space()
        precision = LinearOperator.self_adjoint(
            X, lambda v: v, traits=Traits.POSITIVE_DEFINITE
        )
        surface = EllipsoidSurface(X, precision)
        x = X.random(rng=rng)
        assert surface.contains(X.scale(1.0 / X.norm(x), x))
        assert not surface.contains(X.scale(2.0 / X.norm(x), x))


class TestAcquisitionHelpers:
    def test_clusters_partition_the_points(self, rng):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(8)
        points = X.stations(count=40, rng=rng)
        clusters = X.cluster_points(points, 0.6)
        covered = sorted(index for cluster in clusters for index in cluster)
        assert covered == list(range(40))

    def test_a_cluster_is_no_wider_than_its_radius_from_its_seed(self, rng):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(8)
        points = X.stations(count=30, rng=rng)
        for cluster in X.cluster_points(points, 0.5):
            seed = points[cluster[0]]
            for index in cluster:
                assert X.geodesic_distance(seed, points[index]) <= 0.5 + 1e-12

    def test_paths_respect_a_minimum_separation(self, rng):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Lebesgue as SphereLebesgue

        X = SphereLebesgue(8)
        paths = X.source_receiver_paths(
            sources=5, receivers=4, minimum_separation=0.4, rng=rng
        )
        assert paths
        assert all(X.geodesic_distance(a, b) > 0.4 for a, b in paths)
        assert len(paths) < 20  # some pairs were dropped


class TestCorrelatedMeasures:
    """Several fields on one domain, correlated scale by scale."""

    @staticmethod
    def space():
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        return Sobolev(10, 2.0, 0.2)

    @pytest.mark.parametrize("correlation", [0.0, 0.7, -0.9])
    def test_the_requested_correlation_appears_in_samples(self, correlation, rng):
        X = self.space()
        first = X.sobolev_symbol(-2.0, 0.2)
        second = X.heat_symbol(0.02)
        measure = X.correlated_measure_from_correlations(
            [first, second],
            np.array([[1.0, correlation], [correlation, 1.0]]),
            labels=("u", "v"),
        )
        draws = [measure.sample(rng=rng) for _ in range(3000)]
        u = np.array([X.to_components(draw[0]) for draw in draws])
        v = np.array([X.to_components(draw[1]) for draw in draws])
        empirical = np.corrcoef(u[:, 5], v[:, 5])[0, 1]
        assert empirical == pytest.approx(correlation, abs=0.06)

    def test_the_marginals_are_the_invariant_measures(self, rng):
        X = self.space()
        first = X.sobolev_symbol(-2.0, 0.2)
        measure = X.correlated_measure_from_correlations(
            [first, X.heat_symbol(0.02)], np.array([[1.0, 0.5], [0.5, 1.0]])
        )
        block = measure.covariance.matrix(form="components")[: X.dim, : X.dim]
        assert np.allclose(np.diag(block), first)

    def test_sampling_comes_from_the_factor(self, rng):
        """An extended Karhunen-Loeve expansion, without writing one."""
        X = self.space()
        measure = X.correlated_measure_from_correlations(
            [X.heat_symbol(0.02), X.heat_symbol(0.02)],
            np.array([[1.0, 0.8], [0.8, 1.0]]),
        )
        assert measure.can_sample
        draw = measure.sample(rng=rng)
        assert len(draw) == 2

    def test_a_correlation_beyond_one_is_refused(self):
        X = self.space()
        with pytest.raises(ValueError, match="positive semidefinite"):
            X.correlated_measure_from_correlations(
                [X.heat_symbol(0.02), X.heat_symbol(0.02)],
                np.array([[1.0, 1.5], [1.5, 1.0]]),
            )

    def test_a_correlation_may_vary_with_scale(self, rng):
        """The point of the construction, as against one number times two
        marginals."""
        X = self.space()
        varying = np.zeros((X.dim, 2, 2))
        varying[:, 0, 0] = varying[:, 1, 1] = 1.0
        varying[:, 0, 1] = varying[:, 1, 0] = np.where(X.degrees < 3, 0.9, 0.0)
        measure = X.correlated_measure_from_correlations(
            [X.heat_symbol(0.02), X.heat_symbol(0.02)], varying
        )
        components = measure.covariance.matrix(form="components")
        cross = np.diag(components[: X.dim, X.dim :])
        assert np.all(cross[X.degrees < 3] > 0.0)
        assert np.allclose(cross[X.degrees >= 3], 0.0)

    def test_a_non_symmetric_slice_is_refused(self):
        X = self.space()
        sigma = np.zeros((X.dim, 2, 2))
        sigma[:, 0, 0] = sigma[:, 1, 1] = 1.0
        sigma[:, 0, 1] = 0.5
        with pytest.raises(ValueError, match="symmetric"):
            X.correlated_measure(sigma)

    def test_a_wrong_shape_is_refused(self):
        X = self.space()
        with pytest.raises(ValueError, match="Expected shape"):
            X.correlated_measure(np.zeros((3, 2, 2)))


class TestDeflatedDiagonal:
    @staticmethod
    def operator(space, rng, decay=0.6):
        size = space.dim
        rotation, _ = np.linalg.qr(rng.normal(size=(size, size)))
        spectrum = np.array([100.0 * decay**index for index in range(size)]) + 1e-4
        matrix = rotation @ np.diag(spectrum) @ rotation.T
        return LinearOperator.from_derivative_matrix(
            space,
            space,
            matrix,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
        )

    @pytest.mark.parametrize("form", ["galerkin", "components"])
    def test_deflation_reduces_the_error(self, form, rng):
        """The whole point: the estimator's variance is set by the operator's
        size, not by the size of what it is failing to resolve."""
        from pygeoinf2.numerics.randomised import deflated_diagonal

        X = make_weighted_space()
        operator = self.operator(X, rng)
        truth = np.diag(operator.matrix(form=form))

        def error(rank):
            estimates = [
                deflated_diagonal(
                    operator,
                    rank=rank,
                    samples=80,
                    form=form,
                    rng=np.random.default_rng(seed),
                )
                for seed in range(4)
            ]
            return np.mean(
                [np.abs(e - truth).max() / np.abs(truth).max() for e in estimates]
            )

        assert error(X.dim - 1) < 0.2 * error(0)

    def test_full_rank_deflation_is_essentially_exact(self, rng):
        from pygeoinf2.numerics.randomised import deflated_diagonal

        X = make_weighted_space()
        operator = self.operator(X, rng)
        truth = np.diag(operator.matrix(form="galerkin"))
        estimate = deflated_diagonal(operator, rank=X.dim, samples=10, rng=rng)
        assert np.allclose(estimate, truth, rtol=1e-6)

    def test_zero_rank_is_the_undeflated_estimator(self, rng):
        from pygeoinf2.numerics.randomised import deflated_diagonal, random_diagonal

        X = make_weighted_space()
        operator = self.operator(X, rng)
        first = deflated_diagonal(
            operator, rank=0, samples=40, rng=np.random.default_rng(3)
        )
        second = random_diagonal(operator, samples=40, rng=np.random.default_rng(3))
        assert np.allclose(first, second)

    def test_a_negative_rank_is_refused(self, rng):
        from pygeoinf2.numerics.randomised import deflated_diagonal

        X = make_weighted_space()
        with pytest.raises(ValueError, match="non-negative"):
            deflated_diagonal(self.operator(X, rng), rank=-1)

    def test_the_pointwise_variance_of_a_general_measure(self, rng):
        """Exact for a few points; deflated when there are many."""
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        X = Sobolev(12, 2.0, 0.2)
        symbol = X.sobolev_symbol(-2.0, 0.2)
        measure = X.invariant_measure(symbol)
        points = X.random_points(8, rng=rng)
        exact = X.pointwise_variance_at(measure, points)
        # an invariant measure has a closed form, which is the check
        assert np.allclose(exact, X.pointwise_variance(symbol))
        estimate = X.pointwise_variance_at(
            measure, points, rank=4, samples=300, rng=rng
        )
        assert np.allclose(estimate, exact, rtol=0.25)


class TestPreconditioners:
    """Structural approximations to an inverse, and when each is worth it."""

    @staticmethod
    def spread(space, rng, decay=0.9):
        size = space.dim
        rotation, _ = np.linalg.qr(rng.normal(size=(size, size)))
        spectrum = np.array([1000.0 * decay**index for index in range(size)]) + 5.0
        return LinearOperator.from_derivative_matrix(
            space,
            space,
            rotation @ np.diag(spectrum) @ rotation.T,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )

    @staticmethod
    def banded_operator(space, rng):
        from scipy.sparse import diags

        size = space.dim
        band = diags(
            [
                np.full(size - 1, -1.0),
                np.full(size, 4.0) + 0.01 * np.arange(size),
                np.full(size - 1, -1.0),
            ],
            [-1, 0, 1],
        ).toarray()
        matrix = band + 1e-3 * rng.normal(size=(size, size))
        matrix = 0.5 * (matrix + matrix.T) + 2.0 * np.identity(size)
        return LinearOperator.from_derivative_matrix(
            space,
            space,
            matrix,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )

    @staticmethod
    def sphere(lmax=8):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        return Sobolev(lmax, 2.0, 0.2)

    @staticmethod
    def iterations(operator, preconditioner, vector):
        result = CGSolver(rtol=1e-10, preconditioner=preconditioner, strict=False)(
            operator
        ).solve(vector)
        space = operator.domain
        residual = space.norm(space.subtract(operator(result.solution), vector))
        return result.iterations, residual / space.norm(vector)

    def test_the_spectral_one_helps_a_decaying_spectrum(self, rng):
        from pygeoinf2.numerics.preconditioners import SpectralPreconditioner

        space = self.sphere()
        operator = self.spread(space, rng)
        vector = space.random(rng=rng)
        plain, _ = self.iterations(operator, None, vector)
        for rank in (20, 50):
            count, residual = self.iterations(
                operator,
                SpectralPreconditioner(rank=rank, rng=np.random.default_rng(1)),
                vector,
            )
            assert residual < 1e-8
            assert count < plain
        assert (
            self.iterations(
                operator,
                SpectralPreconditioner(rank=50, rng=np.random.default_rng(1)),
                vector,
            )[0]
            < self.iterations(
                operator,
                SpectralPreconditioner(rank=20, rng=np.random.default_rng(1)),
                vector,
            )[0]
        )

    def test_the_banded_one_helps_a_banded_operator(self, rng):
        from pygeoinf2.numerics.preconditioners import BandedPreconditioner

        space = self.sphere()
        operator = self.banded_operator(space, rng)
        vector = space.random(rng=rng)
        plain, _ = self.iterations(operator, None, vector)
        count, residual = self.iterations(operator, BandedPreconditioner(1), vector)
        assert residual < 1e-8
        assert count < plain / 5

    def test_the_fast_probe_agrees_where_it_is_valid(self, rng):
        """The two probes coincide on a genuinely banded operator."""
        from pygeoinf2.numerics.preconditioners import BandedPreconditioner

        space = self.sphere()
        operator = self.banded_operator(space, rng)
        vector = space.random(rng=rng)
        exact = self.iterations(operator, BandedPreconditioner(3), vector)
        fast = self.iterations(
            operator, BandedPreconditioner(3, probe="banded"), vector
        )
        assert exact[0] == fast[0]

    def test_it_can_make_matters_worse_on_a_dense_operator(self, rng):
        """Recorded, not guarded against: nothing here can detect the structure
        for you, which is why the bandwidth is a required argument."""
        from pygeoinf2.numerics.preconditioners import BandedPreconditioner

        space = self.sphere()
        operator = self.spread(space, rng)
        vector = space.random(rng=rng)
        plain, _ = self.iterations(operator, None, vector)
        count, residual = self.iterations(
            operator, BandedPreconditioner(3, probe="banded"), vector
        )
        assert residual > 1e-3  # it did not converge at all

    def test_the_block_one_partitions_the_components(self, rng):
        from pygeoinf2.numerics.preconditioners import BlockPreconditioner

        space = self.sphere()
        operator = self.spread(space, rng)
        vector = space.random(rng=rng)
        blocks = [
            list(range(start, min(start + 9, space.dim)))
            for start in range(0, space.dim, 9)
        ]
        count, residual = self.iterations(operator, BlockPreconditioner(blocks), vector)
        assert residual < 1e-8

    def test_blocks_that_do_not_partition_are_refused(self, rng):
        from pygeoinf2.numerics.preconditioners import BlockPreconditioner

        space = self.sphere(4)
        operator = self.spread(space, rng)
        with pytest.raises(ValueError, match="partition"):
            BlockPreconditioner([[0, 1]])(operator)

    def test_a_nonsense_rank_or_bandwidth_is_refused(self):
        from pygeoinf2.numerics.preconditioners import (
            BandedPreconditioner,
            SpectralPreconditioner,
        )

        with pytest.raises(ValueError, match="rank must be positive"):
            SpectralPreconditioner(rank=0)
        with pytest.raises(ValueError, match="bandwidth"):
            BandedPreconditioner(-1)
