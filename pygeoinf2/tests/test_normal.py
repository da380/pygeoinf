"""
The normal operator and the preconditioners built against it.

A preconditioner is only ever an approximate inverse, so most of what can be
said about one is a matter of degree. The tests that are not are the ones worth
writing, and there are three kinds here:

* an identity that must hold exactly (Woodbury, and the localised
  preconditioner at full rank on a diagonal error);
* two routes to the same number that were computed differently (the cheap
  diagonal identity against the generic Jacobi one);
* an answer that must not depend on the preconditioner at all (the posterior).

Each runs on a weighted data space as well as a Euclidean one, because a
diagonal is a statement about a basis and the Galerkin form is the one that
makes a self-adjoint operator symmetric. That distinction is invisible when the
metric is the identity, which is where it would otherwise hide. See DESIGN.md
section 23.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.inference import (
    LinearForwardProblem,
    LinearGaussianInversion,
    LocalisedPreconditioner,
    NormalDiagonalPreconditioner,
    NormalOperator,
)
from pygeoinf2.numerics.preconditioners import (
    JacobiPreconditioner,
    WoodburyPreconditioner,
)
from pygeoinf2.numerics.solvers import CGSolver, CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.traits import Traits

from .conftest import make_weighted_space


def dense(space, matrix):
    return LinearOperator.from_derivative_matrix(
        space,
        space,
        matrix,
        traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
    )


def positive(space, rng, scale=1.0):
    root = rng.normal(size=(space.dim, space.dim))
    return dense(space, scale * (root @ root.T + space.dim * np.identity(space.dim)))


@pytest.fixture(params=["euclidean", "weighted"])
def setup(request, rng):
    """A linear Gaussian problem whose data space may carry a metric."""
    model = EuclideanSpace(6)
    data = EuclideanSpace(8) if request.param == "euclidean" else make_weighted_space()
    forward = LinearOperator.from_derivative_matrix(
        model, data, rng.normal(size=(data.dim, model.dim))
    )
    chol = CholeskySolver()
    covariance = positive(model, rng)
    prior = GaussianMeasure(model, covariance=covariance, precision=chol(covariance))
    noise = dense(data, np.diag(rng.uniform(0.5, 2.0, data.dim)))
    error = GaussianMeasure(data, covariance=noise, precision=chol(noise))
    return forward, prior, error


@pytest.fixture
def normal(setup):
    forward, prior, error = setup
    return NormalOperator(forward, prior, error=error, formalism="data_space")


class TestNormalOperator:
    """It must behave as the assembled operator, and remember its factors."""

    def test_it_equals_the_assembly_it_stands_for(self, normal, setup, rng):
        forward, prior, error = setup
        data = forward.codomain
        reference = forward @ prior.covariance @ forward.adjoint + error.covariance
        for _ in range(5):
            vector = data.random(rng=rng)
            assert data.norm(
                data.subtract(normal(vector), reference(vector))
            ) == pytest.approx(0.0, abs=1e-10 * data.norm(vector))

    def test_the_two_formalisms_give_the_same_posterior(self, setup, rng):
        """The whole content of the choice is which is cheaper."""
        forward, prior, error = setup
        problem = LinearForwardProblem(forward, error=error)
        model, data = forward.domain, forward.codomain
        observed = data.random(rng=rng)
        answers = {
            name: LinearGaussianInversion(problem, prior, formalism=name)(observed)
            for name in ("model_space", "data_space")
        }
        difference = model.subtract(
            answers["model_space"].expectation, answers["data_space"].expectation
        )
        assert model.norm(difference) == pytest.approx(
            0.0, abs=1e-8 * model.norm(answers["data_space"].expectation)
        )

    def test_a_surrogate_may_live_on_a_smaller_model_space(self, normal, rng):
        """The tomography case: only the data space is shared."""
        coarse = EuclideanSpace(3)
        cheap = LinearOperator.from_derivative_matrix(
            coarse, normal.data_space, rng.normal(size=(normal.data_space.dim, 3))
        )
        prior = GaussianMeasure(coarse, covariance=positive(coarse, rng))
        surrogate = normal.surrogate(forward=cheap, prior=prior)
        assert surrogate.model_space == coarse
        assert surrogate.data_space == normal.data_space
        assert surrogate.formalism == normal.formalism

    def test_a_surrogate_must_share_the_data_space(self, normal, rng):
        other = EuclideanSpace(normal.data_space.dim + 1)
        cheap = LinearOperator.from_derivative_matrix(
            normal.model_space,
            other,
            rng.normal(size=(other.dim, normal.model_space.dim)),
        )
        with pytest.raises(ValueError, match="share the data space"):
            normal.surrogate(forward=cheap)

    def test_a_new_model_space_needs_a_new_prior(self, normal, rng):
        """Rather than silently inheriting one defined somewhere else."""
        coarse = EuclideanSpace(3)
        cheap = LinearOperator.from_derivative_matrix(
            coarse, normal.data_space, rng.normal(size=(normal.data_space.dim, 3))
        )
        with pytest.raises(ValueError, match="its own prior"):
            normal.surrogate(forward=cheap)

    def test_the_model_space_formalism_says_what_it_needs(self, setup, rng):
        forward, prior, error = setup
        without = GaussianMeasure(forward.domain, covariance=prior.covariance)
        with pytest.raises(ValueError, match="precision"):
            NormalOperator(forward, without, error=error, formalism="model_space")


class TestNormalDiagonalPreconditioner:
    """The cheap identity <v, A Q A* v> == <A* v, Q A* v>."""

    def test_it_reproduces_the_generic_jacobi_preconditioner(self, normal, rng):
        """Two computations of one diagonal: the identity, and the assembled
        Galerkin matrix. They agree or the metric handling is wrong."""
        data = normal.data_space
        cheap = NormalDiagonalPreconditioner()(normal)
        generic = JacobiPreconditioner()(normal)
        for _ in range(10):
            vector = data.random(rng=rng)
            expected = generic(vector)
            assert data.norm(data.subtract(cheap(vector), expected)) == pytest.approx(
                0.0, abs=1e-10 * data.norm(expected)
            )

    def test_blocks_share_one_adjoint_application(self, normal, rng):
        """A coarser answer, and a usable one. The saving is the point: one
        application of A* per block rather than per datum."""
        data = normal.data_space
        blocks = [list(range(i, min(i + 2, data.dim))) for i in range(0, data.dim, 2)]
        blocked = NormalDiagonalPreconditioner(blocks=blocks)(normal)
        vector = data.random(rng=rng)
        assert np.isfinite(data.norm(blocked(vector)))

    def test_a_partial_cover_is_refused(self, normal):
        dim = normal.data_space.dim
        with pytest.raises(ValueError, match="partition the data space"):
            NormalDiagonalPreconditioner(blocks=[[0, 1]])(normal)
        with pytest.raises(ValueError, match="partition the data space"):
            NormalDiagonalPreconditioner(
                blocks=[[0], [0]] + [[i] for i in range(1, dim - 1)]
            )(normal)

    def test_an_assembled_operator_is_refused(self, normal):
        """It cannot do its job without the factors, and says so rather than
        falling back to something that would quietly work worse."""
        with pytest.raises(TypeError, match="still carries them"):
            NormalDiagonalPreconditioner()(normal.assembled)

    def test_the_model_space_formalism_is_refused(self, setup):
        forward, prior, error = setup
        model = NormalOperator(forward, prior, error=error, formalism="model_space")
        with pytest.raises(ValueError, match="data-space normal operator"):
            NormalDiagonalPreconditioner()(model)


class TestLocalisedPreconditioner:
    """Nystrom on the blocks that couple, sparse LU on the assembly."""

    def test_one_full_rank_block_is_the_exact_inverse(self, normal, rng):
        """With a diagonal error covariance, which is what the class documents.
        Nothing is dropped, so nothing may be lost."""
        data = normal.data_space
        exact = LocalisedPreconditioner(
            [list(range(data.dim))], rank=data.dim, rng=np.random.default_rng(3)
        )(normal)
        for _ in range(10):
            vector = data.random(rng=rng)
            assert data.norm(
                data.subtract(normal(exact(vector)), vector)
            ) == pytest.approx(0.0, abs=1e-8 * data.norm(vector))

    def test_a_correlated_error_is_approximated_by_its_diagonal(self, setup, rng):
        """The documented approximation, asserted so that it stays documented:
        the same construction is no longer exact once R has off-diagonal mass."""
        forward, prior, _ = setup
        data = forward.codomain
        root = rng.normal(size=(data.dim, data.dim))
        correlated = GaussianMeasure(
            data,
            covariance=dense(data, root @ root.T + data.dim * np.identity(data.dim)),
        )
        normal = NormalOperator(
            forward, prior, error=correlated, formalism="data_space"
        )
        exact = LocalisedPreconditioner(
            [list(range(data.dim))], rank=data.dim, rng=np.random.default_rng(3)
        )(normal)
        vector = data.random(rng=rng)
        residual = data.norm(data.subtract(normal(exact(vector)), vector))
        assert residual > 1e-3 * data.norm(vector)

    def test_overlapping_blocks_are_allowed(self, normal, rng):
        data = normal.data_space
        overlapping = LocalisedPreconditioner(
            [list(range(0, data.dim - 1)), list(range(1, data.dim))],
            rank=3,
            rng=np.random.default_rng(5),
        )(normal)
        assert np.isfinite(data.norm(overlapping(data.random(rng=rng))))

    def test_out_of_range_blocks_are_refused(self, normal):
        with pytest.raises(ValueError, match="lie in"):
            LocalisedPreconditioner([[0, normal.data_space.dim]])(normal)


class TestPreconditionedInversion:
    """The answer must not depend on how the solve was preconditioned."""

    @pytest.fixture
    def inversion(self, setup):
        forward, prior, error = setup
        problem = LinearForwardProblem(forward, error=error)
        return LinearGaussianInversion(problem, prior, formalism="data_space")

    def test_every_preconditioner_gives_the_same_posterior(
        self, inversion, normal, rng
    ):
        model, data = inversion.problem.model_space, inversion.data_space
        observed = data.random(rng=rng)
        reference = inversion(observed).expectation
        candidates = {
            "jacobi": JacobiPreconditioner(),
            "diagonal": NormalDiagonalPreconditioner(),
            "localised": LocalisedPreconditioner(
                [list(range(data.dim))], rank=4, rng=np.random.default_rng(5)
            ),
            "woodbury": WoodburyPreconditioner.from_normal(
                inversion.normal_operator, solver=CholeskySolver()
            ),
        }
        for name, preconditioner in candidates.items():
            solved = inversion.with_solver(
                CGSolver(rtol=1e-12).with_preconditioner(preconditioner)
            )(observed).expectation
            assert model.norm(model.subtract(solved, reference)) == pytest.approx(
                0.0, abs=1e-8 * model.norm(reference)
            ), name

    def test_woodbury_from_a_normal_operator_is_the_exact_inverse(self, normal, rng):
        """Woodbury is an identity; from_normal must not change that."""
        data = normal.data_space
        approximate = WoodburyPreconditioner.from_normal(
            normal, solver=CholeskySolver()
        )(normal)
        for _ in range(10):
            vector = data.random(rng=rng)
            assert data.norm(
                data.subtract(normal(approximate(vector)), vector)
            ) == pytest.approx(0.0, abs=1e-8 * data.norm(vector))

    def test_from_normal_refuses_an_operator_without_factors(self, normal):
        with pytest.raises(TypeError, match="factors"):
            WoodburyPreconditioner.from_normal(normal.assembled)

    def test_a_surrogate_preconditions_the_true_operator(self, inversion, rng):
        """The point of the whole arrangement, on a small problem: a surrogate
        prior that is merely the true one's diagonal still gives the right
        answer, because a preconditioner cannot change the answer."""
        model = inversion.problem.model_space
        data = inversion.data_space
        true_prior = inversion.prior
        coarse = GaussianMeasure(
            model,
            covariance=dense(
                model, np.diag(np.diag(true_prior.covariance.matrix(form="galerkin")))
            ),
        )
        surrogate = inversion.surrogate(prior=coarse)
        observed = data.random(rng=rng)
        reference = inversion(observed).expectation
        solved = inversion.with_solver(
            CGSolver(rtol=1e-12).with_preconditioner(
                WoodburyPreconditioner.from_normal(surrogate, solver=CholeskySolver())
            )
        )(observed).expectation
        assert model.norm(model.subtract(solved, reference)) == pytest.approx(
            0.0, abs=1e-8 * model.norm(reference)
        )


class TestSurrogateFamily:
    """The rest of v1's surrogate and reduction entry points."""

    @pytest.fixture
    def inversion(self, setup):
        forward, prior, error = setup
        return LinearGaussianInversion(
            LinearForwardProblem(forward, error=error), prior, formalism="data_space"
        )

    def test_a_low_rank_surrogate_is_cheap_and_still_positive(self, inversion, rng):
        surrogate = inversion.low_rank_surrogate(
            prior_rank=3, rng=np.random.default_rng(2)
        )
        data = surrogate.data_space
        for _ in range(5):
            vector = data.random(rng=rng)
            assert data.inner_product(vector, surrogate(vector)) > 0.0

    def test_a_low_rank_measure_keeps_its_expectation(self, setup, rng):
        _, prior, _ = setup
        approximated = prior.low_rank_approximation(
            rank=3, rng=np.random.default_rng(2)
        )
        assert approximated.domain == prior.domain
        assert approximated.can_sample

    def test_low_rank_needs_an_error_measure_to_approximate_one(self, setup):
        forward, prior, _ = setup
        # Model space, because a noise-free underdetermined problem has a
        # singular data-space normal operator: A Q A* has the rank of the model.
        inversion = LinearGaussianInversion(
            LinearForwardProblem(forward), prior, formalism="model_space"
        )
        with pytest.raises(ValueError, match="no data error measure"):
            inversion.low_rank_surrogate(error_rank=2)

    def test_a_parameterised_inversion_lives_on_the_parameter_space(
        self, inversion, rng
    ):
        model = inversion.problem.model_space
        parameters = EuclideanSpace(3)
        parameterisation = LinearOperator.from_derivative_matrix(
            parameters, model, rng.normal(size=(model.dim, 3))
        )
        prior = GaussianMeasure(parameters, covariance=positive(parameters, rng))
        reduced = inversion.parameterised(parameterisation, prior=prior)
        assert reduced.problem.model_space == parameters
        observed = inversion.data_space.random(rng=rng)
        assert parameters.norm(reduced(observed).expectation) >= 0.0


class TestNaming:
    """The inputs must say what the class actually requires."""

    def test_a_non_gaussian_prior_is_refused(self, setup):
        from pygeoinf2.probability.base import ProbabilityMeasure

        forward, prior, error = setup

        class NotGaussian(ProbabilityMeasure):
            def __init__(self, domain):
                self._domain = domain

            @property
            def domain(self):
                return self._domain

            def sample(self, *, rng=None):
                return self._domain.zero

        with pytest.raises(TypeError, match="GaussianMeasure"):
            LinearGaussianInversion(
                LinearForwardProblem(forward, error=error),
                NotGaussian(forward.domain),
            )


class TestInvariantDistancePreconditioner:
    """``A Q A*`` written down from a table of distances.

    Two claims, and both are checkable exactly rather than by degree: that the
    matrix it writes down really is ``A Q A*``, and that what it hands to the
    solver is positive definite. The second is the one v1 got wrong.
    """

    @pytest.fixture(scope="class")
    def sphere(self):
        pytest.importorskip("pyshtools")
        from pygeoinf2.symmetric_space.sphere import Sobolev

        space = Sobolev(32, 2.0, 0.1)
        # Seeded: whether a truncation is indefinite depends on how many pairs
        # fall inside the radius, so points drawn from the global generator
        # would make this test pass or fail by luck.
        points = space.random_points(150, rng=np.random.default_rng(11))
        forward = space.point_evaluation_operator(points, dense=True)
        prior = space.heat_measure(0.02, pointwise_std=1.0)
        error = GaussianMeasure.from_standard_deviation(forward.codomain, 0.05)
        normal = NormalOperator(forward, prior, error=error, formalism="data_space")
        return space, points, forward, prior, normal

    def test_the_distance_formula_reproduces_the_operator(self, sphere):
        """No truncation and no taper: the shortcut must be the thing itself,
        or every use of it is approximating something else."""
        import scipy.sparse as sparse

        space, points, forward, prior, _ = sphere
        exact = (forward @ prior.covariance @ forward.adjoint).matrix(form="galerkin")
        rows, columns, distances = space.pairs_within_distance(
            points, np.pi * space.radius * 1.01, with_distances=True
        )
        values = space.covariance_function(prior, distances)
        built = np.asarray(
            sparse.coo_matrix((values, (rows, columns)), shape=exact.shape).todense()
        )
        assert np.abs(built - exact).max() < 1e-8 * np.abs(exact).max()

    @pytest.mark.parametrize("radius", [0.2, 0.4])
    def test_truncating_without_a_taper_destroys_positive_definiteness(
        self, sphere, radius
    ):
        """The reason v1's version disappointed. Not a marginal loss: the
        truncated matrix has eigenvalues comparable in size to the operator's
        largest, but negative, and an indefinite preconditioner does not slow
        conjugate gradients down, it breaks the recurrence it relies on."""
        import scipy.sparse as sparse

        from pygeoinf2.inference import gaspari_cohn

        space, points, forward, prior, _ = sphere
        size = forward.codomain.dim
        rows, columns, distances = space.pairs_within_distance(
            points, radius, with_distances=True
        )
        values = space.covariance_function(prior, distances)

        def spectrum_minimum(entries):
            matrix = np.asarray(
                sparse.coo_matrix(
                    (entries, (rows, columns)), shape=(size, size)
                ).todense()
            )
            return float(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min())

        assert spectrum_minimum(values) < 0.0
        tapered = values * gaspari_cohn(distances, 0.5 * radius)
        assert spectrum_minimum(tapered) > 0.0

    def test_it_preconditions_without_changing_the_answer(self, sphere, rng):
        from pygeoinf2.inference import InvariantDistancePreconditioner

        space, points, forward, prior, normal = sphere
        data_space = forward.codomain
        error = GaussianMeasure.from_standard_deviation(data_space, 0.05)
        inversion = LinearGaussianInversion(
            LinearForwardProblem(forward, error=error),
            prior,
            formalism="data_space",
            solver=CholeskySolver(),
        )
        observed = data_space.random(rng=rng)
        reference = inversion(observed).expectation
        solved = inversion.with_solver(
            CGSolver(rtol=1e-10, maxiter=4000).with_preconditioner(
                InvariantDistancePreconditioner(space, points, 0.3)
            )
        )(observed).expectation
        assert space.norm(space.subtract(solved, reference)) == pytest.approx(
            0.0, abs=1e-6 * space.norm(reference)
        )

    def test_zero_distance_gives_a_scalar_and_so_does_nothing(self, sphere, rng):
        """An invariant prior has one pointwise variance, so with uniform noise
        the diagonal preconditioner is a multiple of the identity — and a
        multiple of the identity cannot change what conjugate gradients does.
        v1 had a special case for this; it is only worth anything when the
        noise varies between data."""
        from pygeoinf2.inference import InvariantDistancePreconditioner

        space, points, forward, prior, normal = sphere
        data_space = forward.codomain
        preconditioner = InvariantDistancePreconditioner(space, points, 0.0)(normal)
        vector = data_space.random(rng=rng)
        applied = preconditioner(vector)
        ratios = data_space.to_components(applied) / data_space.to_components(vector)
        assert np.allclose(ratios, ratios[0])

    def test_the_points_must_be_the_data(self, sphere):
        from pygeoinf2.inference import InvariantDistancePreconditioner

        space, points, forward, prior, normal = sphere
        with pytest.raises(ValueError, match="points are the data"):
            InvariantDistancePreconditioner(space, points[:-1], 0.3)(normal)


class TestGaspariCohn:
    """The taper itself, which is what makes truncation legitimate."""

    def test_it_is_one_at_zero_and_zero_beyond_its_support(self):
        from pygeoinf2.inference import gaspari_cohn

        assert gaspari_cohn(np.zeros(1), 1.0)[0] == pytest.approx(1.0)
        assert gaspari_cohn(np.array([2.0, 2.5, 10.0]), 1.0) == pytest.approx(0.0)

    def test_it_is_continuous_where_the_pieces_meet(self):
        from pygeoinf2.inference import gaspari_cohn

        left = gaspari_cohn(np.array([1.0 - 1e-9]), 1.0)[0]
        right = gaspari_cohn(np.array([1.0 + 1e-9]), 1.0)[0]
        assert left == pytest.approx(right, abs=1e-7)

    def test_it_is_positive_definite(self):
        """Which is the only property that matters here. Checked as a Gram
        matrix of separations on a line, where a Fourier argument would say the
        same thing less directly."""
        from pygeoinf2.inference import gaspari_cohn

        positions = np.linspace(0.0, 3.0, 60)
        separations = np.abs(positions[:, None] - positions[None, :])
        matrix = gaspari_cohn(separations, 1.0)
        assert np.linalg.eigvalsh(matrix).min() > -1e-10

    def test_a_non_positive_length_is_refused(self):
        from pygeoinf2.inference import gaspari_cohn

        with pytest.raises(ValueError, match="taper length"):
            gaspari_cohn(np.zeros(1), 0.0)
