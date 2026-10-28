"""The preconditioners: the sparse approximation they share, the generic
and structure-aware ones, the Woodbury form of a surrogate, the
column-thresholded and the deflated diagonal."""

import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.numerics.solvers import (
    CGSolver,
    CholeskySolver,
    LUSolver,
)
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.testing import check_operator
from pygeoinf2.traits import Traits

from .conftest import (
    make_dense_metric_space,
    make_weighted_space,
    spd_operator,
)


class TestSparseMatrices:
    def test_a_sparse_matrix_needs_no_separate_constructor(self, rng):
        """Both support ``@`` and ``.T``, which is all the constructors use."""
        X = make_weighted_space()
        Y = EuclideanSpace(X.dim)
        dense = np.diag(np.arange(1.0, X.dim + 1))
        sparse = LinearOperator.from_matrix(X, Y, csr_matrix(dense), form="galerkin")
        check_operator(sparse, rng=rng)
        reference = LinearOperator.from_matrix(X, Y, dense, form="galerkin")
        x, y = X.random(rng=rng), Y.random(rng=rng)
        assert np.allclose(sparse(x), reference(x))
        assert np.allclose(
            X.to_components(sparse.adjoint(y)),
            X.to_components(reference.adjoint(y)),
        )

    def test_a_wrong_shape_is_still_refused(self):
        X = make_weighted_space()
        with pytest.raises(ValueError, match="expected"):
            LinearOperator.from_matrix(
                X, X, csr_matrix(np.zeros((2, 2))), form="components"
            )


class TestPreconditioners:
    """Structural approximations to an inverse, and when each is worth it."""

    @staticmethod
    def spread(space, rng, decay=0.9):
        size = space.dim
        rotation, _ = np.linalg.qr(rng.normal(size=(size, size)))
        spectrum = np.array([1000.0 * decay**index for index in range(size)]) + 5.0
        return LinearOperator.from_matrix(
            space,
            space,
            rotation @ np.diag(spectrum) @ rotation.T,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
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
        return LinearOperator.from_matrix(
            space,
            space,
            matrix,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
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
        # The probe is only taken on an operator that must be applied for
        # its band; one holding its matrix reads it and ignores the setting.
        probed = LinearOperator.from_callables(
            space, space, operator, adjoint=operator.adjoint, traits=operator.traits
        )
        fast = self.iterations(probed, BandedPreconditioner(3, probe="banded"), vector)
        assert exact[0] == fast[0]

    def test_it_can_make_matters_worse_on_a_dense_operator(self, rng):
        """Recorded, not guarded against: nothing here can detect the structure
        for you, which is why the bandwidth is a required argument.

        The operator is wrapped as callables so that the *probe* is what is
        being tested. An operator holding its own matrix reads the band off it
        exactly, which makes the banded preconditioner a genuine approximation
        rather than a contaminated one — and then it converges.
        """
        from pygeoinf2.numerics.preconditioners import BandedPreconditioner

        space = self.sphere()
        assembled = self.spread(space, rng)
        operator = LinearOperator.from_callables(
            space,
            space,
            assembled,
            adjoint=assembled,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        vector = space.random(rng=rng)
        # Whether the contaminated preconditioner is refused outright (a
        # negative inner product in the recurrence) or merely fails to
        # converge within the cap depends on rounding along a run that is
        # going nowhere; either outcome is the point, and neither is success.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                count, residual = self.iterations(
                    operator, BandedPreconditioner(3, probe="banded"), vector
                )
            except ValueError as refused:
                assert "preconditioner" in str(refused)
            else:
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

    def test_blocks_need_not_cover_everything(self, rng):
        """What is left out keeps its own diagonal, so the preconditioner is
        still invertible there and degrades to Jacobi rather than refusing.
        v2 required a partition, which rules out the clusterings a real
        geometry gives."""
        from pygeoinf2.numerics.preconditioners import BlockPreconditioner

        space = self.sphere(4)
        operator = self.spread(space, rng)
        count, residual = self.iterations(
            operator, BlockPreconditioner([[0, 1]]), space.random(rng=rng)
        )
        assert residual < 1e-8

    def test_blocks_may_overlap(self, rng):
        """A point near two clusters belongs to both. The entries come from
        the operator itself rather than from summed block inverses, so an
        index in two blocks is not counted twice."""
        from pygeoinf2.numerics.preconditioners import BlockPreconditioner

        space = self.sphere(4)
        operator = self.spread(space, rng)
        vector = space.random(rng=rng)

        blocks = [list(range(0, 12)), list(range(8, 20)), list(range(18, space.dim))]
        count, residual = self.iterations(operator, BlockPreconditioner(blocks), vector)
        assert residual < 1e-8

    def test_a_block_outside_the_space_is_refused(self, rng):
        from pygeoinf2.numerics.preconditioners import BlockPreconditioner

        space = self.sphere(4)
        operator = self.spread(space, rng)
        with pytest.raises(ValueError, match="has"):
            BlockPreconditioner([[0, space.dim]])(operator)
        with pytest.raises(ValueError, match="non-negative"):
            BlockPreconditioner([[-1, 0]])

    def test_a_nonsense_rank_or_bandwidth_is_refused(self):
        from pygeoinf2.numerics.preconditioners import (
            BandedPreconditioner,
            SpectralPreconditioner,
        )

        with pytest.raises(ValueError, match="rank must be positive"):
            SpectralPreconditioner(rank=0)
        with pytest.raises(ValueError, match="bandwidth"):
            BandedPreconditioner(-1)


class TestWoodburyPreconditioner:
    """Invert a normal operator through the other space.

    Woodbury is an *identity*, so the decisive test is not that the
    preconditioner helps but that it is exactly right when nothing is
    approximated. Anything less than machine precision there is a bug, and no
    amount of iteration-count improvement would reveal it.
    """

    @staticmethod
    def psd(space, rng, scale=1.0):
        return spd_operator(space, rng, scale=scale)

    @pytest.fixture(
        params=[
            "euclidean",
            "weighted-model",
            "weighted-data",
            "dense-model",
            "dense-data",
        ]
    )
    def pieces(self, request, rng):
        """A forward problem, on spaces whose metric is not the identity.

        The whole construction is written in Hilbert adjoints, so a
        non-trivial metric is where a components-for-Galerkin slip would
        show, and only a dense Gram matrix shows every kind of slip.
        """
        model, data = {
            "euclidean": lambda: (EuclideanSpace(5), EuclideanSpace(3)),
            "weighted-model": lambda: (make_weighted_space(), EuclideanSpace(3)),
            "weighted-data": lambda: (EuclideanSpace(5), make_weighted_space()),
            "dense-model": lambda: (make_dense_metric_space(5), EuclideanSpace(3)),
            "dense-data": lambda: (EuclideanSpace(5), make_dense_metric_space(3)),
        }[request.param]()
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
        )
        return forward, self.psd(model, rng), self.psd(data, rng, 0.3)

    def test_exact_pieces_give_the_exact_inverse(self, pieces, rng):
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        forward, prior, noise = pieces
        model, data = forward.domain, forward.codomain
        exact = CholeskySolver()
        prior_inverse, noise_inverse = exact(prior), exact(noise)
        wood = WoodburyPreconditioner(
            forward,
            prior,
            noise,
            solver=exact,
            prior_inverse=prior_inverse,
            noise_inverse=noise_inverse,
        )
        normal_model = (
            prior_inverse + forward.adjoint @ noise_inverse @ forward
        ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
        normal_data = (noise + forward @ prior @ forward.adjoint).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )

        for approximate, normal, space in [
            (wood.model_form(), normal_model, model),
            (wood.data_form(), normal_data, data),
            # Through the solver interface, which must pick the right identity
            # from the space the operator acts on.
            (wood(normal_model), normal_model, model),
            (wood(normal_data), normal_data, data),
        ]:
            for _ in range(5):
                vector = space.random(rng=rng)
                back = approximate(normal(vector))
                assert space.norm(space.subtract(back, vector)) == pytest.approx(
                    0.0, abs=1e-10 * space.norm(vector)
                )

    @staticmethod
    def singular(space, rng):
        """A covariance of rank ``dim - 1``: a prior with no inverse at all."""
        root = rng.normal(size=(space.dim, space.dim))
        root[:, -1] = 0.0
        return LinearOperator.from_matrix(
            space,
            space,
            root @ root.T,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
            form="galerkin",
        )

    def test_damping_gives_the_data_form_of_the_damped_prior(self, pieces, rng):
        """Most priors approximate measures on function spaces and have no
        usable inverse; the data form needs one. ``prior_damping`` makes it
        the exact inverse of ``Q + d I``, which is self-adjoint, positive
        definite and as close to ``Q^-1`` as a preconditioner needs. The model
        form never inverts ``Q`` and is left as it was."""
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        forward, _, noise = pieces
        model, data = forward.domain, forward.codomain
        prior, damping = self.singular(model, rng), 0.1
        wood = WoodburyPreconditioner(
            forward, prior, noise, solver=CholeskySolver(), prior_damping=damping
        )
        damped = prior + damping * LinearOperator.identity(model)
        normal_data = (noise + forward @ damped @ forward.adjoint).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        for approximate, normal in [
            (wood.data_form(), normal_data),
            (wood(normal_data), normal_data),
        ]:
            for _ in range(5):
                vector = data.random(rng=rng)
                back = approximate(normal(vector))
                assert data.norm(data.subtract(back, vector)) == pytest.approx(
                    0.0, abs=1e-10 * data.norm(vector)
                )
        # The model form never inverts Q, so the damping is not its business.
        undamped = WoodburyPreconditioner(
            forward, prior, noise, solver=CholeskySolver()
        )
        for _ in range(5):
            vector = model.random(rng=rng)
            assert model.norm(
                model.subtract(wood.model_form()(vector), undamped.model_form()(vector))
            ) == pytest.approx(0.0, abs=1e-12 * model.norm(vector))

    def test_from_normal_damps_instead_of_reading_a_precision(self, pieces, rng):
        """With a damping the precision a measure carries is not picked up:
        the point of damping is that ``Q`` has no usable inverse of its own."""
        from pygeoinf2.numerics.normal import NormalOperator
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        forward, _, noise = pieces
        model, data = forward.domain, forward.codomain
        prior, damping = self.singular(model, rng), 0.1
        # A precision that is plainly wrong, to prove it is ignored.
        normal = NormalOperator(
            forward,
            GaussianMeasure(
                model, covariance=prior, precision=5.0 * LinearOperator.identity(model)
            ),
            error=GaussianMeasure(data, covariance=noise),
        )
        wood = WoodburyPreconditioner.from_normal(
            normal, solver=CholeskySolver(), prior_damping=damping
        )
        damped = prior + damping * LinearOperator.identity(model)
        normal_data = (noise + forward @ damped @ forward.adjoint).with_traits(
            Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        approximate = wood.data_form()
        for _ in range(5):
            vector = data.random(rng=rng)
            back = approximate(normal_data(vector))
            assert data.norm(data.subtract(back, vector)) == pytest.approx(
                0.0, abs=1e-10 * data.norm(vector)
            )

    def test_damping_is_refused_when_it_contradicts(self, pieces):
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        forward, prior, noise = pieces
        with pytest.raises(ValueError, match="non-negative"):
            WoodburyPreconditioner(forward, prior, noise, prior_damping=-1.0)
        with pytest.raises(ValueError, match="one or the other"):
            WoodburyPreconditioner(
                forward,
                prior,
                noise,
                prior_inverse=CholeskySolver()(prior),
                prior_damping=0.1,
            )

    def test_the_model_form_never_inverts_the_covariances(self, pieces, rng):
        """Which is what lets it survive a prior whose inverse is unbounded.

        The prior and noise are given with no inverse and no solver for one,
        and the model form still has to work.
        """
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        forward, prior, noise = pieces
        wood = WoodburyPreconditioner(forward, prior, noise, solver=CholeskySolver())
        approximate = wood.model_form()
        vector = forward.domain.random(rng=rng)
        assert np.isfinite(forward.domain.norm(approximate(vector)))

    @pytest.mark.parametrize("formalism", ["data_space", "model_space"])
    def test_the_family_is_exact_on_tikhonov_and_damped_members(
        self, pieces, formalism, rng
    ):
        """Along a family the multiplier is read as the factor on the noise
        covariance and the prior covariance is the preconditioner's own, so
        with exact pieces the bound family inverts every member exactly,
        whether the member is a damped normal operator or a Tikhonov one,
        in either space. Not ``t`` times the inverse: conjugate gradients
        is blind to a constant factor, a solver is not."""
        from pygeoinf2.numerics.normal import DampedNormalOperator
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner
        from pygeoinf2.numerics.tikhonov import TikhonovNormalOperator

        forward, _, noise_covariance = pieces
        model, data = forward.domain, forward.codomain
        identity = LinearOperator.identity(model)
        noise = GaussianMeasure(
            data, covariance=noise_covariance, precision=LUSolver()(noise_covariance)
        )
        wood = WoodburyPreconditioner(
            forward,
            identity,
            noise.covariance,
            prior_inverse=identity,
            noise_inverse=noise.precision,
            solver=CholeskySolver(),
        )
        template = TikhonovNormalOperator(
            forward, 0.0, error=noise, formalism=formalism
        )
        bound = wood.for_family(template.base, template.shift)
        for multiplier in (0.1, 1.0, 7.0):
            tikhonov = TikhonovNormalOperator(
                forward, multiplier, error=noise, formalism=formalism
            )
            damped = DampedNormalOperator(
                forward,
                identity,
                noise.covariance,
                multiplier,
                prior_precision=identity,
                noise_precision=noise.precision,
                formalism=formalism,
            )
            for member in (tikhonov, damped):
                space = member.domain
                vector = space.random(rng=rng)
                back = bound.at(multiplier, member)(member(vector))
                assert space.norm(space.subtract(back, vector)) <= 1e-10 * space.norm(
                    vector
                ), (type(member).__name__, formalism, multiplier)

    def test_the_data_kernel_is_read_once_as_a_matrix_along_a_family(self, rng):
        """``A* R^-1 A`` does not move with the multiplier, so the family
        holds it as one stored matrix and each member's inner operator is a
        sum of stored matrices rather than a product assembled afresh. The
        kernel is read only where it is no larger than the factors, which
        is the case it is for: more data than coarse model coefficients."""
        from pygeoinf2.algebra.operators import MatrixLinearOperator
        from pygeoinf2.numerics.normal import DampedNormalOperator
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        model, data = EuclideanSpace(3), make_weighted_space()
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
        )
        prior, noise_covariance = self.psd(model, rng), self.psd(data, rng, 0.3)
        prior_inverse = LUSolver()(prior)
        noise_inverse = LUSolver()(noise_covariance)
        wood = WoodburyPreconditioner(
            forward,
            prior,
            noise_covariance,
            prior_inverse=prior_inverse,
            noise_inverse=noise_inverse,
            solver=CholeskySolver(),
        )
        kernel = wood.data_kernel()
        assert isinstance(kernel, MatrixLinearOperator)
        expected = forward.adjoint @ noise_inverse @ forward
        vector = model.random(rng=rng)
        assert model.norm(model.subtract(kernel(vector), expected(vector))) <= (
            1e-10 * model.norm(expected(vector))
        )
        base = forward @ prior @ forward.adjoint
        bound = wood.for_family(base, noise_covariance)
        for multiplier in (0.5, 2.0):
            member = DampedNormalOperator(
                forward,
                prior,
                noise_covariance,
                multiplier,
                prior_precision=prior_inverse,
                noise_precision=noise_inverse,
            )
            back = bound.at(multiplier, member)(member(data.random(rng=rng)))
            assert np.all(np.isfinite(data.to_components(back)))
        assert isinstance(bound._kernel, MatrixLinearOperator)

    def test_a_surrogate_prior_still_preconditions(self, rng):
        """The practical case: cheap stand-in pieces, exact outer problem.

        Correctness never depends on how good the surrogate is — only the
        iteration count does, which is the property being measured here.
        """
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        model, data = EuclideanSpace(120), EuclideanSpace(8)
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(8, 120)), form="galerkin"
        )
        decay = np.exp(-np.arange(120) / 6.0) + 1e-4

        def diagonal(space, values):
            return LinearOperator.from_matrix(
                space,
                space,
                np.diag(values),
                traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
                form="galerkin",
            )

        noise_inverse = diagonal(data, np.full(8, 100.0))
        normal = (
            diagonal(model, 1.0 / decay) + forward.adjoint @ noise_inverse @ forward
        ).with_traits(Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE)
        vector = model.random(rng=rng)

        plain = CGSolver(rtol=1e-10, max_iterations=2000)(normal).solve(vector)
        # A coarse staircase in place of the true exponential decay.
        staircase = np.exp(-(np.arange(120) // 10) * 10 / 6.0) + 1e-4
        surrogate = WoodburyPreconditioner(
            forward,
            diagonal(model, staircase),
            diagonal(data, np.full(8, 0.01)),
            solver=CholeskySolver(),
        )
        helped = (
            CGSolver(rtol=1e-10, max_iterations=2000)
            .with_preconditioner(surrogate)(normal)
            .solve(vector)
        )
        assert helped.iterations < plain.iterations / 4
        assert model.norm(
            model.subtract(plain.solution, helped.solution)
        ) == pytest.approx(0.0, abs=1e-8 * model.norm(plain.solution))

    def test_mismatched_pieces_are_refused(self, rng):
        from pygeoinf2.numerics.preconditioners import WoodburyPreconditioner

        model, data = EuclideanSpace(5), EuclideanSpace(3)
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(3, 5)), form="galerkin"
        )
        with pytest.raises(ValueError, match="prior covariance"):
            WoodburyPreconditioner(forward, self.psd(data, rng), self.psd(data, rng))
        with pytest.raises(ValueError, match="noise covariance"):
            WoodburyPreconditioner(forward, self.psd(model, rng), self.psd(model, rng))

        wood = WoodburyPreconditioner(
            forward, self.psd(model, rng), self.psd(data, rng)
        )
        other = EuclideanSpace(7)
        with pytest.raises(ValueError, match="which is neither"):
            wood(LinearOperator.identity(other))


class TestColumnThresholdedPreconditioner:
    """Keep the large entries wherever they are, rather than assuming a band."""

    @staticmethod
    def scattered(space, rng):
        """Sparse in this basis, but not banded: the couplings are geometric."""
        positions = rng.uniform(0.0, 1.0, size=(space.dim, 2))
        separations = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        matrix = np.exp(-((separations / 0.15) ** 2)) + space.dim * np.identity(
            space.dim
        )
        return LinearOperator.from_matrix(
            space,
            space,
            matrix,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
            form="galerkin",
        )

    @pytest.fixture(params=["euclidean", "weighted", "dense"])
    def operator(self, request, rng):
        space = {
            "euclidean": lambda: EuclideanSpace(40),
            "weighted": make_weighted_space,
            "dense": lambda: make_dense_metric_space(12),
        }[request.param]()
        return self.scattered(space, rng)

    def test_a_zero_threshold_keeps_everything_and_is_exact(self, operator, rng):
        """The boundary case, which pins the metric handling: with nothing
        dropped this is a sparse *exact* inverse, so it must be one."""
        from pygeoinf2.numerics.preconditioners import ColumnThresholdedPreconditioner

        space = operator.domain
        inverse = ColumnThresholdedPreconditioner(0.0)(operator)
        for _ in range(5):
            vector = space.random(rng=rng)
            assert space.norm(
                space.subtract(inverse(operator(vector)), vector)
            ) == pytest.approx(0.0, abs=1e-9 * space.norm(vector))

    def test_the_thresholded_operator_stays_symmetric(self, operator, rng):
        """Column-wise dropping does not, on its own, produce a symmetric
        matrix, and conjugate gradients needs one."""
        from pygeoinf2.numerics.preconditioners import ColumnThresholdedPreconditioner

        space = operator.domain
        inverse = ColumnThresholdedPreconditioner(0.05)(operator)
        worst = 0.0
        for _ in range(10):
            left, right = space.random(rng=rng), space.random(rng=rng)
            difference = abs(
                space.inner_product(left, inverse(right))
                - space.inner_product(inverse(left), right)
            )
            worst = max(worst, difference / (space.norm(left) * space.norm(right)))
        assert worst < 1e-10

    def test_it_reduces_the_iteration_count(self, operator, rng):
        from pygeoinf2.numerics.preconditioners import ColumnThresholdedPreconditioner

        space = operator.domain
        vector = space.random(rng=rng)
        plain = CGSolver(rtol=1e-12)(operator).solve(vector)
        helped = (
            CGSolver(rtol=1e-12)
            .with_preconditioner(ColumnThresholdedPreconditioner(0.05))(operator)
            .solve(vector)
        )
        assert helped.iterations <= plain.iterations
        assert space.norm(
            space.subtract(plain.solution, helped.solution)
        ) == pytest.approx(0.0, abs=1e-8 * space.norm(plain.solution))

    def test_a_cap_keeps_the_diagonal(self, operator, rng):
        """With one entry per column allowed it must be exactly Jacobi, which
        is the only sane reading of 'keep the largest, and the diagonal'."""
        from pygeoinf2.numerics.preconditioners import (
            ColumnThresholdedPreconditioner,
            JacobiPreconditioner,
        )

        space = operator.domain
        capped = ColumnThresholdedPreconditioner(0.0, max_per_column=1)(operator)
        jacobi = JacobiPreconditioner()(operator)
        vector = space.random(rng=rng)
        assert space.norm(
            space.subtract(capped(vector), jacobi(vector))
        ) == pytest.approx(0.0, abs=1e-10 * space.norm(jacobi(vector)))

    def test_bad_arguments_are_refused(self):
        from pygeoinf2.numerics.preconditioners import ColumnThresholdedPreconditioner

        with pytest.raises(ValueError, match="non-negative"):
            ColumnThresholdedPreconditioner(-1.0)
        with pytest.raises(ValueError, match="the diagonal"):
            ColumnThresholdedPreconditioner(0.1, max_per_column=0)


class TestDeflatedDiagonal:
    @staticmethod
    def operator(space, rng, decay=0.6):
        size = space.dim
        rotation, _ = np.linalg.qr(rng.normal(size=(size, size)))
        spectrum = np.array([100.0 * decay**index for index in range(size)]) + 1e-4
        matrix = rotation @ np.diag(spectrum) @ rotation.T
        return LinearOperator.from_matrix(
            space,
            space,
            matrix,
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_SEMIDEFINITE,
            form="galerkin",
        )

    @pytest.mark.parametrize("form", ["galerkin", "components"])
    def test_deflation_reduces_the_error(self, form, rng):
        """The whole point: the estimator's variance is set by the operator's
        size, not by the size of what it is failing to resolve."""
        from pygeoinf2.numerics.randomized import deflated_diagonal

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
        from pygeoinf2.numerics.randomized import deflated_diagonal

        X = make_weighted_space()
        operator = self.operator(X, rng)
        truth = np.diag(operator.matrix(form="galerkin"))
        estimate = deflated_diagonal(operator, rank=X.dim, samples=10, rng=rng)
        assert np.allclose(estimate, truth, rtol=1e-6)

    def test_zero_rank_is_the_undeflated_estimator(self, rng):
        from pygeoinf2.numerics.randomized import deflated_diagonal, random_diagonal

        X = make_weighted_space()
        operator = self.operator(X, rng)
        first = deflated_diagonal(
            operator, rank=0, samples=40, rng=np.random.default_rng(3)
        )
        second = random_diagonal(operator, samples=40, rng=np.random.default_rng(3))
        assert np.allclose(first, second)

    def test_a_negative_rank_is_refused(self, rng):
        from pygeoinf2.numerics.randomized import deflated_diagonal

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
