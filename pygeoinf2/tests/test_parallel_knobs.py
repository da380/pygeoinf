"""Every independent loop takes ``n_jobs``.

The loops that can run on several cores: the dense point and path
operators, the column probes behind the sparse preconditioners, the
per-block probes of the normal-diagonal preconditioner, the covariance
assembly behind the scipy bridge, the log determinant's Hutchinson probes,
and the support sweep on every route but the dual. Each takes ``n_jobs``
and forwards it to :func:`pygeoinf2.parallel.parallel_map`, and each test
here says the same thing: the answer does not depend on the worker count.
"""

from __future__ import annotations

import numpy as np
import pytest

from pygeoinf2.algebra.operators import LinearOperator
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.geometry.convex import Ball
from pygeoinf2.inference import (
    BackusGilbertParker,
    LinearForwardProblem,
    NormalDiagonalPreconditioner,
    NormalOperator,
)
from pygeoinf2.numerics.functional_calculus import log_determinant
from pygeoinf2.numerics.preconditioners import (
    BandedPreconditioner,
    BlockPreconditioner,
    ColumnThresholdedPreconditioner,
    SpectralPreconditioner,
    sparse_approximation,
)
from pygeoinf2.numerics.solvers import CholeskySolver
from pygeoinf2.probability.gaussian import GaussianMeasure
from pygeoinf2.traits import Traits

from .conftest import make_dense_metric_space, make_weighted_space, spd_operator


def spd(space, rng):
    return spd_operator(space, rng)


def same(space, x, y):
    return space.norm(space.subtract(x, y)) <= 1e-10 * max(space.norm(x), 1.0)


class TestPreconditionerProbes:
    """Column probes, diagonals and the range finder all go to the workers."""

    @pytest.fixture
    def operator(self, rng):
        return spd(make_dense_metric_space(8), rng)

    @pytest.mark.parametrize(
        "build",
        [
            lambda n_jobs: BandedPreconditioner(1, n_jobs=n_jobs),
            lambda n_jobs: BlockPreconditioner(
                [[0, 1, 2, 3], [3, 4, 5], [6, 7]], n_jobs=n_jobs
            ),
            lambda n_jobs: ColumnThresholdedPreconditioner(0.1, n_jobs=n_jobs),
            lambda n_jobs: ColumnThresholdedPreconditioner(
                0.05, max_per_column=3, n_jobs=n_jobs
            ),
        ],
        ids=["banded", "block", "thresholded", "capped"],
    )
    def test_the_sparse_preconditioners_do_not_depend_on_the_job_count(
        self, operator, build, rng
    ):
        space = operator.domain
        serial, parallel = build(None)(operator), build(2)(operator)
        for _ in range(5):
            vector = space.random(rng=rng)
            assert same(space, serial(vector), parallel(vector))

    def test_the_spectral_preconditioner_does_not_depend_on_the_job_count(
        self, operator, rng
    ):
        """The probes are drawn from the generator whichever loop applies
        them, so the same seed gives the same modes."""
        space = operator.domain
        serial = SpectralPreconditioner(rank=3, rng=np.random.default_rng(1))(operator)
        parallel = SpectralPreconditioner(
            rank=3, rng=np.random.default_rng(1), n_jobs=2
        )(operator)
        for _ in range(5):
            vector = space.random(rng=rng)
            assert same(space, serial(vector), parallel(vector))

    def test_the_sparse_approximation_does_not_depend_on_the_job_count(self, operator):
        serial = sparse_approximation(operator, threshold=0.3).matrix(form="galerkin")
        parallel = sparse_approximation(operator, threshold=0.3, n_jobs=2).matrix(
            form="galerkin"
        )
        assert np.allclose(serial, parallel, atol=1e-12)

    def test_the_normal_diagonal_probes_go_to_the_workers(self, rng):
        model, data = EuclideanSpace(6), make_weighted_space()
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
        )
        chol = CholeskySolver()
        covariance = spd(model, rng)
        prior = GaussianMeasure(
            model, covariance=covariance, precision=chol(covariance)
        )
        noise = LinearOperator.from_matrix(
            data,
            data,
            np.diag(rng.uniform(0.5, 2.0, data.dim)),
            form="galerkin",
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        error = GaussianMeasure(data, covariance=noise, precision=chol(noise))
        normal = NormalOperator(forward, prior, error=error, formalism="data_space")
        blocks = [[0, 1], [2, 3]]
        for kwargs in ({}, {"blocks": blocks}):
            serial = NormalDiagonalPreconditioner(**kwargs)(normal)
            parallel = NormalDiagonalPreconditioner(n_jobs=2, **kwargs)(normal)
            for _ in range(5):
                vector = data.random(rng=rng)
                assert same(data, serial(vector), parallel(vector))


class TestAssemblies:
    """The dense covariance behind the scipy bridge and the log determinant."""

    def test_the_multivariate_normal_does_not_depend_on_the_job_count(self, rng):
        space = make_dense_metric_space(5)
        measure = GaussianMeasure(space, covariance=spd(space, rng))
        serial = measure.as_multivariate_normal()
        parallel = measure.as_multivariate_normal(n_jobs=2)
        assert np.allclose(serial.cov, parallel.cov, atol=1e-12)
        assert np.allclose(serial.mean, parallel.mean)

    def test_the_log_determinant_does_not_depend_on_the_job_count(self, rng):
        space = make_dense_metric_space(6)
        operator = spd(space, rng)
        # An operator scipy cannot read the matrix of, so both routes work.
        opaque = LinearOperator.from_callables(
            space, space, operator, adjoint=operator.adjoint, traits=operator.traits
        )
        dense = log_determinant(opaque, method="dense").value
        assert log_determinant(opaque, method="dense", n_jobs=2).value == pytest.approx(
            dense
        )
        serial = log_determinant(
            opaque, method="stochastic", samples=6, rng=np.random.default_rng(3)
        )
        parallel = log_determinant(
            opaque,
            method="stochastic",
            samples=6,
            rng=np.random.default_rng(3),
            n_jobs=2,
        )
        assert parallel.value == pytest.approx(serial.value)
        assert parallel.standard_error == pytest.approx(serial.standard_error)


class TestSupportSweeps:
    """``n_jobs`` on every route of the feasible set's sweep, not the dual only."""

    @pytest.fixture
    def setting(self, rng):
        model = make_weighted_space()
        data_space, property_space = EuclideanSpace(2), EuclideanSpace(2)
        forward = LinearOperator.from_matrix(
            model, data_space, rng.normal(size=(2, model.dim)), form="galerkin"
        )
        target = LinearOperator.from_matrix(
            model, property_space, rng.normal(size=(2, model.dim)), form="galerkin"
        )
        raw = model.random(rng=rng)
        truth = model.scale(2.0 / model.norm(raw), raw)
        angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        directions = [
            property_space.from_components(np.array([np.cos(a), np.sin(a)]))
            for a in angles
        ]
        return model, forward, target, forward(truth), directions

    def test_the_closed_form_takes_workers(self, setting):
        model, forward, target, data, directions = setting
        result = BackusGilbertParker(
            LinearForwardProblem(forward), target, Ball(model, radius=3.0)
        )(data)
        assert result.form == "closed_form"
        serial = result.support_values(directions)
        assert np.allclose(result.support_values(directions, n_jobs=2), serial)

    def test_the_primal_takes_workers(self, setting):
        model, forward, target, data, directions = setting
        result = BackusGilbertParker(
            LinearForwardProblem(forward, error=Ball(forward.codomain, radius=0.1)),
            target,
            Ball(model, radius=3.0),
        )(data)
        serial = result.support_values(directions)
        assert np.allclose(
            result.support_values(directions, n_jobs=2), serial, rtol=1e-6
        )

    @pytest.mark.parametrize("via", ["dual", "splitting"])
    def test_the_general_algorithms_honour_the_workers(self, setting, via):
        """Each worker warm-starts within its own chunk, so the comparison
        is to the algorithm's own tolerance."""
        model, forward, target, data, directions = setting
        result = BackusGilbertParker(
            LinearForwardProblem(forward, error=Ball(forward.codomain, radius=0.1)),
            target,
            Ball(model, radius=3.0),
        )(data)
        serial = result.support_values(directions, via=via)
        fresh = BackusGilbertParker(
            LinearForwardProblem(forward, error=Ball(forward.codomain, radius=0.1)),
            target,
            Ball(model, radius=3.0),
        )(data)
        parallel = fresh.support_values(directions, via=via, n_jobs=2)
        assert np.allclose(parallel, serial, rtol=1e-5, atol=1e-8)


finufft = pytest.importorskip("finufft")

from pygeoinf2.symmetric_space import Sobolev as BoxSobolev  # noqa: E402
from pygeoinf2.symmetric_space.sphere import Sobolev  # noqa: E402


class TestDenseObservationOperators:
    """The dense point, path and ball operators assemble on the workers."""

    @pytest.fixture
    def sphere(self):
        return Sobolev(12, 2.0, 0.2)

    def test_the_basis_matrix_does_not_depend_on_the_job_count(self, sphere, rng):
        points = sphere.random_points(9, rng=rng)
        serial = sphere.basis_matrix(points)
        assert np.allclose(sphere.basis_matrix(points, n_jobs=2), serial)
        assert np.allclose(sphere.basis_matrix(points[:1], n_jobs=2), serial[:1])

    def test_the_generic_basis_matrix_does_not_depend_on_the_job_count(self, rng):
        box = BoxSobolev((8, 8), 1.0, 0.3)
        points = box.random_points(7, rng=rng)
        serial = box.basis_matrix(points)
        assert np.allclose(box.basis_matrix(points, n_jobs=2), serial)

    def test_the_dense_operators_do_not_depend_on_the_job_count(self, sphere, rng):
        points = sphere.random_points(8, rng=rng)
        paths = list(zip(points[:4], points[4:]))
        builders = [
            lambda n_jobs: sphere.point_evaluation_operator(
                points, route="dense", n_jobs=n_jobs
            ),
            lambda n_jobs: sphere.path_integral_operator(
                paths, count=6, dense=True, n_jobs=n_jobs
            ),
            lambda n_jobs: sphere.path_average_operator(
                paths, count=6, dense=True, n_jobs=n_jobs
            ),
            lambda n_jobs: sphere.geodesic_ball_average_operator(
                points[:3], 0.3, n_jobs=n_jobs
            ),
            lambda n_jobs: sphere.geodesic_ball_average_operator(
                points[:3], 0.3, count=8, dense=True, n_jobs=n_jobs
            ),
        ]
        for build in builders:
            serial = build(None).matrix(form="galerkin")
            assert np.allclose(build(2).matrix(form="galerkin"), serial)


class TestTheContextProbe:
    """``parallel_map`` asks joblib whether a ``parallel_config`` context has
    chosen a backend, through two internals joblib does not export. The
    current spelling is pinned here, and a renamed internal must degrade to
    "no context" rather than break every parallel loop."""

    def test_the_current_spelling_answers(self):
        from joblib import parallel_config

        from pygeoinf2.parallel import _context_is_set

        assert not _context_is_set()
        with parallel_config(backend="threading"):
            assert _context_is_set()
        assert not _context_is_set()

    def test_a_renamed_internal_degrades_to_the_default(self, monkeypatch):
        import joblib.parallel

        from pygeoinf2.parallel import _context_is_set

        monkeypatch.delattr(joblib.parallel, "default_parallel_config")
        assert _context_is_set() is False
        monkeypatch.undo()
        monkeypatch.delattr(joblib.parallel, "_backend")
        assert _context_is_set() is False


class TestTheFamilyAndDiagonalProbesForwardTheirWorkers:
    """The spectral family's shift on its resolved modes, the damped sweep's
    two matrix extractions under a direct solver, and the diagonals of a
    normal operator's error covariance and of a family's shift all go to the
    workers ``n_jobs`` names."""

    @staticmethod
    def recording(monkeypatch):
        import pygeoinf2.parallel as parallel

        seen = []
        real = parallel.parallel_map

        def record(function, items, *args, n_jobs=None, **kwargs):
            seen.append(n_jobs)
            return real(function, items, *args, n_jobs=n_jobs, **kwargs)

        monkeypatch.setattr(parallel, "parallel_map", record)
        return seen

    @staticmethod
    def opaque(space, matrix, traits):
        """Matrix-free, so its matrix and diagonal have to be probed."""
        return LinearOperator.self_adjoint(
            space,
            lambda x: space.solve_gram(matrix @ space.to_components(x)),
            traits=traits,
        )

    def test_the_spectral_family_applies_its_shift_on_the_workers(
        self, rng, monkeypatch
    ):
        space = make_dense_metric_space(8)
        base = spd(space, rng)
        shift = self.opaque(
            space, space.gram_matrix(), Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        serial = SpectralPreconditioner(rank=3, rng=np.random.default_rng(1))
        expected = serial.for_family(base, shift).at(0.5, base + 0.5 * shift)
        seen = self.recording(monkeypatch)
        parallel = SpectralPreconditioner(
            rank=3, rng=np.random.default_rng(1), n_jobs=2
        ).for_family(base, shift)
        assert 2 in seen
        member = base + 0.5 * shift
        vector = space.random(rng=rng)
        assert same(space, parallel.at(0.5, member)(vector), expected(vector))

    def test_the_damped_sweep_extracts_on_the_direct_solvers_workers(
        self, rng, monkeypatch
    ):
        from pygeoinf2.numerics import DampedSolves

        space = make_dense_metric_space(8)
        matrix = spd_operator(space, rng).matrix(form="galerkin")
        base = self.opaque(space, matrix, Traits.POSITIVE_SEMIDEFINITE)
        seen = self.recording(monkeypatch)
        solves = DampedSolves(
            base,
            LinearOperator.identity(space),
            CholeskySolver(n_jobs=2),
            traits=Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE,
        )
        vector = space.random(rng=rng)
        solution = solves.solve(0.5, vector).solution
        assert 2 in seen
        gram = space.gram_matrix()
        expected = np.linalg.solve(
            matrix + 0.5 * gram, gram @ space.to_components(vector)
        )
        assert solution == pytest.approx(expected, rel=1e-10)

    def test_the_normal_diagonal_probes_its_error_on_the_workers(
        self, rng, monkeypatch
    ):
        model, data = EuclideanSpace(6), make_weighted_space()
        forward = LinearOperator.from_matrix(
            model, data, rng.normal(size=(data.dim, model.dim)), form="galerkin"
        )
        chol = CholeskySolver()
        covariance = spd(model, rng)
        prior = GaussianMeasure(
            model, covariance=covariance, precision=chol(covariance)
        )
        noise_matrix = np.diag(data.gram_diagonal() * rng.uniform(0.5, 2.0, data.dim))
        noise = self.opaque(
            data, noise_matrix, Traits.SELF_ADJOINT | Traits.POSITIVE_DEFINITE
        )
        error = GaussianMeasure(data, covariance=noise)
        normal = NormalOperator(forward, prior, error=error, formalism="data_space")
        serial = NormalDiagonalPreconditioner()(normal)
        seen = self.recording(monkeypatch)
        parallel = NormalDiagonalPreconditioner(n_jobs=2)(normal)
        assert seen and all(jobs == 2 for jobs in seen)
        for _ in range(3):
            vector = data.random(rng=rng)
            assert same(data, serial(vector), parallel(vector))


class TestTheMapRunsProcesses:
    def test_there_is_no_switch_to_threads(self):
        """A thread backend is chosen by a ``joblib.parallel_config`` context
        around the call, never by a keyword on the loop."""
        from pygeoinf2.parallel import parallel_map

        assert parallel_map(abs, [-1, -2], n_jobs=None) == [1, 2]
        with pytest.raises(TypeError):
            parallel_map(abs, [-1, -2], n_jobs=2, threads=True)


class TestTheAlgebrasProbes:
    """The adjoint probe and the banded diagonal probe are loops of
    independent applications too."""

    def test_the_probed_adjoint_does_not_depend_on_the_job_count(self, rng):
        X, Y = make_weighted_space(), EuclideanSpace(3)
        matrix = rng.normal(size=(3, X.dim))
        bare = LinearOperator.from_callables(
            X, Y, lambda x: matrix @ X.to_components(x)
        )
        serial = bare.with_probed_adjoint().matrix(by="rows")
        parallel = bare.with_probed_adjoint(n_jobs=2).matrix(by="rows")
        assert np.allclose(parallel, serial)

    def test_the_banded_probe_does_not_depend_on_the_job_count(self, rng):
        X = make_dense_metric_space(8)
        operator = spd_operator(X, rng)
        probed = LinearOperator.from_callables(X, X, operator, adjoint=operator.adjoint)
        serial = probed.diagonals(offsets=(-1, 0, 1), probe="banded")
        parallel = probed.diagonals(offsets=(-1, 0, 1), probe="banded", n_jobs=2)
        assert np.allclose(parallel, serial)
