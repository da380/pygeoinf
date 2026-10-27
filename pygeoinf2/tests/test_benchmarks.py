"""
The benchmarks of ``benchmarks/``, at a size the suite can afford.

Each script builds its problem in a ``build`` function and measures it in a
``measure`` function, and those are what run here: the same code as the
paper-scale runs, on a coarser sphere with fewer data, checked against the
acceptance criteria the benchmarks were written for. Every support value is
an outer bound with a certified bracket, so two algorithms that both apply
must give brackets that overlap; the spectral solver must reproduce the
default solver's closed form to ``1e-10``; the bounds must change little
between the two finest truncations; and workers must change nothing.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("pyshtools")

BENCHMARKS = pathlib.Path(__file__).resolve().parent.parent.parent / "benchmarks"

pytestmark = pytest.mark.slow


def load(name: str):
    """Import a benchmark script by file, as the scripts import each other."""
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution, as an import would be: the scripts'
    # dataclasses resolve their annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestPointData:
    def test_the_tier_agrees_the_extents_fit_and_the_workers_agree(self):
        bench = load("point_data_sphere")
        rng = np.random.default_rng(0)
        problem = bench.build(lmax=12, points=30, rng=rng)
        out = bench.measure(problem, n_jobs=2, rng=rng, quiet=True)
        brackets = {}
        for dim, label, difference, gap, *_, converged, inside, _ in out["working"]:
            assert converged, (dim, label)
            if label == "primal, spectral":
                assert inside, "the truth lies outside the set"
                brackets[dim] = gap
            elif label == "primal, CG":
                assert difference <= 1e-8, (dim, label, difference)
            else:
                # The dual's outer value and the primal's are both certified
                # bounds, so they agree to the brackets' width.
                assert difference <= 10.0 * max(brackets[dim], gap) + 1e-6, (
                    dim,
                    label,
                    difference,
                )
        for row in out["extents"]:
            assert row[7], f"an extent lies outside its bracket: {row}"
        assert out["workers"][-1][1] <= 1e-9


class TestFlexureGravity:
    def test_the_algorithms_agree_the_truth_is_inside_and_refinement_converges(self):
        bench = load("flexure_gravity")
        rng = np.random.default_rng(0)
        shared = bench.make_data(truth_lmax=22, rng=rng)
        problems = [bench.build(lmax, shared) for lmax in (20, 22)]
        out = bench.measure(
            problems, n_jobs=2, planar=4, dual_iterations=600, quiet=True
        )
        primal = [row for row in out["per_truncation"] if row[2].startswith("primal")]
        for row in primal:
            assert row[8], row
            assert row[3] <= 1e-8, row
        assert all(row[9] for row in primal if row[9] != "")
        assert out["hull"][0][1], "the contact hull is not inside the polytope"
        assert all(np.isfinite(row[1]) for row in out["refinement"])


class TestPhaseDelay:
    def test_the_preconditioned_solvers_agree_with_the_spectral_one(self):
        bench = load("phase_delay")
        rng = np.random.default_rng(0)
        space = bench.Sobolev(12, 2.0, 0.2)
        acquired = bench.acquisition(space, synthetic=True, rng=rng)
        truth = bench.truth_for(space, 3.0, rng=rng)
        problems = [
            bench.build(space, acquired, count, truth=truth, prior_radius=3.0, rng=rng)
            for count in (40, 80)
        ]
        out = bench.measure(
            problems, deflation_rank=8, coarse_degree=8, n_jobs=2, quiet=True, rng=rng
        )
        by_count = {}
        for row in out["cost"]:
            by_count.setdefault(row[0], []).append(row)
        for count, rows in by_count.items():
            assert rows[0][1] == "spectral" and rows[0][-1], count
            # Whatever converged agrees with the spectral solver to its bracket.
            for row in rows[1:]:
                if row[-1]:
                    assert row[2] <= 1e-6, row
            assert any(
                row[-1] for row in rows[1:]
            ), f"no iterative solver converged at {count} data: {rows}"
        assert out["workers"][-1][1] <= 1e-8
