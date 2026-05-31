"""Tests for the sphere DLI example work script (all 4 phases)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace


SPHERE_DLI_DIR = Path(__file__).resolve().parents[2] / "sphere_dli_paper"
if str(SPHERE_DLI_DIR) not in sys.path:
    sys.path.insert(0, str(SPHERE_DLI_DIR))


def test_gaussian_noise_ball_radius_matches_chi_square_quantile():
    from sphere_dli_example import gaussian_noise_ball_radius

    radius = gaussian_noise_ball_radius(0.001, 100, 0.9973002039367398)

    assert_allclose(radius, 0.011993572212187787, rtol=0.0, atol=1e-15)


def test_gaussian_noise_ball_radius_validates_arguments():
    from sphere_dli_example import gaussian_noise_ball_radius

    with pytest.raises(ValueError, match="sigma_noise"):
        gaussian_noise_ball_radius(-1.0, 100, 0.99)
    with pytest.raises(ValueError, match="data_dim"):
        gaussian_noise_ball_radius(1.0, 0, 0.99)
    with pytest.raises(ValueError, match="confidence"):
        gaussian_noise_ball_radius(1.0, 100, 1.0)


def test_generate_synthetic_data_does_not_clip_gaussian_noise(monkeypatch):
    from sphere_dli_example import generate_synthetic_data

    class FakeRng:
        def integers(self, *_args, **_kwargs):
            return 1

        def normal(self, loc, scale, size):
            return np.array([-4.0, 0.0, 3.5]) * scale + loc

    class FakePrior:
        def sample(self):
            return object()

    class FakeModelSpace:
        def point_value_scaled_heat_kernel_gaussian_measure(self, _scale):
            return FakePrior()

    class FakeForwardOperator:
        def __call__(self, _model):
            return np.zeros(3)

    monkeypatch.setattr(np.random, "default_rng", lambda _seed: FakeRng())

    _truth, data = generate_synthetic_data(
        FakeModelSpace(),
        FakeForwardOperator(),
        sigma_noise=0.02,
        seed=7,
    )

    assert np.max(np.abs(data)) > 3.0 * 0.02


def test_model_space_builds():
    from sphere_dli_example import ORDER, build_model_space

    model_space = build_model_space()

    assert model_space.dim > 0
    assert model_space.order == ORDER == 1.5


def test_cap_property_operator_shape():
    from sphere_dli_example import DEFAULT_TARGET_LATLON, build_cap_property_operator, build_model_space

    model_space = build_model_space()
    operator = build_cap_property_operator(model_space, DEFAULT_TARGET_LATLON)

    assert operator.domain == model_space
    assert operator.codomain == EuclideanSpace(len(DEFAULT_TARGET_LATLON))
    assert operator.matrix().shape == (len(DEFAULT_TARGET_LATLON), model_space.dim)


def test_cap_property_operator_constant_field():
    from sphere_dli_example import DEFAULT_TARGET_LATLON, build_cap_property_operator, build_model_space

    model_space = build_model_space()
    operator = build_cap_property_operator(model_space, DEFAULT_TARGET_LATLON)
    constant_field = model_space.project_function(lambda _: 1.0)

    cap_values = operator(constant_field)

    assert_allclose(cap_values, np.ones(len(DEFAULT_TARGET_LATLON)), rtol=1e-3, atol=1e-3)


def test_cap_property_operator_exact_default_is_seed_deterministic():
    from sphere_dli_example import build_cap_property_operator, build_model_space

    model_space = build_model_space(min_degree=16)
    target_latlon = [(90.0, 0.0), (20.0, 40.0)]

    operator_a = build_cap_property_operator(
        model_space,
        target_latlon,
        n_cap=5,
        seed=1,
    )
    operator_b = build_cap_property_operator(
        model_space,
        target_latlon,
        n_cap=5,
        seed=99,
    )
    test_field = model_space.from_components(
        np.linspace(-0.5, 0.5, model_space.dim, dtype=float)
    )

    assert_allclose(operator_a(test_field), operator_b(test_field), rtol=0.0, atol=1e-12)


def test_cap_property_operator_exact_mode_is_n_cap_independent():
    from sphere_dli_example import build_cap_property_operator, build_model_space

    model_space = build_model_space(min_degree=16)
    target_latlon = [(90.0, 0.0), (20.0, 40.0)]

    operator_a = build_cap_property_operator(
        model_space,
        target_latlon,
        n_cap=5,
        seed=1,
        method="exact",
    )
    operator_b = build_cap_property_operator(
        model_space,
        target_latlon,
        n_cap=30,
        seed=99,
        method="exact",
    )
    test_field = model_space.from_components(
        np.linspace(-0.5, 0.5, model_space.dim, dtype=float)
    )

    assert_allclose(operator_a(test_field), operator_b(test_field), rtol=0.0, atol=1e-12)


def test_forward_operator_shape():
    from sphere_dli_example import build_forward_operator, build_model_space

    model_space = build_model_space()
    n_sources = 2
    n_receivers = 3
    forward_operator, paths = build_forward_operator(
        model_space,
        n_sources=n_sources,
        n_receivers=n_receivers,
        seed=0,
    )

    assert forward_operator.domain == model_space
    assert forward_operator.codomain == EuclideanSpace(n_sources * n_receivers)
    assert len(paths) == n_sources * n_receivers


def test_forward_operator_finite():
    from sphere_dli_example import build_forward_operator, build_model_space

    model_space = build_model_space()
    forward_operator, _ = build_forward_operator(
        model_space,
        n_sources=2,
        n_receivers=3,
        seed=0,
    )
    rng = np.random.default_rng(0)
    raw_components = rng.standard_normal(model_space.dim)
    random_model = model_space.from_components(raw_components)

    mapped = forward_operator(random_model)

    assert np.all(np.isfinite(mapped))


def test_forward_operator_constant_field_is_reference_weighted_path_average():
    """A constant perturbation field should map to the ray-average of ``1 / c_0``.

    This verifies that the forward operator now implements the remaining
    inverse-reference-velocity factor required by the linearized physics.
    """
    from sphere_dli_example import OMEGA, build_forward_operator, build_model_space, reference_phase_velocity

    model_space = build_model_space()
    forward_operator, paths = build_forward_operator(model_space, n_sources=2, n_receivers=3, seed=0)
    c = 2.5
    constant_field = model_space.project_function(lambda _: c)

    weighted_path_averages = np.asarray(forward_operator(constant_field), dtype=float)

    expected = []
    for point_1, point_2 in paths:
        _, trial_weights = model_space.geodesic_quadrature(point_1, point_2, n_points=2)
        trial_arc_length = float(np.sum(trial_weights))
        n_points = max(2, int(np.ceil((trial_arc_length / model_space.scale) * 2.0)))
        points, weights = model_space.geodesic_quadrature(point_1, point_2, n_points=n_points)
        arc_length = float(np.sum(weights))
        expected.append(
            -OMEGA
            * c
            * np.sum(
                [weight / reference_phase_velocity(point) for point, weight in zip(points, weights)]
            )
            / arc_length
        )
    expected = np.asarray(expected, dtype=float)

    assert_allclose(weighted_path_averages, expected, rtol=1e-3, atol=1e-3)
    assert np.ptp(weighted_path_averages) > 1e-3


def test_synthetic_data_shape():
    from sphere_dli_example import build_forward_operator, build_model_space, generate_synthetic_data

    model_space = build_model_space()
    forward_operator, _ = build_forward_operator(
        model_space,
        n_sources=2,
        n_receivers=3,
        seed=0,
    )

    truth_model, data_vector = generate_synthetic_data(
        model_space,
        forward_operator,
        sigma_noise=0.02,
        seed=7,
    )

    assert truth_model is not None
    assert isinstance(data_vector, np.ndarray)
    assert data_vector.shape == (forward_operator.codomain.dim,)


def test_dli_bounds_finite():
    """Bounds must be finite on a tiny problem."""
    from sphere_dli_example import (
        build_cap_property_operator,
        build_forward_operator,
        build_model_space,
        generate_synthetic_data,
        solve_dli,
    )

    tiny_target = [(-60.0, 0.0), (20.0, 25.0)]
    model_space = build_model_space(min_degree=16)
    prop_op = build_cap_property_operator(model_space, tiny_target, n_cap=5, seed=0)
    fwd_op, _ = build_forward_operator(model_space, n_sources=2, n_receivers=3, seed=0)
    truth, data = generate_synthetic_data(model_space, fwd_op, seed=1)

    result = solve_dli(model_space, fwd_op, prop_op, truth, data)

    assert np.all(np.isfinite(result["lower"]))
    assert np.all(np.isfinite(result["upper"]))


def test_dli_bounds_ordered():
    """Lower bound must be ≤ upper bound for every property."""
    from sphere_dli_example import (
        build_cap_property_operator,
        build_forward_operator,
        build_model_space,
        generate_synthetic_data,
        solve_dli,
    )

    tiny_target = [(-60.0, 0.0), (20.0, 25.0)]
    model_space = build_model_space(min_degree=16)
    prop_op = build_cap_property_operator(model_space, tiny_target, n_cap=5, seed=0)
    fwd_op, _ = build_forward_operator(model_space, n_sources=2, n_receivers=3, seed=0)
    truth, data = generate_synthetic_data(model_space, fwd_op, seed=1)

    result = solve_dli(model_space, fwd_op, prop_op, truth, data)

    assert np.all(result["lower"] <= result["upper"] + 1e-6)


def test_truth_inside_bounds():
    """
    With zero noise and a data-confidence ball of near-zero radius, the true property
    value should lie within the admissible interval to a loose tolerance.
    """
    from sphere_dli_example import (
        build_cap_property_operator,
        build_forward_operator,
        build_model_space,
        generate_synthetic_data,
        solve_dli,
    )

    tiny_target = [(-60.0, 0.0), (20.0, 25.0)]
    model_space = build_model_space(min_degree=8)
    prop_op = build_cap_property_operator(model_space, tiny_target, n_cap=5, seed=0)
    fwd_op, _ = build_forward_operator(model_space, n_sources=2, n_receivers=3, seed=0)
    truth, _ = generate_synthetic_data(model_space, fwd_op, seed=1)
    data = np.asarray(fwd_op(truth), dtype=float)

    result = solve_dli(
        model_space,
        fwd_op,
        prop_op,
        truth,
        data,
        sigma_noise=1e-8,
        n_jobs=1,
    )

    true_vals = result["true_values"]
    tol = 0.05
    assert np.all(result["lower"] <= true_vals + tol)
    assert np.all(result["upper"] >= true_vals - tol)


def test_end_to_end_tiny():
    """Run the whole pipeline on a minimal configuration and check output structure."""
    from sphere_dli_example import run_example

    result = run_example(min_degree=8, n_sources=2, n_receivers=3, n_target=2, n_cap=5)

    assert "lower" in result and "upper" in result and "true_values" in result
    assert np.all(np.isfinite(result["lower"]))
    assert np.all(np.isfinite(result["upper"]))
    assert len(result["lower"]) == 2


def test_script_importable():
    """Import the script without executing any heavy computation."""
    import sphere_dli_example

    assert hasattr(sphere_dli_example, "run_example")
    assert hasattr(sphere_dli_example, "solve_dli")
    assert hasattr(sphere_dli_example, "plot_results")