"""Tests for the sphere DLI example work script (all 4 phases)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from pygeoinf.hilbert_space import EuclideanSpace


WORK_DIR = Path(__file__).resolve().parents[1] / "work"
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))


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


def test_forward_operator_shape():
    from sphere_dli_example import N_RECEIVERS, N_SOURCES, build_forward_operator, build_model_space

    model_space = build_model_space()
    forward_operator, paths = build_forward_operator(model_space)

    assert forward_operator.domain == model_space
    assert forward_operator.codomain == EuclideanSpace(N_SOURCES * N_RECEIVERS)
    assert len(paths) == N_SOURCES * N_RECEIVERS


def test_forward_operator_finite():
    from sphere_dli_example import build_forward_operator, build_model_space

    model_space = build_model_space()
    forward_operator, _ = build_forward_operator(model_space)
    rng = np.random.default_rng(0)
    raw_components = rng.standard_normal(model_space.dim)
    random_model = model_space.from_components(raw_components)

    mapped = forward_operator(random_model)

    assert np.all(np.isfinite(mapped))


def test_forward_operator_constant_field_is_path_average():
    """Applying the forward operator to a constant-value field must return
    the constant on every path, verifying that each row is normalised to a
    true path average rather than a raw arc-length integral."""
    from sphere_dli_example import build_forward_operator, build_model_space

    model_space = build_model_space()
    forward_operator, _ = build_forward_operator(model_space)
    # A constant field f(x) = c; the path average of a constant is c.
    c = 2.5
    constant_field = model_space.project_function(lambda _: c)

    path_averages = np.asarray(forward_operator(constant_field), dtype=float)

    assert_allclose(path_averages, c * np.ones(forward_operator.codomain.dim), rtol=1e-3, atol=1e-3)


def test_synthetic_data_shape():
    from sphere_dli_example import build_forward_operator, build_model_space, generate_synthetic_data

    model_space = build_model_space()
    forward_operator, _ = build_forward_operator(model_space)

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

    result = solve_dli(model_space, fwd_op, prop_op, truth, data, sigma_noise=1e-8)

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