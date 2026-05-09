import numpy as np
import pytest

from pygeoinf.convex_optimisation import SubgradientDescent
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.nonlinear_forms import NonLinearForm


def _quadratic_oracle():
    """Return a 1-D quadratic oracle with minimiser at x=1."""
    domain = EuclideanSpace(1)

    def f(x):
        return 0.5 * float((x[0] - 1.0) ** 2)

    def g(x):
        return np.array([x[0] - 1.0])

    return NonLinearForm(domain, f, subgradient=g), domain


def _absolute_value_oracle():
    """Return a 1-D absolute-value oracle with minimiser at x=0.5."""
    domain = EuclideanSpace(1)

    def f(x):
        return float(abs(x[0] - 0.5))

    def g(x):
        value = x[0] - 0.5
        return np.array([1.0 if value > 0 else (-1.0 if value < 0 else 0.0)])

    return NonLinearForm(domain, f, subgradient=g), domain


def test_subgradient_constant_step_improves_quadratic():
    """A stable constant step should move toward the quadratic minimiser."""
    oracle, domain = _quadratic_oracle()
    x0 = domain.from_components(np.array([3.0]))

    solver = SubgradientDescent(oracle, step_size=0.2, max_iterations=25)
    result = solver.solve(x0)

    np.testing.assert_allclose(
        domain.to_components(result.x_best),
        np.array([1.0]),
        atol=1e-2,
    )
    assert result.f_best < oracle(x0)
    assert result.num_iterations == 25
    assert not result.converged


def test_subgradient_store_iterates_tracks_history():
    """When enabled, iterate history should align with the function-value history."""
    oracle, domain = _quadratic_oracle()
    x0 = domain.from_components(np.array([2.0]))

    solver = SubgradientDescent(
        oracle,
        step_size=0.1,
        max_iterations=8,
        store_iterates=True,
    )
    result = solver.solve(x0)

    assert result.iterates is not None
    assert len(result.iterates) == result.num_iterations
    assert len(result.function_values) == result.num_iterations
    np.testing.assert_allclose(
        domain.to_components(result.iterates[0]),
        domain.to_components(x0),
    )


def test_subgradient_stagnation_window_detects_plateau():
    """Repeated non-improvement at the minimiser should trigger convergence."""
    oracle, domain = _absolute_value_oracle()
    x0 = domain.from_components(np.array([0.5]))

    solver = SubgradientDescent(
        oracle,
        step_size=0.5,
        max_iterations=20,
        stagnation_window=3,
    )
    result = solver.solve(x0)

    assert result.converged
    assert result.num_iterations == 4
    np.testing.assert_allclose(
        domain.to_components(result.x_final),
        domain.to_components(x0),
    )
    np.testing.assert_allclose(result.f_best, 0.0, atol=1e-12)


def test_subgradient_requires_positive_step_size():
    """Constructor should reject non-positive step sizes."""
    oracle, _ = _quadratic_oracle()

    with pytest.raises(ValueError, match="step_size must be positive"):
        SubgradientDescent(oracle, step_size=0.0)
