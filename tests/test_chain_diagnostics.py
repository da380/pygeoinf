"""Chain diagnostics against chains whose answers are known.

An AR(1) chain ``x_{n+1} == phi x_n + sqrt(1 - phi^2) e_n`` has
autocorrelation ``phi^k`` and integrated autocorrelation time
``(1 + phi) / (1 - phi)``; independent draws have ``tau == 1``.
"""

import numpy as np
import pytest

from pygeoinf.numerics.chain_diagnostics import (
    autocorrelation,
    effective_sample_size,
    integrated_autocorrelation_time,
    split_rhat,
)


def ar1(phi, steps, chains, rng, *, shift=0.0):
    noise = rng.standard_normal((chains, steps))
    x = np.empty((chains, steps))
    x[:, 0] = noise[:, 0]
    scale = np.sqrt(1.0 - phi**2)
    for n in range(1, steps):
        x[:, n] = phi * x[:, n - 1] + scale * noise[:, n]
    return x + shift


class TestTheAutocorrelation:
    def test_an_ar1_chain_decays_geometrically(self, rng):
        rho = autocorrelation(ar1(0.8, 50000, 1, rng)[0])
        assert rho[0] == 1.0
        assert rho[1:4] == pytest.approx(0.8 ** np.arange(1, 4), abs=0.02)

    def test_it_keeps_the_shape(self, rng):
        assert autocorrelation(rng.standard_normal((3, 100))).shape == (3, 100)

    def test_a_constant_chain_is_one_at_lag_zero_only(self):
        rho = autocorrelation(np.full(10, 2.5))
        assert rho[0] == 1.0 and np.all(rho[1:] == 0.0)


class TestTheAutocorrelationTime:
    @pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
    def test_it_is_that_of_an_ar1_chain(self, phi, rng):
        tau = integrated_autocorrelation_time(ar1(phi, 20000, 4, rng))
        assert tau == pytest.approx((1 + phi) / (1 - phi), rel=0.15)

    def test_the_effective_sample_size_is_the_total_over_tau(self, rng):
        draws = ar1(0.5, 5000, 2, rng)
        assert effective_sample_size(draws) == pytest.approx(
            draws.size / integrated_autocorrelation_time(draws)
        )

    def test_chains_that_disagree_have_a_long_time(self, rng):
        draws = ar1(0.0, 2000, 2, rng)
        draws[1] += 3.0
        assert integrated_autocorrelation_time(draws) > 10.0

    def test_an_antithetic_chain_is_floored(self):
        draws = np.tile([1.0, -1.0], 500)
        assert integrated_autocorrelation_time(draws) == pytest.approx(1 / np.log10(1000))

    def test_a_constant_chain_has_unit_time(self):
        assert integrated_autocorrelation_time(np.ones((2, 10))) == 1.0


class TestRHat:
    def test_agreeing_chains_have_rhat_near_one(self, rng):
        assert split_rhat(ar1(0.5, 4000, 4, rng)) < 1.01

    def test_a_shifted_chain_is_caught(self, rng):
        draws = ar1(0.5, 1000, 4, rng)
        draws[0] += 2.0
        assert split_rhat(draws) > 1.1

    def test_a_chain_still_drifting_is_caught_by_the_split(self, rng):
        trend = np.linspace(0.0, 4.0, 2000)
        assert split_rhat(rng.standard_normal(2000) + trend) > 1.1

    def test_a_chain_of_different_spread_is_caught_by_the_tail(self, rng):
        draws = rng.standard_normal((4, 2000))
        draws[0] *= 3.0
        assert split_rhat(draws) > 1.05


class TestWhatItRefuses:
    @pytest.mark.parametrize("function", [autocorrelation, split_rhat])
    def test_an_array_of_three_dimensions(self, function):
        with pytest.raises(ValueError, match="shape"):
            function(np.zeros((2, 2, 10)))

    def test_a_chain_too_short(self):
        with pytest.raises(ValueError, match="four steps"):
            integrated_autocorrelation_time(np.zeros(3))
