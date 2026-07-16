"""
Regression tests for the Nyquist-mode conventions on the circle and torus.

These pin down three facts that must hold simultaneously:

1. The Lebesgue metric assigns each basis function its true L² norm,
   including the real-only Nyquist modes (squared norm 1/2).
2. Point evaluation (`dirac`) reproduces a function's grid values exactly,
   including functions with Nyquist content.
3. The degree transfer operator embeds a space isometrically into a finer
   one (the Nyquist coefficient is halved as it becomes a conjugate pair).
"""

import numpy as np
import pytest

from pygeoinf.symmetric_space.circle import (
    Lebesgue as CircleLebesgue,
    Sobolev as CircleSobolev,
)
from pygeoinf.symmetric_space.torus import (
    Lebesgue as TorusLebesgue,
    Sobolev as TorusSobolev,
)

KMAX = 8


class TestCircleNyquist:
    def test_nyquist_metric_is_true_norm(self):
        space = CircleLebesgue(KMAX, radius=1.3)
        comp = np.zeros(space.dim)
        comp[KMAX] = 1.0
        u = space.from_components(comp)
        # basis function is cos(kmax * theta) / sqrt(2 pi r): squared norm 1/2
        assert space.inner_product(u, u) == pytest.approx(0.5, rel=1e-12)

    def test_dirac_reproduces_grid_values(self):
        space = CircleSobolev(KMAX, 2.0, 0.1)
        rng = np.random.default_rng(42)
        u = space.from_components(rng.standard_normal(space.dim))
        values = np.asarray(u)
        thetas = np.arange(2 * KMAX) * np.pi / KMAX
        op = space.point_evaluation_operator(list(thetas))
        np.testing.assert_allclose(op(u), values, rtol=1e-10, atol=1e-12)

    def test_degree_transfer_is_isometric_embedding(self):
        space = CircleLebesgue(KMAX)
        fine = space.degree_transfer_operator(2 * KMAX)
        rng = np.random.default_rng(43)
        u = space.from_components(rng.standard_normal(space.dim))
        v = fine(u)
        assert fine.codomain.inner_product(v, v) == pytest.approx(
            space.inner_product(u, u), rel=1e-12
        )
        # round trip: project back onto the coarse space
        u_back = fine.adjoint(v)
        np.testing.assert_allclose(
            space.to_components(u_back), space.to_components(u), rtol=1e-10
        )


class TestTorusNyquist:
    def test_corner_metric_is_true_norm(self):
        space = TorusLebesgue(KMAX)
        corners = [(KMAX, 0), (0, KMAX), (KMAX, KMAX)]
        for kx, ky in corners:
            (idx,) = [
                i
                for i in range(space.dim)
                if space._kx_freqs[i] == kx
                and space._ky_freqs[i] == ky
                and not space._is_imag[i]
            ]
            assert space.metric_values[idx] == pytest.approx(0.5)
        # constant mode keeps unit norm
        assert space.metric_values[0] == pytest.approx(1.0)

    def test_dirac_reproduces_grid_values(self):
        space = TorusSobolev(KMAX, 2.0, 0.1)
        rng = np.random.default_rng(44)
        u = space.from_components(rng.standard_normal(space.dim))
        values = np.asarray(u)
        h = np.pi / KMAX
        points = [(2 * h, 3 * h), (0.0, 0.0), (5 * h, 7 * h)]
        expected = np.array([values[2, 3], values[0, 0], values[5, 7]])
        op = space.point_evaluation_operator(points)
        np.testing.assert_allclose(op(u), expected, rtol=1e-10, atol=1e-12)
