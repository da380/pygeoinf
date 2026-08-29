"""
Scattered evaluation on a sphere, through the double Fourier sphere.

Two routes to the same numbers: sum the basis at each point, or extend the
field to the torus and use a non-uniform FFT. The first is obviously right and
unusably slow; the second is neither. So every test here forces both and
compares, and the ones about the delicate parts — the quadrature weights and
the pole the grid does not sample — carry a negative control.

See DESIGN.md section 21.15.
"""

import numpy as np
import pytest

from pygeoinf2.symmetric_space.base import SymmetricSpace
from pygeoinf2.testing import check_operator

pyshtools = pytest.importorskip("pyshtools")
finufft = pytest.importorskip("finufft")

import pygeoinf2.symmetric_space.sphere as sphere_module  # noqa: E402
from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev  # noqa: E402


class forced:
    """Force one route or the other, whatever the size."""

    def __init__(self, transform: bool) -> None:
        self.transform = transform

    def __enter__(self):
        self.points = sphere_module._TRANSFORM_MIN_POINTS
        self.dimension = sphere_module._TRANSFORM_MIN_DIM
        value = 0 if self.transform else 10**18
        sphere_module._TRANSFORM_MIN_POINTS = value
        sphere_module._TRANSFORM_MIN_DIM = value
        return self

    def __exit__(self, *args):
        sphere_module._TRANSFORM_MIN_POINTS = self.points
        sphere_module._TRANSFORM_MIN_DIM = self.dimension


class TestQuadratureWeights:
    def test_the_transform_really_is_a_quadrature(self):
        """The fact the whole adjoint rests on.

        ``to_components(e_jk)`` must be a row-dependent multiple of
        ``basis_at(p_jk)`` — the same multiple for every component. If it were
        not, the weights would not exist and there would be no adjoint.
        """
        X = Lebesgue(6)
        rows, columns = X.grid_shape
        for row in (1, 3, rows // 2, rows - 1):
            indicator = np.zeros(X.grid_shape)
            indicator[row, 2] = 1.0
            got = X.to_components(indicator)
            want = X.basis_at(
                X.to_latitude_degrees(np.array([X.colatitudes[row], X.longitudes[2]]))[
                    0
                ]
            )
            live = np.abs(want) > 1e-6 * np.abs(want).max()
            ratios = got[live] / want[live]
            assert np.allclose(ratios, ratios[0], rtol=1e-8)

    def test_the_pole_row_carries_no_weight(self):
        """Which is why it has to be added back by hand.

        The grid samples colatitude on ``[0, pi)`` and the quadrature gives the
        pole no area, so anything sitting there is invisible to the transform.
        """
        X = Lebesgue(6)
        assert X._quadrature[0] == 0.0
        assert np.all(X._quadrature[1:] > 0.0)

        pole_only = np.zeros(X.grid_shape)
        pole_only[0] = 1.0
        assert np.allclose(X.to_components(pole_only), 0.0)

    def test_synthesis_adjoint_matches_an_explicit_sum(self, rng):
        X = Lebesgue(6)
        rows, columns = X.grid_shape
        values = rng.normal(size=X.grid_shape)
        expected = np.zeros(X.dim)
        for row in range(rows):
            for column in range(columns):
                expected += values[row, column] * X.basis_at(
                    X.to_latitude_degrees(
                        np.array([X.colatitudes[row], X.longitudes[column]])
                    )[0]
                )
        assert np.allclose(X._synthesis_adjoint(values), expected)

    def test_dropping_the_pole_would_be_wrong(self, rng):
        """The negative control for the pole term."""
        X = Lebesgue(6)
        values = rng.normal(size=X.grid_shape)
        weights = X._quadrature
        live = weights > 0.0
        scaled = np.zeros(X.grid_shape)
        scaled[live] = values[live] / weights[live, None]
        without_pole = X.to_components(scaled)
        assert not np.allclose(without_pole, X._synthesis_adjoint(values))


class TestDoubling:
    def test_the_extension_agrees_with_the_field_it_extends(self, rng):
        X = Lebesgue(8)
        rows, columns = X.grid_shape
        field = X.random(rng=rng)
        values = X.grid_values(field)
        doubled = X._double(field)
        assert np.allclose(doubled[:rows], values)
        # beyond the equator: g(2 pi - theta, phi) == f(theta, phi + pi)
        assert np.allclose(
            doubled[rows + 1 :], np.roll(values[1:][::-1], columns // 2, axis=1)
        )

    def test_the_middle_row_is_the_south_pole(self, rng):
        X = Lebesgue(8)
        rows, _ = X.grid_shape
        field = X.random(rng=rng)
        pole = X.evaluate(field, [np.array([-90.0, 0.0])])[0]
        assert np.allclose(X._double(field)[rows], pole)

    def test_the_extension_is_a_trigonometric_polynomial(self, rng):
        """Which is what makes one FFT of it exact rather than approximate.

        Its Fourier coefficients must vanish above the truncation in both
        variables; if the extension were merely continuous they would not.
        """
        X = Lebesgue(8)
        coefficients = np.fft.fft2(X._double(X.random(rng=rng)))
        rows, columns = coefficients.shape
        frequencies = np.fft.fftfreq(rows, d=1.0 / rows)
        beyond = np.abs(frequencies) > X.lmax + 1
        assert np.abs(coefficients[beyond]).max() < 1e-9 * np.abs(coefficients).max()

    @pytest.mark.parametrize("lmax", [4, 8, 17, 32])
    @pytest.mark.parametrize("radius,sampling", [(1.0, 1), (1.7, 1), (1.0, 2)])
    def test_the_south_pole_comes_from_the_row_means(self, lmax, radius, sampling, rng):
        """REVIEW2 4.2.8. The pole value used to cost a full analysis -- 35% of
        a forward evaluation at lmax 256 -- to read one number out. Only the
        zonal harmonics are non-zero at a pole and they do not depend on
        longitude, so it is a weighting of the row means."""
        X = Sobolev(lmax, 2.0, 0.2, radius=radius, sampling=sampling)
        rows, _ = X.grid_shape
        field = X.random(rng=rng)
        analysis = float(X._south_pole_basis @ X.to_components(field))
        assert X._double(field)[rows][0] == pytest.approx(analysis, rel=1e-11)

    def test_the_pole_value_costs_no_transform(self, rng, monkeypatch):
        """The point of the closed form: nothing in the extension analyses."""
        X = Lebesgue(12)
        field = X.random(rng=rng)
        X._double(field)  # warm the cached kernel, which does probe once

        def refuse(*args, **kwargs):
            raise AssertionError("_double analysed the field")

        monkeypatch.setattr(type(X), "to_components", refuse)
        assert X._double(field) is not None

    def test_doubling_and_its_adjoint_are_adjoint(self, rng):
        X = Lebesgue(6)
        rows, columns = X.grid_shape
        field = X.random(rng=rng)
        other = rng.normal(size=(2 * rows, columns))
        forward = float(np.sum(X._double(field) * other))
        backward = float(np.dot(X._double_adjoint(other), X.to_components(field)))
        assert forward == pytest.approx(backward)


class TestBothRoutesAgree:
    @pytest.mark.parametrize("lmax", [6, 16, 32])
    @pytest.mark.parametrize("radius", [1.0, 1.7])
    def test_evaluation(self, lmax, radius, rng):
        X = Sobolev(lmax, 2.0, 0.2, radius=radius)
        field = X.random(rng=rng)
        points = X.random_points(60, rng=rng)
        reference = SymmetricSpace.evaluate(X, field, points)
        with forced(transform=False):
            assert np.allclose(X.evaluate(field, points), reference)
        with forced(transform=True):
            assert np.allclose(X.evaluate(field, points), reference, atol=1e-9)

    @pytest.mark.parametrize("lmax", [6, 16, 32])
    def test_accumulation(self, lmax, rng):
        X = Sobolev(lmax, 2.0, 0.2)
        points = X.random_points(60, rng=rng)
        weights = rng.normal(size=60)
        reference = SymmetricSpace.accumulate(X, weights, points)
        scale = np.abs(reference).max()
        with forced(transform=False):
            assert np.allclose(X.accumulate(weights, points), reference)
        with forced(transform=True):
            assert np.allclose(
                X.accumulate(weights, points), reference, atol=1e-8 * scale
            )

    def test_the_operator_works_on_either_route(self, rng):
        X = Sobolev(16, 2.0, 0.2)
        points = X.random_points(40, rng=rng)
        for transform in (False, True):
            with forced(transform=transform):
                check_operator(X.point_evaluation_operator(points), rng=rng)

    def test_a_path_average_agrees_on_either_route(self, rng):
        X = Sobolev(16, 2.0, 0.2)
        paths = [
            (a, b)
            for a, b in zip(X.random_points(8, rng=rng), X.random_points(8, rng=rng))
        ]
        field = X.random(rng=rng)
        with forced(transform=False):
            direct = X.path_average_operator(paths, count=10)(field)
        with forced(transform=True):
            through = X.path_average_operator(paths, count=10)(field)
        assert np.allclose(direct, through, atol=1e-9)

    def test_the_route_is_chosen_by_size(self):
        small, large = Lebesgue(4), Lebesgue(32)
        assert not small._use_transform(10_000)  # too few components
        assert not large._use_transform(10)  # too few points
        assert large._use_transform(10_000)

    def test_a_mismatched_weight_count_is_refused(self, rng):
        X = Lebesgue(8)
        with pytest.raises(ValueError, match="weights for"):
            X.accumulate(np.ones(3), X.random_points(4, rng=rng))


class TestWalkingOverThePole:
    """REVIEW2 3.4. A walk longer than a quarter circumference used to run the
    colatitude past ``pi`` and return a latitude below -90, which the two
    evaluation routes then read as two different points."""

    def test_it_stays_on_the_sphere(self):
        X = Lebesgue(8)
        distances = np.linspace(0.0, 2.0 * np.pi, 41)
        points = np.stack(X.walk_from([20.0, 40.0], distances))
        assert np.all(np.abs(points[:, 0]) <= 90.0)

    def test_it_reflects_through_the_pole(self):
        """Half a circumference from a point is its antipode, and the meridian
        continues down the far side rather than off the end of the sphere."""
        X = Lebesgue(8)
        (antipode,) = X.walk_from([20.0, 40.0], np.array([np.pi]))
        assert np.allclose(antipode, [-20.0, -140.0])
        (past,) = X.walk_from([20.0, 40.0], np.array([1.5 * np.pi]))
        assert np.allclose(past, [70.0, -140.0])

    def test_the_distance_walked_is_the_distance_asked_for(self):
        X = Lebesgue(8)
        start = np.array([20.0, 40.0])
        distances = np.linspace(0.0, 2.0 * np.pi, 41)
        expected = np.minimum(distances, 2.0 * np.pi - distances)
        walked = [X.geodesic_distance(start, p) for p in X.walk_from(start, distances)]
        assert np.allclose(walked, expected)

    def test_both_evaluation_routes_agree_past_the_pole(self, rng):
        """The symptom. On a non-zonal field the two routes differed by 1.02
        on a field of maximum 0.47, because the direct sum read the bad
        colatitude as ``(2 pi - theta, phi)`` and the doubled grid read it as
        ``(2 pi - theta, phi + pi)``."""
        X = Sobolev(16, 2.0, 0.2)
        field = X.random(rng=rng)
        points = X.walk_from([20.0, 40.0], np.linspace(0.0, 2.0 * np.pi, 400))
        reference = SymmetricSpace.evaluate(X, field, points)
        scale = np.abs(reference).max()
        with forced(transform=True):
            assert np.allclose(X.evaluate(field, points), reference, atol=1e-8 * scale)

    def test_a_latitude_out_of_range_is_refused(self):
        """The check the docstring promised and the code never made, and the
        one that would have caught the bug above."""
        X = Lebesgue(4)
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            X.to_colatitude_radians([[-126.0, 0.0]])
        with pytest.raises(ValueError, match=r"\[-90, 90\]"):
            X.to_colatitude_radians([[10.0, 0.0], [91.0, 0.0]])
        # A pole that arrived from an arcsine is not an error.
        assert X.to_colatitude_radians([[90.0 + 1e-13, 0.0]])[0, 0] == 0.0


class TestQuadratureFromDriscollHealy:
    """The weights come from pyshtools' closed form, not from probing the
    transform. This is the check that the two are the same thing."""

    @pytest.mark.parametrize("lmax", [4, 8, 16])
    @pytest.mark.parametrize("sampling", [1, 2])
    def test_the_closed_form_matches_the_transform(self, lmax, sampling):
        """Up to the common constant, which is all either one is defined to."""
        space = Lebesgue(lmax, sampling=sampling)
        measured = space._quadrature_from_transform()
        live = measured != 0.0

        ratio = space._quadrature[live] / measured[live]
        assert np.ptp(ratio) / ratio.mean() < 1e-12

    def test_the_north_pole_row_carries_no_weight(self):
        """The grid samples colatitude on [0, pi), so the quadrature gives the
        pole no area -- and anything sitting there is invisible to the
        transform. Both routes have to agree about which row that is."""
        space = Lebesgue(8)
        assert space._quadrature[0] == 0.0
        assert np.flatnonzero(space._quadrature == 0.0).tolist() == [0]
        assert np.flatnonzero(space._quadrature_from_transform() == 0.0).tolist() == [0]
