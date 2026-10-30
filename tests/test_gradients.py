"""
The gradient of a field at scattered points, on every geometry.

Checked against central differences of the field's own point evaluation,
which is the only thing it has to agree with.
"""

import numpy as np
import pytest

from pygeoinf.symmetric_space.circle import Sobolev as CircleSobolev
from pygeoinf.symmetric_space.line import Sobolev as LineSobolev
from pygeoinf.symmetric_space.plane import Sobolev as PlaneSobolev
from pygeoinf.symmetric_space.torus import Sobolev as TorusSobolev


def differences(space, field, points, step, move):
    """Central differences of the field along the directions ``move`` gives."""
    columns = []
    for axis in range(space.spatial_dimension):
        forward = space.evaluate(field, [move(p, axis, step) for p in points])
        backward = space.evaluate(field, [move(p, axis, -step) for p in points])
        columns.append((forward - backward) / (2.0 * step))
    return np.column_stack(columns)


class TestOnTheBoxes:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: TorusSobolev((32, 24), 2.0, 0.2, lengths=(2.0, 1.5)),
            lambda: PlaneSobolev((32, 24), 2.0, 0.2, bounds=[(0.0, 2.0), (1.0, 2.5)]),
            lambda: CircleSobolev(48, 2.0, 0.2),
            lambda: LineSobolev(48, 2.0, 0.2, lower=0.0, upper=3.0),
        ],
        ids=["torus", "plane", "circle", "line"],
    )
    def test_it_matches_central_differences(self, make, rng):
        space = make()
        field = space.random(rng=rng)
        points = [space.random_point(rng=rng) for _ in range(12)]
        # Keep clear of a bounded box's edges, where the padding takes over.
        if hasattr(space, "bounds"):
            points = [
                np.array(
                    [
                        0.2 * (b - a) + 0.6 * (b - a) * rng.uniform() + a
                        for a, b in space.bounds
                    ]
                )
                for _ in range(12)
            ]
            if space.spatial_dimension == 1:
                points = [p[0] for p in points]

        def move(point, axis, step):
            moved = np.atleast_1d(np.asarray(point, dtype=float)).copy()
            moved[axis] += step
            return moved if moved.size > 1 else moved[0]

        gradient = space.evaluate_gradient(field, points)
        assert gradient.shape == (12, space.spatial_dimension)
        expected = differences(space, field, points, 1.0e-5, move)
        scale = np.max(np.abs(expected))
        assert np.allclose(gradient, expected, atol=1.0e-6 * scale)


class TestOnTheSphere:
    def test_it_matches_central_differences_in_both_angles(self, rng):
        pytest.importorskip("pyshtools")
        pytest.importorskip("ducc0")
        from pygeoinf.symmetric_space.sphere import Sobolev

        space = Sobolev(24, 2.0, 0.3, radius=2.0)
        field = space.random(rng=rng)
        points = [
            np.array([rng.uniform(-70.0, 70.0), rng.uniform(0.0, 360.0)])
            for _ in range(12)
        ]
        gradient = space.evaluate_gradient(field, points)
        assert gradient.shape == (12, 2)
        step = 1.0e-4  # degrees
        for index, point in enumerate(points):
            lat, lon = point
            south = (
                space.evaluate(field, [np.array([lat - step, lon])])[0]
                - space.evaluate(field, [np.array([lat + step, lon])])[0]
            ) / (2.0 * np.radians(step) * space.radius)
            east = (
                space.evaluate(field, [np.array([lat, lon + step])])[0]
                - space.evaluate(field, [np.array([lat, lon - step])])[0]
            ) / (2.0 * np.radians(step) * space.radius * np.cos(np.radians(lat)))
            assert gradient[index, 0] == pytest.approx(south, rel=1e-6, abs=1e-8)
            assert gradient[index, 1] == pytest.approx(east, rel=1e-6, abs=1e-8)

    def test_many_points_at_once_agree_with_one_at_a_time(self, rng):
        pytest.importorskip("pyshtools")
        pytest.importorskip("ducc0")
        from pygeoinf.symmetric_space.sphere import Sobolev

        space = Sobolev(16, 1.5, 0.3)
        field = space.random(rng=rng)
        points = [space.random_point(rng=rng) for _ in range(200)]
        batched = space.evaluate_gradient(field, points)
        single = np.stack([space.evaluate_gradient(field, [p])[0] for p in points[:5]])
        assert np.allclose(batched[:5], single, rtol=1e-9, atol=1e-12)
