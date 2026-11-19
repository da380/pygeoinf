"""The strip and the slab: flat layers, Fourier across and spectral elements
in depth.

The independent routes are ``planetmodel``'s own sampler of a flat layer,
``BoxGRF``, whose covariance between points is summed from its basis without
any of the space's operators; the closed-form Matern function of the plane
and of three-space, which a prior in a domain large against its length scale
must reproduce; the grid against the point, values synthesized on the grid
by FFT against the same field evaluated through the element polynomials and
closed-form cosines; and the three routes of point evaluation against one
another.
"""

import numpy as np
import pytest
from planetmodel.cartesian import BoxGRF
from planetmodel.fourier import PaddedDomain
from planetmodel.testing import check_layer_basis
from scipy.special import gamma, kv

from pygeoinf import GaussianMeasure, LinearOperator, disk
from pygeoinf import slab as slab_module
from pygeoinf import strip as strip_module
from pygeoinf.slab import Slab
from pygeoinf.spectral_element import base as spectral_base
from pygeoinf.spectral_element.base import _RadialGroup
from pygeoinf.strip import Strip
from pygeoinf.testing import (
    check_coordinates,
    check_measure,
    check_operator,
    check_space,
    check_traits,
    check_white_noise,
)


def strip(order, /, *, shape=12, **options):
    """A strip that is its whole lateral period, on a grid of even size."""
    settings = dict(lateral=2.0, length_scale=0.3, element_length=0.3)
    settings.update(options)
    return Strip(shape, order=order, **settings).with_order(order)


def bounded_strip(order, /, *, shape=11, **options):
    """A strip bounded across, inside a wider period, its top below zero
    depth."""
    settings = dict(
        lateral=PaddedDomain([(0.5, 2.0)], padding=0.6),
        top=0.2,
        bottom=1.0,
        length_scale=0.3,
        element_length=0.3,
    )
    settings.update(options)
    return Strip(shape, order=order, **settings).with_order(order)


def slab(order, /, *, shape=(6, 5), **options):
    """A slab that is its whole lateral period, one axis even and one odd."""
    settings = dict(lateral=(2.0, 1.5), length_scale=0.4, element_length=0.4)
    settings.update(options)
    return Slab(shape, order=order, **settings).with_order(order)


def bounded_slab(order, /, *, shape=(5, 6), **options):
    """A slab bounded along ``x`` and periodic along ``y``."""
    settings = dict(
        lateral=PaddedDomain([(0.0, 1.0), (-1.0, 0.5)], padding=(0.5, 0.0)),
        length_scale=0.4,
        element_length=0.4,
    )
    settings.update(options)
    return Slab(shape, order=order, **settings).with_order(order)


GEOMETRIES = [strip, bounded_strip, slab, bounded_slab]


def grid_points(space, /):
    """The grid points of the domain proper, as points ``(depth, x[, y])``,
    and the mask that reads their values off a vector."""
    axes = np.meshgrid(space.depths, *space.lateral_axes, indexing="ij")
    mask = space.interior_mask
    return np.column_stack([axis[mask] for axis in axes]), mask


def point_covariance(space, measure, points, /, *, unsafe=False):
    """``E C E*``: the covariance of the field between points."""
    look = space.point_evaluation_operator(points, unsafe=unsafe)
    return (look @ measure.covariance @ look.adjoint).matrix(form="components")


class TestTheGroups:
    """The shared machinery takes a group's components as a slice or as an
    array of indices, and does the same with either."""

    @staticmethod
    def scattered(space, /, *, reverse=False):
        groups = []
        for group in space._radial_groups():
            entries = np.arange(group.entries.start, group.entries.stop)
            entries = entries.reshape(group.count, -1)
            groups.append(
                _RadialGroup.scattered(
                    group.basis, entries[::-1] if reverse else entries, group[3]
                )
            )
        return groups

    @pytest.mark.parametrize("reverse", [False, True])
    def test_index_arrays_act_as_the_slices_they_spell(self, reverse, rng, monkeypatch):
        """A disk's prior with scales of its own, its groups given once as
        the consecutive runs they are and once as the same components
        listed; the runs of a group in either order, as each is treated
        alike."""
        space = disk.Sobolev(5, 2.0, 0.3, element_length=0.3)
        options = dict(length_scale=0.4, lateral_length_scale=0.25)
        sliced = space.sobolev_measure(2.5, **options)
        variance = space.sobolev_pointwise_variance(2.5, **options)
        listed = self.scattered(space, reverse=reverse)
        monkeypatch.setattr(space, "_radial_groups", lambda: listed)
        indexed = space.sobolev_measure(2.5, **options)
        x = space.random(rng=rng)
        for name in ("covariance", "covariance_factor", "precision"):
            assert np.allclose(
                getattr(indexed, name)(x), getattr(sliced, name)(x), rtol=1e-13
            )
        assert np.allclose(
            space.sobolev_pointwise_variance(2.5, **options), variance, rtol=1e-13
        )

    def test_a_group_knows_its_first_run_either_way(self):
        space = disk.Sobolev(3, 2.0, 0.3, element_length=0.3)
        for sliced, listed in zip(space._radial_groups(), self.scattered(space)):
            assert isinstance(sliced.entries, slice)
            assert listed.count == sliced.count
            assert np.array_equal(
                np.arange(space.dim)[sliced.first], np.arange(space.dim)[listed.first]
            )

    def test_entries_of_the_wrong_shape_are_refused(self):
        basis = disk.Sobolev(3, 2.0, 0.3, element_length=0.3).basis[1]
        with pytest.raises(ValueError, match="entries"):
            _RadialGroup.scattered(basis, np.arange(basis.nmodes + 1)[None, :], 1.0)


class TestTheSpace:
    @pytest.mark.parametrize("build", GEOMETRIES)
    @pytest.mark.parametrize("order", [0.0, 2.0])
    def test_the_axioms(self, build, order, rng):
        space = build(order)
        check_space(space, rng=rng, rebuild=lambda: build(order))
        check_coordinates(space, rng=rng)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_basis_holds_planetmodels_contract(self, build, rng):
        """``planetmodel``'s own check of a layer basis: one basis per
        distinct ``|k|^2``, the blocks of the components tiling the vector
        in the transform's order, analysis inverting synthesis."""
        space = build(0.0)
        family = space.basis.family
        assert (family.weight, family.lateral, family.index) == ("one", "one", "value")
        check_layer_basis(space.basis, rng=rng)
        assert space.basis.transform is space.transform
        assert space.dim == space.basis.size

    def test_with_order_names_the_subclass_and_shares_the_basis(self):
        for module, build in ((strip_module, strip), (slab_module, slab)):
            sobolev = build(2.0)
            lebesgue = sobolev.with_order(0.0)
            assert type(sobolev) is module.Sobolev
            assert type(lebesgue) is module.Lebesgue
            assert lebesgue.basis is sobolev.basis
            assert sobolev.shares_vectors_with(lebesgue)
            assert sobolev != lebesgue
        assert strip(2.0) == strip_module.Sobolev(
            12, 2.0, 0.3, lateral=2.0, element_length=0.3
        )
        assert slab(0.0) == slab_module.Lebesgue(
            (6, 5), lateral=(2.0, 1.5), length_scale=0.4, element_length=0.4
        )
        assert not strip(2.0).shares_vectors_with(strip(2.0, shape=13))
        assert strip(2.0) != bounded_strip(2.0)
        assert repr(strip(2.0)).startswith("Sobolev(shape=(12,)")

    def test_depth_is_padded_above_the_top_and_below_the_base(self):
        """Two length scales by default, four under the natural condition,
        above a free surface as below the base: the plain measure has no
        special point."""
        space = strip(0.0)
        assert space.padding == pytest.approx((0.6, 0.6))
        assert strip(0.0, boundary=None).padding == pytest.approx((1.2, 1.2))
        assert space.depths[0] == pytest.approx(-0.6)
        assert space.depths[-1] == pytest.approx(1.6)
        assert (space.top, space.bottom) == space.bounds == (0.0, 1.0)
        assert space.interior_depths[0] == 0.0 and space.interior_depths[-1] == 1.0
        assert space.robin == pytest.approx((0.7 / 0.3, 0.7 / 0.3))
        assert strip(0.0, boundary=None).robin == (0.0, 0.0)
        lower = bounded_strip(0.0, padding=(0.1, 0.4))
        assert lower.depths[0] == pytest.approx(0.1)
        assert lower.depths[-1] == pytest.approx(1.4)

    def test_the_grid(self):
        space = strip(0.0)
        assert space.grid_shape == (space.depths.size, 12)
        assert space.lateral_shape == (12,)
        assert np.allclose(space.x, 2.0 * np.arange(12) / 12)
        assert space.spatial_dimension == 2
        assert space.domain_volume == pytest.approx(2.0)
        assert space.interior_mask.all(axis=1).sum() == space.interior_depths.size
        assert space.interior_values(space.zero()).shape == (
            space.interior_depths.size,
            12,
        )
        box = slab(0.0)
        assert box.grid_shape == (box.depths.size, 6, 5)
        assert np.allclose(box.y, 1.5 * np.arange(5) / 5)
        assert box.spatial_dimension == 3
        assert box.domain_volume == pytest.approx(3.0)

    def test_a_bounded_domain_leaves_its_lateral_padding_out(self):
        """The period is the bounds widened by the padding, the grid runs
        from its lower edge, and the domain proper is the run of grid
        points between the bounds."""
        space = bounded_strip(0.0)
        assert space.lateral_domain.lengths == pytest.approx((2.7,))
        assert space.x[0] == pytest.approx(-0.1)
        inside = (space.x >= 0.5) & (space.x <= 2.0)
        assert np.array_equal(space.interior_mask.any(axis=0), inside)
        assert space.interior_values(space.zero()).shape == (
            space.interior_depths.size,
            int(inside.sum()),
        )
        assert space.domain_volume == pytest.approx(1.5 * 0.8)
        box = bounded_slab(0.0)
        along_x = (box.x >= 0.0) & (box.x <= 1.0)
        assert box.interior_values(box.zero()).shape == (
            box.interior_depths.size,
            int(along_x.sum()),
            6,
        )
        assert box.domain_volume == pytest.approx(1.5)

    def test_the_lateral_box_is_read_as_planetmodel_reads_one(self):
        """Periods from the origin, a padded domain, or one of
        ``planetmodel``'s own flat domains, which state bounds and padding."""
        from planetmodel.cartesian import Box, Rectangle

        domain = PaddedDomain([(0.5, 2.0)], padding=0.6)
        assert Strip(8, lateral=Rectangle((0.5, 2.0), padding=0.6)) == Strip(
            8, lateral=domain
        )
        assert Strip(8, lateral=3.0).lateral_domain == PaddedDomain([(0.0, 3.0)])
        assert Strip(8).lateral_domain.lengths == (1.0,)
        box = Slab((4, 4), lateral=Box([(0.0, 1.0), (0.0, 2.0)], padding=0.5))
        assert box.lateral_domain.lengths == pytest.approx((2.0, 3.0))
        assert Slab((4, 4)).lateral_domain.lengths == (1.0, 1.0)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_labels_are_the_packing(self, build):
        space = build(0.0)
        basis, transform = space.basis, space.transform
        assert space.wavevectors.shape == (space.spatial_dimension - 1, space.dim)
        for component in range(transform.size):
            block = basis.block(component)
            modes = basis.basis_of(component)
            assert np.all(
                space.wavevectors[:, block] == transform.wavevectors[:, [component]]
            )
            assert np.all(space.phases[block] == transform.phases[component])
            assert np.array_equal(space.depth_indices[block], np.arange(modes.nmodes))
            assert np.array_equal(space.eigenvalues[block], modes.theta)
        assert space.dim == int(np.sum(basis.multiplicity * basis.nmodes))

    def test_the_default_truncation_is_isotropic(self):
        """Every mode of depth below the first eigenvalue of the axis's
        Nyquist wavenumber, and no other: with ``L`` constant and the natural
        condition that eigenvalue is ``1 + L^2 k^2``, its mode constant in
        depth, so the modes kept at ``k = 0`` have wavenumbers in depth up to
        the lateral band's."""
        space = strip(0.0, shape=24, boundary=None, element_length=None)
        nyquist = 2.0 * np.pi * 12 / 2.0
        bound = 1.0 + 0.3**2 * nyquist**2
        family = space.basis.family
        for value, modes in zip(space.basis.eigenvalues, space.basis):
            assert modes.nmodes == np.sum(
                family.eigvalsh(float(value)) <= bound * 1.0001
            )
        assert np.all(np.diff(space.basis.nmodes) <= 0)
        assert space.basis.nmodes[-1] == 1
        assert space.eigenvalues.max() == pytest.approx(bound, rel=1e-6)
        in_depth = np.sqrt(space.basis[0].theta[-1] - 1.0) / 0.3
        assert in_depth == pytest.approx(nyquist, rel=0.1)

    def test_the_corners_of_a_grid_of_two_axes_keep_one_mode_each(self):
        """The bound is the Nyquist wavenumber of the coarser axis; the
        wavevectors longer than that, which the grid holds whole, keep the
        one mode a basis must have."""
        space = slab(0.0, shape=(8, 8), lateral=(2.0, 1.0), boundary=None)
        band = (2.0 * np.pi * 4 / 2.0) ** 2
        bound = 1.0 + 0.4**2 * band
        beyond = space.basis.eigenvalues > band * 1.0001
        assert beyond.any() and not beyond.all()
        assert np.all(space.basis.nmodes[beyond] == 1)
        assert np.all(space.basis.nmodes[~beyond] >= 1)
        first = np.array([modes.theta[0] for modes in space.basis])
        assert np.all(first[beyond] > bound)
        kept = np.concatenate([modes.theta for modes in space.basis][: (~beyond).sum()])
        assert kept.max() <= bound * 1.0001

    def test_the_truncation_can_be_stated(self):
        assert np.all(strip(0.0, depth_modes=4).basis.nmodes == 4)
        assert strip(0.0, depth_modes=4).dim == 4 * 12
        bounded = strip(0.0, max_eigenvalue=30.0)
        counts = bounded.basis.nmodes
        firsts = np.array([modes.theta[0] for modes in bounded.basis])
        assert np.all(counts[firsts > 30.0] == 1) and np.any(firsts > 30.0)
        within = np.concatenate(
            [m.theta for m, first in zip(bounded.basis, firsts) if first <= 30.0]
        )
        assert within.max() <= 30.0
        # A bound below every eigenvalue leaves the one mode of each
        # component, which a grid cannot drop.
        assert strip(0.0, max_eigenvalue=0.5).dim == 12
        full = strip(0.0, depth_modes=strip(0.0).depths.size)
        assert full.dim == 12 * full.depths.size

    def test_what_the_constructor_refuses(self):
        with pytest.raises(ValueError, match="not both"):
            strip(0.0, depth_modes=4, max_eigenvalue=30.0)
        with pytest.raises(ValueError, match="holds between one"):
            strip(0.0, depth_modes=10_000)
        with pytest.raises(ValueError, match="1 axis"):
            Strip((8, 8))
        with pytest.raises(ValueError, match="2 axes"):
            Slab(8)
        with pytest.raises(ValueError, match="2 axes"):
            Slab((8, 8), lateral=3.0)
        with pytest.raises(ValueError, match="top < bottom"):
            Strip(8, top=1.0, bottom=1.0)
        with pytest.raises(ValueError, match="positive"):
            Strip(8, length_scale=-1.0)
        with pytest.raises(ValueError, match="element length"):
            Strip(8, element_length=0.0)

    def test_the_default_element_length_resolves_the_modes_kept(self):
        space = Strip(24, lateral=2.0, length_scale=0.3)
        fine = Strip(24, lateral=2.0, length_scale=0.3, element_length=0.02)
        assert space.basis.nmodes.tolist() == fine.basis.nmodes.tolist()
        assert np.allclose(space.eigenvalues, fine.eigenvalues, rtol=5e-3)

    def test_so_it_does_for_a_length_scale_that_varies_with_depth(self):
        """The bound is then a wavenumber where the length scale is smallest,
        and the mesh is sized to that and not to the ratio of the scales."""

        def scale(depth):
            return 0.15 + 0.35 * np.asarray(depth) ** 2

        space = Strip(24, lateral=2.0, length_scale=scale)
        fine = Strip(24, lateral=2.0, length_scale=scale, element_length=0.01)
        assert space.depths.size < fine.depths.size / 4
        assert space.basis.nmodes.tolist() == fine.basis.nmodes.tolist()
        assert np.allclose(space.eigenvalues, fine.eigenvalues, rtol=5e-3)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_lebesgue_inner_product_is_the_volume_quadrature(self, build, rng):
        """``int u v`` over the padded mesh and the period, summed on the
        grid with its own weights, for fields in the span."""
        space = build(0.0)
        u, v = space.random(rng=rng), space.random(rng=rng)
        family = space.basis.family
        in_depth = np.zeros(space.depths.size)
        np.add.at(in_depth, family.mesh.gmap, family.element_mass())
        cell = space.transform.volume / space.transform.size
        shape = (-1,) + (1,) * (space.spatial_dimension - 1)
        quadrature = np.sum(in_depth.reshape(shape) * cell * u * v)
        assert space.inner_product(u, v) == pytest.approx(quadrature, rel=1e-10)

    @pytest.mark.parametrize("build", [strip, bounded_slab])
    def test_white_noise_has_identity_covariance(self, build, rng):
        check_white_noise(
            build(2.0, depth_modes=3, shape=6 if build is strip else (3, 4)),
            rng=rng,
            samples=6000,
            rtol=0.14,
        )

    @pytest.mark.parametrize("unit", [1.0e-3, 7.3, 6371.0])
    def test_the_same_geometry_in_other_units(self, unit):
        here = Strip(12, lateral=2.0, bottom=1.0, length_scale=0.3)
        there = Strip(12, lateral=2.0 * unit, bottom=unit, length_scale=0.3 * unit)
        assert there.grid_shape == here.grid_shape
        assert there.basis.nmodes.tolist() == here.basis.nmodes.tolist()
        assert np.allclose(there.eigenvalues, here.eigenvalues, rtol=1e-9)


class TestPointEvaluation:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_every_basis_function_is_its_own_grid_values(self, build):
        """One component at a time, so that an error in the order of the
        blocks cannot hide in a sum: closed-form cosines and the element
        polynomials against the FFT and the nodal values."""
        space = build(2.0)
        points, mask = grid_points(space)
        rows = space.basis_matrix(points)
        for k in range(space.dim):
            unit = np.zeros(space.dim)
            unit[k] = 1.0
            on_grid = space.from_components(unit)[mask]
            assert np.allclose(
                rows[:, k], on_grid, atol=1e-10 * max(1.0, np.abs(on_grid).max())
            )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_field_in_the_span_is_its_grid_values_at_the_grid(self, build, rng):
        space = build(2.0)
        x = space.random(rng=rng)
        points, mask = grid_points(space)
        scale = np.abs(x[mask]).max()
        assert np.allclose(space.evaluate(x, points), x[mask], atol=1e-10 * scale)
        assert space.basis_at(points[4]) == pytest.approx(space.basis_matrix(points)[4])

    @pytest.mark.parametrize("build", GEOMETRIES)
    @pytest.mark.parametrize("route", ["dense", "local", "nufft"])
    def test_the_operator_and_its_adjoint(self, build, route, rng):
        """Every route against the rows of the basis matrix, and each
        adjoint against its action: to rounding where the lateral sums are
        direct, to the accuracy asked of the non-uniform transforms where
        they are not."""
        space = build(2.0)
        points = [space.random_point(rng=rng) for _ in range(7)]
        operator = space.point_evaluation_operator(points, route=route)
        check_operator(operator, rng=rng, rtol=1e-9)
        x = space.random(rng=rng)
        exact = space.basis_matrix(points) @ space.to_components(x)
        tolerance = 1e-9 if route == "nufft" else 1e-12
        assert np.allclose(operator(x), exact, atol=tolerance * np.abs(exact).max())
        assert np.allclose(space.evaluate(x, points), exact, atol=1e-12)
        assert operator(x)[1] == pytest.approx(space.dirac(points[1])(x), rel=1e-8)

    @pytest.mark.parametrize("build", [strip, slab])
    def test_auto_is_dense_within_the_limit_and_nufft_beyond(
        self, build, rng, monkeypatch
    ):
        space = build(2.0)
        points = [space.random_point(rng=rng) for _ in range(20)]
        x = space.random(rng=rng)
        within = space.point_evaluation_operator(points)
        assert within.known_matrix(form="galerkin") is not None
        monkeypatch.setattr(
            spectral_base, "_DENSE_EVALUATION_LIMIT", 8 * len(points) * space.dim - 1
        )
        beyond = space.point_evaluation_operator(points, epsilon=1e-6)
        assert beyond.known_matrix(form="galerkin") is None
        named = space.point_evaluation_operator(points, route="nufft", epsilon=1e-6)
        assert np.array_equal(beyond(x), named(x))
        scale = np.abs(within(x)).max()
        assert 1e-13 < np.abs(beyond(x) - within(x)).max() / scale < 1e-4

    def test_a_looser_non_uniform_transform_is_looser(self, rng):
        space = slab(2.0)
        points = [space.random_point(rng=rng) for _ in range(40)]
        x = space.random(rng=rng)
        exact = space.point_evaluation_operator(points, route="dense")(x)
        errors = [
            np.abs(
                space.point_evaluation_operator(points, route="nufft", epsilon=eps)(x)
                - exact
            ).max()
            / np.abs(exact).max()
            for eps in (1e-3, 1e-12)
        ]
        assert errors[1] < 1e-10 < errors[0] < 1e-2

    def test_a_point_is_depth_first_and_depth_is_positive_downward(self):
        """A function that tells the coordinates apart, periodic across,
        projected and read back: the first coordinate is the depth, and the
        mesh runs from above the top, at negative depth, to below the
        base."""
        space = strip(2.0, depth_modes=10)
        field = space.project_function(lambda p: p[0] + 10.0 * np.cos(np.pi * p[1]))
        assert space.evaluate(field, [(0.25, 1.0), (0.9, 0.5)]) == pytest.approx(
            [-9.75, 0.9], abs=5e-3
        )
        box = slab(2.0, depth_modes=10)
        field = box.project_function(
            lambda p: (
                p[0]
                + 10.0 * np.cos(np.pi * p[1])
                + 100.0 * np.sin(2.0 * np.pi * p[2] / 1.5)
            )
        )
        wave = 100.0 * np.sin(2.0 * np.pi * 0.25 / 1.5)
        assert box.evaluate(field, [(0.25, 0.5, 0.25)]) == pytest.approx(
            [0.25 + wave], abs=5e-3
        )
        assert box.depths[0] < 0.0 == box.top < box.bottom < box.depths[-1]
        assert np.all(np.diff(box.depths) > 0.0)

    def test_a_periodic_axis_wraps_and_a_bounded_one_refuses(self, rng):
        space = strip(2.0)
        x = space.random(rng=rng)
        assert space.evaluate(x, [(0.4, -0.3)]) == pytest.approx(
            space.evaluate(x, [(0.4, 1.7)]), abs=1e-12
        )
        bounded = bounded_strip(2.0)
        for route in ("dense", "local", "nufft"):
            with pytest.raises(ValueError, match="lateral axis 0"):
                bounded.point_evaluation_operator([(0.5, 2.3)], route=route)
        box = bounded_slab(2.0)
        y = box.random(rng=rng)
        assert box.evaluate(y, [(0.4, 0.5, 0.75)]) == pytest.approx(
            box.evaluate(y, [(0.4, 0.5, -0.75)]), abs=1e-12
        )
        with pytest.raises(ValueError, match="lateral axis 0"):
            box.dirac((0.4, 1.2, 0.0))

    @pytest.mark.parametrize("route", ["dense", "local", "nufft"])
    def test_a_depth_outside_the_layer_is_refused(self, route):
        for depth in (-0.1, 1.2):
            with pytest.raises(ValueError):
                strip(2.0).point_evaluation_operator([(depth, 0.5)], route=route)
        with pytest.raises(ValueError, match="route"):
            strip(2.0).point_evaluation_operator([(0.5, 0.5)], route="fast")
        with pytest.raises(ValueError, match="At least one"):
            strip(2.0).point_evaluation_operator([], route=route)

    def test_evaluation_off_the_grid_converges_for_a_smooth_function(self, rng):
        """A function periodic across and smooth in depth, fitted and read
        at points off the grid in both: the error falls as the grid and the
        modes of depth are refined together."""

        def smooth(p):
            return np.exp(-p[0]) * np.cos(np.pi * p[1] + 0.3) + 0.2 * p[
                0
            ] ** 2 * np.sin(2.0 * np.pi * p[1])

        points = [(rng.uniform(0.0, 1.0), rng.uniform(0.0, 2.0)) for _ in range(40)]
        exact = np.array([smooth(p) for p in points])
        errors = []
        for shape in (5, 9, 13, 17):
            space = Strip(shape, order=2.0, lateral=2.0, length_scale=0.3)
            field = space.project_function(smooth)
            errors.append(np.abs(space.evaluate(field, points) - exact).max())
        assert errors[0] > 1e-2
        assert all(b < 0.05 * a for a, b in zip(errors[:-1], errors[1:]))
        assert errors[-1] < 1e-7

        def wave(p):
            return (
                np.exp(-p[0])
                * np.cos(np.pi * p[1] + 0.3)
                * np.sin(4.0 * np.pi * p[2] / 3)
                + 0.2 * p[0] ** 2
            )

        points = [
            (rng.uniform(0.0, 1.0), rng.uniform(0.0, 2.0), rng.uniform(0.0, 1.5))
            for _ in range(40)
        ]
        exact = np.array([wave(p) for p in points])
        errors = []
        for modes in (3, 6, 12):
            space = slab(2.0, shape=(5, 5), depth_modes=modes, element_length=0.15)
            field = space.project_function(wave)
            errors.append(np.abs(space.evaluate(field, points) - exact).max())
        assert errors[0] > 1e-3
        assert errors[1] < 0.1 * errors[0] and errors[2] < 0.01 * errors[1]

    def test_random_points_lie_in_the_domain_proper(self, rng):
        space = bounded_slab(0.0, top=0.5, bottom=2.0)
        points = np.array([space.random_point(rng=rng) for _ in range(2000)])
        assert np.all((points[:, 0] >= 0.5) & (points[:, 0] <= 2.0))
        assert np.all((points[:, 1] >= 0.0) & (points[:, 1] <= 1.0))
        assert np.all((points[:, 2] >= -1.0) & (points[:, 2] <= 0.5))
        assert points.mean(axis=0) == pytest.approx([1.25, 0.5, -0.25], abs=0.04)


def nufft_with_the_points_given(space, points, /):
    """The non-uniform route's values and its transpose, with the lateral
    points handed to ``planetmodel``'s transform at every application: per
    element of the depth mesh, the coefficient functions at its nodes
    synthesised at the lateral positions of its points and weighted by
    their Lagrange weights in depth."""
    points = np.asarray(points, dtype=float)
    mesh = space.basis.family.mesh
    elements, weights = mesh.interpolation_weights(points[:, 0])
    transform = space.transform

    def values(components):
        functions = space._nodal_functions(components)
        out = np.empty(len(points))
        for element in np.unique(elements):
            here = elements == element
            local = transform.synthesise(
                functions[:, mesh.gmap[element]].T, points[here, 1:]
            )
            out[here] = np.einsum("ji,ij->i", local, weights[here])
        return out

    def derivative_components(data):
        functions = np.zeros((transform.size, mesh.nglob))
        for element in np.unique(elements):
            here = elements == element
            weighted = weights[here] * data[here][:, None]
            functions[:, mesh.gmap[element]] += transform.accumulate(
                weighted.T, points[here, 1:]
            ).T
        return space._nodal_projection(functions)

    return values, derivative_components


class TestTheNonUniformRouteKeepsItsTransforms:
    """The non-uniform route prepares the lateral points of each element of
    the depth mesh once, ``planetmodel``'s transform at them, and applies
    it at every application: the results are those of the transform with
    the points given, to rounding, however often it is applied."""

    @staticmethod
    def close(actual, expected):
        scale = np.abs(expected).max()
        return np.allclose(actual, expected, rtol=0.0, atol=1e-13 * scale)

    @pytest.mark.parametrize("build", [bounded_strip, slab])
    def test_repeated_applications_are_the_transform_with_the_points_given(
        self, build, rng
    ):
        space = build(2.0)
        points = np.array([space.random_point(rng=rng) for _ in range(60)])
        operator = space.point_evaluation_operator(points, route="nufft")
        values, derivative_components = nufft_with_the_points_given(space, points)
        x = space.random(rng=rng)
        y = rng.normal(size=60)
        expected = values(space.to_components(x))
        spread = space.from_components(space.solve_gram(derivative_components(y)))
        for _ in range(3):
            assert self.close(operator(x), expected)
            assert self.close(operator.adjoint(y), spread)

    def test_each_element_is_prepared_once(self, rng, monkeypatch):
        from planetmodel.fourier import Transform

        made = []
        at = Transform.at

        def counting(transform, points, /, **options):
            made.append(len(points))
            return at(transform, points, **options)

        monkeypatch.setattr(Transform, "at", counting)
        space = slab(2.0)
        points = np.array([space.random_point(rng=rng) for _ in range(60)])
        operator = space.point_evaluation_operator(points, route="nufft")
        elements, _ = space.basis.family.mesh.interpolation_weights(points[:, 0])
        assert sorted(made) == sorted(np.bincount(elements)[np.unique(elements)])
        x = space.random(rng=rng)
        for _ in range(3):
            operator(x)
            operator.adjoint(rng.normal(size=60))
        assert len(made) == np.unique(elements).size

    def test_a_copy_or_a_pickle_applies_as_the_original(self, rng):
        import copy

        from .conftest import joblib_cloudpickle

        cloudpickle = joblib_cloudpickle()

        space = bounded_slab(2.0)
        points = [space.random_point(rng=rng) for _ in range(30)]
        operator = space.point_evaluation_operator(points, route="nufft")
        x = space.random(rng=rng)
        y = rng.normal(size=30)
        values = operator(x)
        adjoint = space.to_components(operator.adjoint(y))
        for copied in (
            cloudpickle.loads(cloudpickle.dumps(operator)),
            copy.deepcopy(operator),
        ):
            assert self.close(copied(x), values)
            assert self.close(space.to_components(copied.adjoint(y)), adjoint)
        assert self.close(operator(x), values)


class TestProjection:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_fit_over_the_domain_is_right_on_it(self, build):
        """A function periodic along the unbounded axes, fitted in depth
        with enough modes, at the grid points of the domain proper."""
        space = build(2.0, depth_modes=10, element_length=0.15)
        periods = space.lateral_domain.lengths

        def function(p):
            lateral = sum(
                np.cos(2.0 * np.pi * x / period) for x, period in zip(p[1:], periods)
            )
            return (1.0 + p[0] ** 2) * (2.0 + lateral)

        field = space.project_function(function)
        points, mask = grid_points(space)
        exact = np.array([function(p) for p in points])
        assert np.allclose(field[mask], exact, atol=2e-4 * np.abs(exact).max())
        assert np.allclose(space.truncate(field), field, atol=1e-9)

    def test_the_function_is_never_called_outside_the_domain(self):
        space = bounded_slab(0.0, top=0.5, bottom=2.0)
        called = []

        def function(p):
            called.append(p)
            return 1.0

        for extension in ("fit", "odd", "constant"):
            space.project_function(function, extension=extension)
        called = np.array(called)
        assert called[:, 0].min() >= 0.5 and called[:, 0].max() <= 2.0
        assert called[:, 1].min() >= 0.0 and called[:, 1].max() <= 1.0

    def test_constant_keeps_a_positive_function_positive_everywhere(self):
        """In depth and across the lateral padding alike, where the other
        extensions roll the function off to zero."""
        space = bounded_strip(0.0)
        constant = space.project_function(lambda p: 1.0 + p[0], extension="constant")
        assert constant.min() >= 1.2 - 1e-12
        assert np.allclose(constant[space.depths < 0.2], 1.2)
        odd = space.project_function(lambda p: 1.0 + p[0], extension="odd")
        assert np.allclose(odd[space.interior_mask], constant[space.interior_mask])
        assert np.abs(odd[:, 0]).max() < 0.2
        with pytest.raises(ValueError, match="extension"):
            space.project_function(lambda p: 1.0, extension="even")

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_integral_runs_over_the_domain_alone(self, build, rng):
        """Against the lateral basis's own integrals, taken by a fine
        midpoint rule over the lateral domain, and the quadrature in depth
        of the domain's elements."""
        space = build(2.0)
        x = space.random(rng=rng)
        bounds = space.lateral_domain.bounds
        count = 200
        axes = [
            low + (np.arange(count) + 0.5) * (high - low) / count
            for low, high in bounds
        ]
        lateral = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, len(bounds)
        )
        cell = space.lateral_domain.domain_volume / lateral.shape[0]
        integrals = cell * space.transform.basis(lateral).sum(axis=0)
        functions = space.basis.synthesise(space.to_components(x))
        in_depth = functions @ space.basis[0].weights()
        assert space.integral_functional()(x) == pytest.approx(
            integrals @ in_depth, rel=1e-3
        )

    def test_the_integral_of_one_is_the_volume(self):
        for space in (
            strip(0.0, depth_modes=12, element_length=0.15),
            slab(0.0, depth_modes=12, element_length=0.2),
        ):
            one = space.project_function(lambda p: 1.0)
            assert space.integral_functional()(one) == pytest.approx(
                space.domain_volume, rel=1e-6
            )

    def test_the_integral_reads_through_the_components(self, rng):
        space = strip(2.0)
        integral = space.integral_functional()
        u, x = space.random(rng=rng), space.random(rng=rng)
        product = space.multiply(u, x)
        assert integral(product) == pytest.approx(
            integral.derivative_components @ space.to_components(product), rel=1e-10
        )
        check_operator(
            integral @ space.multiplication_operator(u),
            rng=rng,
            measure=space.sobolev_measure(2.0),
        )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_multiplication_is_self_adjoint_on_lebesgue_only(self, build, rng):
        f = build(0.0).random(rng=rng)
        lebesgue = build(0.0).multiplication_operator(f)
        check_operator(lebesgue, rng=rng)
        check_traits(lebesgue, rng=rng)
        check_operator(build(2.0).multiplication_operator(f), rng=rng)
        projection = build(0.0).support_projection()
        x = build(0.0).random(rng=rng)
        assert np.all(projection(x)[~build(0.0).interior_mask] == 0.0)


def matern(distance, smoothness, length, /):
    """The Matern correlation function, one at zero distance."""
    distance = np.asarray(distance, dtype=float)
    out = np.ones_like(distance)
    away = distance > 0.0
    scaled = distance[away] / length
    out[away] = (
        2.0 ** (1.0 - smoothness)
        / gamma(smoothness)
        * scaled**smoothness
        * kv(smoothness, scaled)
    )
    return out


class TestAgainstPlanetmodel:
    """The space's priors against ``planetmodel``'s sampler of the same
    field, which shares the family and the transform and nothing built
    here."""

    @pytest.mark.parametrize(
        "module, shape, lateral",
        [
            (strip_module, (10,), 3.0),
            (strip_module, (9,), PaddedDomain([(0.5, 2.5)], padding=0.7)),
            (slab_module, (6, 5), (2.0, 1.5)),
            (slab_module, (5, 6), PaddedDomain([(0.0, 1.0), (0.0, 1.2)], padding=0.5)),
        ],
    )
    @pytest.mark.parametrize("order", [0.0, "admits points"])
    def test_a_matern_prior_has_box_grfs_covariance(
        self, module, shape, lateral, order, rng
    ):
        """With every mode of the mesh kept and the same mesh, the two are
        one field: a Matern field of smoothness ``nu``, covariance ``A^-p``
        with ``p = nu + d / 2``, calibrated to one standard deviation, its
        covariance between pairs of points anywhere in the domain. On a
        Sobolev space of order ``s`` a covariance is an operator in that
        space's metric, so the same field is the measure of order ``p - s``
        there, and its points have values."""
        dimension = len(shape) + 1
        smoothness, length, deviation = 1.5 + dimension - 2, 0.4, 0.7
        power = smoothness + dimension / 2.0
        reach = 1.5 * np.sqrt(8.0 * smoothness) * length
        options = dict(
            lateral=lateral,
            top=0.2,
            bottom=1.4,
            length_scale=length,
            padding=reach,
            boundary=None,
            element_length=0.2,
        )
        nodes = module.Lebesgue(shape, depth_modes=1, **options).depths.size
        space = module.Lebesgue(shape, depth_modes=nodes, **options)
        oracle = BoxGRF(
            0.2,
            1.4,
            space.transform,
            smoothness,
            length,
            sigma=deviation,
            drmax=0.2,
            pad_factor=1.5,
            tol=0.0,
        )
        assert np.array_equal(oracle.mesh.rglob, space.depths)
        assert oracle.size == space.dim
        points = np.array([space.random_point(rng=rng) for _ in range(8)])
        exact = oracle.covariance(points, points)

        if order != 0.0:
            order = 0.5 * dimension + 0.25
            space = space.with_order(order)
        measure = space.sobolev_measure(power - order, pointwise_std=deviation)
        got = point_covariance(space, measure, points, unsafe=order == 0.0)
        assert np.allclose(got, exact, rtol=0.0, atol=1e-10 * deviation**2)
        # The calibration is the same number at every node in depth.
        raw = space.interior_values(space.sobolev_pointwise_variance(power - order))
        profile = raw.reshape(raw.shape[0], -1)[:, 0]
        assert np.allclose(deviation / np.sqrt(profile), oracle.scale, rtol=1e-12)

    @pytest.mark.parametrize(
        "module, shape, period, bottom, tolerance",
        [
            (strip_module, (61,), 24.0, 12.0, 1e-4),
            (slab_module, (21, 21), 14.0, 6.0, 5e-3),
        ],
    )
    def test_in_a_large_domain_the_covariance_is_the_matern_function(
        self, module, shape, period, bottom, tolerance
    ):
        """``A^-p`` in ``d`` dimensions has the kernel ``sigma^2`` times the
        Matern function of smoothness ``nu = p - d / 2``, with ``sigma^2 =
        Gamma(nu) / (Gamma(p) (4 pi)^(d/2) L^d)``: the value, uncalibrated,
        and its fall along a lateral direction, along depth and obliquely,
        from the middle of a domain many length scales across. What is left
        is the periodic images and the ends of the mesh."""
        dimension = len(shape) + 1
        order, power = 0.5 * dimension + 0.5, 2.5 + 0.5 * dimension
        smoothness = power - 0.5 * dimension
        space = module.Sobolev(
            shape, order, 1.0, lateral=(period,) * len(shape), bottom=bottom
        )
        measure = space.sobolev_measure(power - order)
        variance = gamma(smoothness) / (
            gamma(power) * (4.0 * np.pi) ** (0.5 * dimension)
        )
        centre = np.array([0.5 * bottom] + [0.5 * period] * len(shape))
        lags = np.array([0.0, 0.3, 0.8, 1.5, 2.5])
        directions = [
            np.eye(dimension)[1],
            np.eye(dimension)[0],
            np.ones(dimension) / np.sqrt(dimension),
        ]
        for direction in directions:
            points = [centre + lag * direction for lag in lags]
            got = point_covariance(space, measure, points)[0]
            assert np.allclose(
                got, variance * matern(lags, smoothness, 1.0), atol=tolerance * variance
            )


class TestMeasures:
    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_pointwise_variance_is_the_sum_of_squared_rows(self, build, rng):
        """At grid points, against ``basis_matrix``: once for a function of
        ``A``, where the lateral sum is done in closed form, and once for
        variances that tell the components of one wavenumber apart, where
        it cannot be."""
        space = build(2.0)
        points, mask = grid_points(space)
        rows = space.basis_matrix(points)
        isotropic = space.eigenvalues**-1.5
        ragged = isotropic * rng.uniform(0.5, 1.5, size=space.dim)
        for variances in (isotropic, ragged):
            field = space.pointwise_variance(variances)
            direct = (rows**2) @ (variances / space.metric_values)
            assert np.allclose(field[mask], direct, rtol=1e-10)
        constant = space.pointwise_variance(isotropic)
        lateral = tuple(range(1, constant.ndim))
        assert np.all(np.ptp(constant, axis=lateral) == 0.0)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_the_variance_is_exact_at_the_grid_points(self, build):
        """``diag(E C E*)`` at grid points against the variance field, to
        rounding, for the default prior and one with scales of its own."""
        space = build(2.0)
        points, mask = grid_points(space)
        chosen = np.linspace(0, len(points) - 1, 12).astype(int)
        for options in ({}, dict(length_scale=0.5, lateral_length_scale=0.3)):
            measure = space.sobolev_measure(2.0, **options)
            field = space.sobolev_pointwise_variance(2.0, **options)
            got = np.diag(point_covariance(space, measure, points[chosen]))
            assert np.allclose(got, field[mask][chosen], rtol=1e-10)

    @pytest.mark.parametrize("build", [strip, slab])
    def test_calibrated_draws_have_the_stated_deviation_at_every_grid_point(
        self, build
    ):
        """A draw on the Lebesgue space is the factor applied to a sum of
        the basis functions with independent standard normal weights, so
        its variance at a grid point is the sum of the squares of the
        factor's images of the basis functions there: the target's square,
        to rounding, on the padding too."""
        space = build(0.0, depth_modes=4)
        target = 0.5 + 0.1 * space.project_function(
            lambda p: 1.0 + p[0], extension="constant"
        )
        factor = space.sobolev_measure(2.5, pointwise_std=target).covariance_factor
        variance = sum(
            factor(space.from_components(unit)) ** 2 for unit in np.identity(space.dim)
        )
        assert np.allclose(variance, target**2, rtol=1e-12)

    def test_between_grid_points_an_even_grid_dips_by_its_nyquist_cosine(self):
        """Midway between the points of the lateral grid, at nodes in depth.
        On an odd grid every cosine has its sine and the variance is the
        grid's, to rounding. On an even grid the cosine at the Nyquist
        wavenumber has none, and vanishes midway, so the variance there is
        short by that mode's share and by nothing else."""
        for shape in (12, 13):
            space = Strip(shape, order=1.5, lateral=2.0, length_scale=0.3)
            measure = space.sobolev_measure(1.5)
            field = space.interior_values(space.sobolev_pointwise_variance(1.5))[:, 0]
            levels = [0, 7, space.interior_depths.size - 1]
            depths = space.interior_depths[levels]
            half = 0.5 * (space.x[1] - space.x[0])
            points = [(depth, space.x[3] + half) for depth in depths]
            got = np.diag(point_covariance(space, measure, points))
            if shape % 2:
                assert np.allclose(got, field[levels], rtol=1e-12)
                continue
            component = space.transform.component_of((shape // 2,))
            block = space.basis.block(component)
            modes = space.basis.basis_of(component).evaluate(depths)
            spectrum = space.eigenvalues[block] ** -1.5 / space.metric_values[block]
            share = (modes**2) @ spectrum / space.transform.volume
            assert np.all(share > 1e-7 * field[levels])
            assert np.allclose(field[levels] - got, share, rtol=1e-8)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_pointwise_std_is_what_the_covariance_says(self, build):
        """``diag(E C E*)`` through the operator algebra, against a smooth
        field asked for on the padding too: what the covariance sees is the
        scaled draw's projection onto the kept modes."""
        space = build(2.0, depth_modes=30, element_length=0.1)
        target = space.project_function(
            lambda p: 0.3 + 0.1 * p[0] ** 2, extension="constant"
        )
        measure = space.sobolev_measure(2.5, pointwise_std=target)
        points, _ = grid_points(space)
        chosen = points[np.linspace(0, len(points) - 1, 9).astype(int)]
        deviation = np.sqrt(np.diag(point_covariance(space, measure, chosen)))
        assert np.allclose(deviation, 0.3 + 0.1 * chosen[:, 0] ** 2, rtol=1e-2)

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_prior_with_scales_of_its_own(self, build, rng):
        """Self-adjoint and positive in the space's metric, the precision
        the covariance's inverse, and with the space's own scales the
        default measure."""
        space = build(2.0)
        default = space.sobolev_measure(2.0)
        same = space.sobolev_measure(2.0, length_scale=space.length_scale)
        x = space.random(rng=rng)
        assert np.allclose(same.covariance(x), default.covariance(x), atol=1e-10)
        measure = space.sobolev_measure(
            2.0, length_scale=0.5, lateral_length_scale=0.25
        )
        check_operator(measure.covariance, rng=rng, measure=default)
        check_traits(measure.covariance, rng=rng, measure=default)
        assert np.allclose(measure.precision(measure.covariance(x)), x, atol=1e-8)
        longer = space.sobolev_measure(2.0, length_scale=0.5, lateral_length_scale=1.0)
        centre = [np.mean(space.bounds)] + [
            np.mean(bound) for bound in space.lateral_domain.bounds
        ]
        along = np.array(centre)
        along[1] += 0.3
        correlations = []
        for prior in (measure, longer):
            matrix = point_covariance(space, prior, [centre, along])
            correlations.append(matrix[0, 1] / np.sqrt(matrix[0, 0] * matrix[1, 1]))
        assert correlations[1] > correlations[0] + 0.05

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_sample_with_a_fixed_seed_is_reproducible(self, build, rng):
        space = build(2.0)
        for options in ({}, dict(pointwise_std=0.3), dict(length_scale=0.5)):
            measure = space.sobolev_measure(2.0, **options)
            first = measure.sample(rng=np.random.default_rng(7))
            again = measure.sample(rng=np.random.default_rng(7))
            assert np.array_equal(first, again)
            assert not np.array_equal(first, measure.sample(rng=rng))

    @pytest.mark.parametrize("build", [strip, bounded_slab])
    def test_the_moments_match_the_samples(self, build, rng):
        space = build(2.0, depth_modes=3)
        for options in (dict(amplitude=0.7), dict(pointwise_std=0.3)):
            check_measure(
                space.sobolev_measure(2.0, **options), rng=rng, samples=2000, rtol=0.2
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("build", [strip, bounded_slab])
    def test_the_sample_covariance_approaches_the_exact_one(self, build):
        """Between points of the domain, for a calibrated prior with scales
        of its own: every entry within five of its own standard errors,
        ``sqrt((C_ii C_jj + C_ij^2) / n)`` for a Gaussian, and the largest
        error halving when the samples are four times as many."""
        rng = np.random.default_rng(2026)
        space = build(2.0)
        measure = space.sobolev_measure(
            2.0, length_scale=0.5, lateral_length_scale=0.3, pointwise_std=0.6
        )
        points = [space.random_point(rng=rng) for _ in range(6)]
        look = space.point_evaluation_operator(points)
        exact = point_covariance(space, measure, points)
        draws = np.array([look(measure.sample(rng=rng)) for _ in range(40_000)])
        errors = []
        for count in (2_500, 10_000, 40_000):
            sample = draws[:count].T @ draws[:count] / count
            spread = np.sqrt(
                (np.outer(np.diag(exact), np.diag(exact)) + exact**2) / count
            )
            assert np.all(np.abs(sample - exact) < 5.0 * spread)
            errors.append(np.abs(sample - exact).max() / np.abs(exact).max())
        assert errors[2] < 0.02
        assert errors[2] < 0.7 * errors[0]


class TestWhatAFunctionSpaceRefuses:
    """As on the ball: what does not survive refinement is refused, with
    ``unsafe=True`` for a test that means it."""

    def test_point_evaluation_at_and_below_half_the_dimension(self):
        for build, threshold, point in (
            (strip, 1.0, (0.5, 0.5)),
            (slab, 1.5, (0.5, 0.5, 0.5)),
        ):
            assert build(0.0).point_evaluation_order() == threshold
            for order in (0.0, threshold):
                space = build(order)
                with pytest.raises(ValueError, match=f"above {threshold:g}"):
                    space.dirac(point)
                for route in ("auto", "dense", "local", "nufft"):
                    with pytest.raises(ValueError, match=f"above {threshold:g}"):
                        space.point_evaluation_operator([point], route=route)
            build(0.0).dirac(point, unsafe=True)
            build(0.0).point_evaluation_operator([point], route="nufft", unsafe=True)
            build(threshold + 0.25).dirac(point)

    def test_a_measure_that_is_not_trace_class(self):
        for build, threshold in ((strip, 1.0), (slab, 1.5)):
            space = build(2.0)
            with pytest.raises(ValueError, match="trace class"):
                space.sobolev_measure(threshold)
            with pytest.raises(ValueError, match="trace class"):
                space.sobolev_measure(
                    threshold, length_scale=0.3, lateral_length_scale=0.6
                )
            space.sobolev_measure(threshold, unsafe=True)
            space.sobolev_measure(threshold + 0.25)

    @pytest.mark.parametrize("build", [strip, slab])
    def test_an_identity_covariance(self, build):
        space = build(0.0)
        with pytest.raises(ValueError, match="identity"):
            space.spectral_measure(np.ones(space.dim))
        with pytest.raises(ValueError, match="identity"):
            GaussianMeasure(space, covariance=LinearOperator.identity(space))
        with pytest.raises(ValueError):
            space.sobolev_measure(0.0)
        space.spectral_measure(np.ones(space.dim), unsafe=True)
        GaussianMeasure(space, covariance=LinearOperator.identity(space), unsafe=True)

    @pytest.mark.parametrize("build", [strip, slab])
    def test_a_density_against_the_discretizations_measure(self, build, rng):
        space = build(2.0)
        assert space.is_function_space
        measure = space.sobolev_measure(2.0)
        x = measure.sample(rng=rng)
        with pytest.raises(ValueError, match="unsafe"):
            measure.log_density(x)
        assert np.isfinite(measure.log_density(x, unsafe=True))

    def test_what_a_prior_refuses(self):
        space = strip(2.0)
        with pytest.raises(ValueError, match="positive"):
            space.sobolev_measure(2.5, length_scale=-0.3)
        with pytest.raises(ValueError, match="positive"):
            space.sobolev_measure(2.5, lateral_length_scale=lambda z: 0.3 - z)
        with pytest.raises(ValueError, match="positive"):
            space.sobolev_measure(2.0, pointwise_std=-1.0)
        assert space.sobolev_measure(2.0).precision is not None


# ------------------------------------------------------------------------- #
#                                 Plotting                                  #
# ------------------------------------------------------------------------- #


def drawn(mesh, /):
    """The coordinates and values of a gouraud mesh: ``(rows, columns, 2)``
    and ``(rows, columns)``."""
    where = np.asarray(mesh.get_coordinates())
    return where, np.asarray(mesh.get_array()).reshape(where.shape[:2])


def grid_index(space, axis, coordinates, /):
    """Which point of the lateral grid along an axis each coordinate is, one
    period on read as the period's first point."""
    grid = space.lateral_axes[axis]
    period = space.lateral_domain.lengths[axis]
    offset = (np.asarray(coordinates)[:, None] - grid[None, :]) / period
    gaps = np.abs(offset - np.round(offset))
    index = np.argmin(gaps, axis=1)
    # Every coordinate is a point of the grid, or one a period on.
    assert np.max(gaps[np.arange(index.size), index]) < 1e-12
    return index


def bound_columns(space, axis, coordinates, /):
    """Which of the coordinates drawn along a lateral axis are the bounds of
    a bounded domain, and not points of the grid. No bound of these
    fixtures is a point of the grid, so along a bounded axis they are the
    first coordinate and the last, and they are the bounds."""
    coordinates = np.asarray(coordinates)
    edges = np.zeros(coordinates.size, dtype=bool)
    if space.lateral_domain.padding[axis] > 0.0:
        bounds = space.lateral_domain.bounds[axis]
        assert tuple(coordinates[[0, -1]]) == tuple(bounds)
        gaps = np.abs(space.lateral_axes[axis][:, None] - np.asarray(bounds)[None, :])
        assert np.min(gaps) > 1e-3
        edges[[0, -1]] = True
    return edges


def interior_rows(space, /):
    """The rows of a grid array that are the domain's nodes in depth."""
    lateral = tuple(range(1, space.interior_mask.ndim))
    return np.flatnonzero(space.interior_mask.any(axis=lateral))


class TestPlotting:
    """The pictures are checked through their numbers: the values handed to
    matplotlib against the vector's own grid values and the space's own
    point evaluation, and the axes against the domain, depth downward."""

    @pytest.fixture(autouse=True)
    def _headless(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        yield
        pyplot.close("all")

    @pytest.mark.parametrize("build", [strip, bounded_strip])
    def test_a_strip_is_its_grid_with_depth_downward(self, build, rng):
        """The values drawn are the vector's on the domain's part of the grid,
        read back at the coordinates drawn, and the field the space
        evaluates there to rounding: 4e-15 relative at most measured. The
        columns at the bounds of a bounded domain are evaluated, and are the
        field there too."""
        from pygeoinf import plotting

        space = build(2.0)
        x = space.random(rng=rng)
        ax, mesh = plotting.plot(space, x, symmetric=True)
        where, values = drawn(mesh)
        depths, across = where[:, 0, 1], where[0, :, 0]
        assert np.array_equal(depths, space.interior_depths)
        assert np.array_equal(
            where[..., 1], np.broadcast_to(depths[:, None], values.shape)
        )
        # Shallow at the top: the depth axis runs downward, over the domain.
        assert ax.yaxis_inverted()
        assert ax.get_ylim() == (space.bottom, space.top)
        assert ax.get_ylabel() == "depth" and ax.get_xlabel() == "x"
        assert ax.get_aspect() == 1.0
        on = ~bound_columns(space, 0, across)
        columns = grid_index(space, 0, across[on])
        assert np.array_equal(values[:, on], x[np.ix_(interior_rows(space), columns)])
        points = [(depth, a) for depth in depths for a in across]
        evaluated = space.evaluate(x, points).reshape(values.shape)
        scale = np.max(np.abs(evaluated))
        assert np.max(np.abs(values - evaluated)) <= 1e-12 * scale
        low, high = mesh.get_clim()
        assert low == pytest.approx(-high)
        # plot on a strip is its section, whole.
        _, same = plotting.plot_section(space, x, symmetric=True)
        assert np.array_equal(drawn(same)[0], where)
        assert np.array_equal(drawn(same)[1], values)

    def test_a_periodic_strip_spans_its_period(self, rng):
        space = strip(2.0)
        from pygeoinf import plotting

        x = space.random(rng=rng)
        ax, mesh = plotting.plot(space, x)
        where, values = drawn(mesh)
        assert np.allclose(where[0, :, 0], np.append(space.x, 2.0))
        assert np.array_equal(values[:, -1], values[:, 0])
        assert ax.get_xlim() == (0.0, 2.0)

    def test_a_bounded_strip_fills_its_bounds(self, rng):
        """Neither bound is a point of the grid: a column at each is
        evaluated, the projection onto the kept modes, and the rest are the
        grid values as they stand. Shown on an array off the span of the
        kept modes, where the two differ."""
        from pygeoinf import plotting

        space = bounded_strip(2.0)
        x = rng.standard_normal(space.grid_shape)
        ax, mesh = plotting.plot(space, x)
        where, values = drawn(mesh)
        depths, across = where[:, 0, 1], where[0, :, 0]
        inside = np.flatnonzero(space.transform.mask)
        assert np.array_equal(across, np.concatenate([[0.5], space.x[inside], [2.0]]))
        assert ax.get_xlim() == (0.5, 2.0)
        rows = interior_rows(space)
        assert np.array_equal(values[:, 1:-1], x[np.ix_(rows, inside)])
        points = [(depth, bound) for depth in depths for bound in (0.5, 2.0)]
        evaluated = space.evaluate(x, points).reshape(depths.size, 2)
        scale = np.max(np.abs(evaluated))
        assert np.max(np.abs(values[:, [0, -1]] - evaluated)) <= 1e-12 * scale
        # The projection is not the grid's values: at the nearest points of
        # the grid it differs from them by much more than rounding.
        nearest = x[np.ix_(rows, inside[[0, -1]])]
        assert np.max(np.abs(values[:, [0, -1]] - nearest)) > 0.1

    def test_a_bound_on_the_grid_adds_no_column(self, rng):
        """Bounds that are points of the grid, the upper one to rounding: the
        picture of a strip is its grid values alone, and a slab's map is
        evaluated at the grid's points alone."""
        from pygeoinf import plotting

        domain = PaddedDomain([(0.3, 1.0)], padding=0.1)
        space = bounded_strip(2.0, shape=9, lateral=domain)
        inside = np.flatnonzero(space.transform.mask)
        # The upper bound is a point of the grid to rounding, not exactly.
        assert space.x[inside[0]] == 0.3 and space.x[inside[-1]] != 1.0
        x = rng.standard_normal(space.grid_shape)
        ax, mesh = plotting.plot(space, x)
        where, values = drawn(mesh)
        assert np.array_equal(where[0, :, 0], space.x[inside])
        assert np.array_equal(values, x[np.ix_(interior_rows(space), inside)])
        assert ax.get_xlim() == (0.3, 1.0)
        box = bounded_slab(
            2.0,
            shape=(9, 5),
            lateral=PaddedDomain([(0.3, 1.0), (0.0, 1.0)], padding=(0.1, 0.0)),
        )
        _, mesh = plotting.plot(box, box.random(rng=rng))
        where, _ = drawn(mesh)
        along_x = np.flatnonzero(box.transform.mask.any(axis=1))
        assert np.array_equal(where[:, 0, 0], box.x[along_x])

    def test_the_padding_is_drawn_on_request(self, rng):
        """The whole grid, closed across, with the domain outlined: the top
        and the base across, the lateral bounds down."""
        from pygeoinf import plotting

        space = bounded_strip(2.0)
        x = space.random(rng=rng)
        ax, mesh = plotting.plot(space, x, padding=True)
        where, values = drawn(mesh)
        assert np.array_equal(where[:, 0, 1], space.depths)
        assert ax.get_ylim() == (space.depths[-1], space.depths[0])
        period = space.lateral_domain.lengths[0]
        assert np.allclose(where[0, :, 0], np.append(space.x, space.x[0] + period))
        assert np.array_equal(values[:, :-1], x)
        assert np.array_equal(values[:, -1], x[:, 0])
        across = {
            line.get_ydata()[0]
            for line in ax.lines
            if line.get_ydata()[0] == line.get_ydata()[1]
        }
        down = {
            line.get_xdata()[0]
            for line in ax.lines
            if line.get_xdata()[0] == line.get_xdata()[1]
        }
        assert across == {0.2, 1.0}
        assert down == {0.5, 2.0}

    def test_a_vertical_exaggeration(self, rng):
        from pygeoinf import plotting

        space = strip(2.0)
        ax, _ = plotting.plot(space, space.random(rng=rng), aspect=5.0, title="t")
        assert ax.get_aspect() == 5.0
        assert ax.get_title() == "t"

    @pytest.mark.parametrize("build", [slab, bounded_slab])
    def test_plot_on_a_slab_is_a_map_at_its_top(self, build, rng):
        """At the top, a node of the grid in depth, the evaluated map is the
        grid's top level, a second route to the same numbers; at the bounds
        of a bounded domain, the field evaluated there."""
        from pygeoinf import plotting

        space = build(2.0)
        x = space.random(rng=rng)
        ax, mesh = plotting.plot(space, x, colorbar_label="field")
        where, values = drawn(mesh)
        xs, ys = where[:, 0, 0], where[0, :, 1]
        # x across and y up.
        assert np.array_equal(where[..., 0], np.broadcast_to(xs[:, None], values.shape))
        assert np.array_equal(where[..., 1], np.broadcast_to(ys[None, :], values.shape))
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("x", "y")
        assert ax.get_aspect() == 1.0
        edges = [bound_columns(space, 0, xs), bound_columns(space, 1, ys)]
        on = np.ix_(~edges[0], ~edges[1])
        top = x[interior_rows(space)[0]]
        expected = top[
            np.ix_(
                grid_index(space, 0, xs[~edges[0]]), grid_index(space, 1, ys[~edges[1]])
            )
        ]
        assert np.max(np.abs(values[on] - expected)) <= 1e-12 * np.max(np.abs(expected))
        rim = edges[0][:, None] | edges[1][None, :]
        assert rim.any() == (build is bounded_slab)
        if rim.any():
            points = [(space.top, a, b) for a, b in where[rim]]
            evaluated = space.evaluate(x, points)
            scale = np.max(np.abs(evaluated))
            assert np.max(np.abs(values[rim] - evaluated)) <= 1e-12 * scale
        for axis, limits in enumerate((ax.get_xlim(), ax.get_ylim())):
            low, high = space.lateral_domain.bounds[axis]
            if space.lateral_domain.padding[axis] == 0.0:
                high = low + space.lateral_domain.lengths[axis]
            assert limits == (low, high)

    def test_a_map_at_any_depth_is_the_field_there(self, rng):
        from pygeoinf import plotting

        space = bounded_slab(2.0)
        x = space.random(rng=rng)
        node = interior_rows(space)[3]
        _, mesh = plotting.plot(space, x, depth=space.depths[node])
        where, values = drawn(mesh)
        # Bounded along x: the first and last columns are at the bounds.
        on = ~bound_columns(space, 0, where[:, 0, 0])
        level = x[node]
        expected = level[
            np.ix_(
                grid_index(space, 0, where[on, 0, 0]),
                grid_index(space, 1, where[0, :, 1]),
            )
        ]
        assert np.max(np.abs(values[on] - expected)) <= 1e-12 * np.max(np.abs(expected))
        _, mesh = plotting.plot(space, x, depth=0.37)
        where, values = drawn(mesh)
        points = [(0.37, *p) for p in where.reshape(-1, 2)]
        assert np.allclose(
            values.ravel(), space.evaluate(x, points), rtol=0, atol=1e-13
        )
        with pytest.raises(ValueError, match="domain"):
            plotting.plot(space, x, depth=1.5)

    @pytest.mark.parametrize("build", [slab, bounded_slab])
    def test_a_slab_section_on_a_grid_line_is_the_grid_there(self, build, rng):
        """Along x at a y of the grid and along y at an x of it: evaluated,
        and the vector's values on that line of the grid to rounding; at the
        bounds of a bounded domain, the field evaluated there."""
        from pygeoinf import plotting

        space = build(2.0)
        x = space.random(rng=rng)
        inside = [np.flatnonzero(space.transform.mask.any(axis=1 - a)) for a in (0, 1)]
        rows = interior_rows(space)
        j, i = inside[1][1], inside[0][1]
        for name, at, along in (("y", space.y[j], 0), ("x", space.x[i], 1)):
            ax, mesh = plotting.plot_section(space, x, **{name: at})
            where, values = drawn(mesh)
            depths, across = where[:, 0, 1], where[0, :, 0]
            assert np.array_equal(depths, space.interior_depths)
            assert ax.yaxis_inverted() and ax.get_ylim() == (space.bottom, space.top)
            assert ax.get_xlabel() == ("x", "y")[along]
            edges = bound_columns(space, along, across)
            columns = grid_index(space, along, across[~edges])
            plane = x[:, :, j] if along == 0 else x[:, i, :]
            expected = plane[np.ix_(rows, columns)]
            scale = np.max(np.abs(expected))
            assert np.max(np.abs(values[:, ~edges] - expected)) <= 1e-12 * scale
            if edges.any():
                lateral = [(a, at) if along == 0 else (at, a) for a in across[edges]]
                points = [(depth, *p) for depth in depths for p in lateral]
                evaluated = space.evaluate(x, points).reshape(depths.size, -1)
                scale = np.max(np.abs(evaluated))
                assert np.max(np.abs(values[:, edges] - evaluated)) <= 1e-12 * scale

    def test_a_section_with_no_plane_named_is_through_the_middle(self, rng):
        from pygeoinf import plotting

        space = bounded_slab(2.0)
        x = space.random(rng=rng)
        _, mesh = plotting.plot_section(space, x)
        where, values = drawn(mesh)
        middle = -0.25  # of the bounds in y, (-1, 0.5)
        points = [(d, a, middle) for d, a in where.reshape(-1, 2)[:, ::-1]]
        assert np.allclose(
            values.ravel(), space.evaluate(x, points), rtol=0, atol=1e-13
        )

    @pytest.mark.parametrize("build", GEOMETRIES)
    def test_a_profile_is_the_field_beneath_a_position(self, build, rng):
        from pygeoinf import plotting

        space = build(2.0)
        x = space.random(rng=rng)
        position = space.random_point(rng=rng)[1:]
        ax, line = plotting.plot_profile(space, x, *position, points=17)
        depths = np.linspace(space.top, space.bottom, 17)
        assert np.allclose(line.get_ydata(), depths)
        points = [(depth, *position) for depth in depths]
        assert np.allclose(line.get_xdata(), space.evaluate(x, points))
        assert ax.get_ylim() == (space.bottom, space.top)

    def test_points_go_where_the_picture_puts_them(self):
        """Depth down in a strip, and the depth set aside on a slab's map;
        reduced into the period across a periodic axis and not across a
        bounded one."""
        from pygeoinf import plotting

        ax, collection = plotting.plot_points(
            strip(2.0), [(0.0, 0.5), (0.3, 2.5)], data=[1.0, 2.0], colorbar_label="v"
        )
        assert np.allclose(collection.get_offsets(), [[0.5, 0.0], [0.5, 0.3]])
        assert ax.yaxis_inverted()
        _, collection = plotting.plot_points(slab(2.0), [(0.4, -0.5, 2.0)])
        assert np.allclose(collection.get_offsets(), [[1.5, 0.5]])
        _, collection = plotting.plot_points(bounded_slab(2.0), [(0.4, 1.4, 0.7)])
        assert np.allclose(collection.get_offsets(), [[1.4, -0.8]])
        with pytest.raises(ValueError, match="values"):
            plotting.plot_points(strip(2.0), [(0.0, 0.5)], data=[1.0, 2.0])

    def test_subplots_are_plain_axes(self):
        from pygeoinf import plotting

        figure, axes = plotting.subplots(strip(2.0), columns=2)
        assert len(axes) == 2 and not hasattr(axes[0], "projection")
        assert tuple(figure.get_size_inches()) == (12.0, 3.2)

    def test_what_is_not_drawn_is_refused(self, rng):
        from pygeoinf import plotting

        flat, box = strip(2.0), slab(2.0)
        with pytest.raises(ValueError, match="shape"):
            plotting.plot(flat, np.zeros(3))
        with pytest.raises(ValueError, match="slab"):
            plotting.plot_section(flat, flat.random(rng=rng), x=0.5)
        x = box.random(rng=rng)
        with pytest.raises(ValueError, match="one of"):
            plotting.plot_section(box, x, x=0.5, y=0.5)
        with pytest.raises(ValueError, match="padding"):
            plotting.plot_section(box, x, padding=True)
        with pytest.raises(ValueError, match="domain"):
            plotting.plot_section(bounded_slab(2.0), bounded_slab(2.0).zero(), x=1.5)
        with pytest.raises(ValueError, match="x, y"):
            plotting.plot_profile(box, x, 0.5)
        # A map at one depth is of a slab: on a strip, plot refuses one.
        with pytest.raises(TypeError, match="line"):
            plotting.plot(flat, flat.random(rng=rng), depth=0.5)
        with pytest.raises(TypeError, match="line"):
            plotting.plot(flat, flat.random(rng=rng), side="above")
        # A disk has no renderer of a section or a profile.
        with pytest.raises(NotImplementedError, match="plot_section .* Lebesgue"):
            plotting.plot_section(disk.Lebesgue(4, length_scale=0.3), np.zeros(3))
        with pytest.raises(NotImplementedError, match="plot_profile .* Lebesgue"):
            plotting.plot_profile(disk.Lebesgue(4, length_scale=0.3), np.zeros(3), 0.5)
