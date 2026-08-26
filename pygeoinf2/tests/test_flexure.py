"""
Pointwise field algebra, and the flexure operator built on it.

The variable-coefficient flexure operator is the first thing in the package
whose correctness cannot be read off a spectrum: with a constant rigidity every
term that makes it interesting vanishes identically. So every test here
compares it against something computed another way, and the ones that matter
carry a negative control showing what a wrong answer would look like.

See DESIGN.md sections 20.5 (F) and 21.
"""

import numpy as np
import pytest

from pygeoinf2.algebra.spaces import require_module
from pygeoinf2.symmetric_space import Lebesgue as BoxLebesgue
from pygeoinf2.testing import check_operator
from pygeoinf2.traits import Traits

pyshtools = pytest.importorskip("pyshtools")

from pygeoinf2.symmetric_space.sphere import Lebesgue, Sobolev  # noqa: E402


def spectral_second_derivative(field, length):
    """d^2/dx^2 on a periodic 1D grid, by FFT. Independent of the package."""
    size = field.size
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(size, d=length / size)
    return np.fft.irfft(-(wavenumbers**2) * np.fft.rfft(field), n=size)


class TestPointwiseAlgebra:
    def test_multiply_and_sqrt_are_pointwise(self, rng):
        X = BoxLebesgue((32,), lengths=(1.0,))
        x = X.project_function(lambda t: 2.0 + np.sin(2.0 * np.pi * t))
        assert np.allclose(X.multiply(x, x), x**2)
        assert np.allclose(X.sqrt(X.multiply(x, x)), np.abs(x))

    def test_multiplication_is_self_adjoint_on_a_lebesgue_space(self, rng):
        """True because the inner product there *is* the L2 one."""
        X = Lebesgue(12)
        f = X.project_function(lambda p: 1.5 + np.cos(p[0]))
        A = X.multiplication_operator(f)
        check_operator(A, rng=rng)
        assert Traits.SELF_ADJOINT & A.traits

    def test_multiplication_is_not_self_adjoint_on_a_sobolev_space(self, rng):
        """The negative control for the lift: a Sobolev space weights its modes.

        v1 gets this right by lifting the formal adjoint; the trap is claiming
        self-adjointness because it holds in L2 and the operator looks the same.
        """
        X = Sobolev(12, 2.0, 0.2)
        f = X.project_function(lambda p: 1.5 + np.cos(p[0]))
        A = X.multiplication_operator(f)
        check_operator(A, rng=rng)
        assert not (Traits.SELF_ADJOINT & A.traits)

        u, v = X.random(rng=rng), X.random(rng=rng)
        forward = X.inner_product(A(u), v)
        backward = X.inner_product(u, A(v))
        assert not np.isclose(forward, backward, rtol=1e-6)

    def test_the_lift_changes_the_adjoint_and_not_the_action(self, rng):
        X = Sobolev(12, 2.0, 0.2)
        f = X.project_function(lambda p: 1.5 + np.cos(p[0]))
        u = X.random(rng=rng)
        assert np.allclose(X.multiplication_operator(f)(u), X.multiply(f, u))

    def test_a_product_is_truncated_back_into_the_space(self, rng):
        """Otherwise a product is not a well-defined function of its factors.

        The Driscoll-Healy grid is oversampled, so a raw grid product is one of
        many arrays with those components. Truncating picks the one in the span
        of the basis, which is what makes the formal-adjoint lift agree with a
        direct application.
        """
        X = Lebesgue(12)
        f = X.project_function(lambda p: 1.5 + np.cos(p[0]))
        u = X.random(rng=rng)
        raw = np.asarray(f) * np.asarray(u)
        truncated = X.multiply(f, u)
        assert not np.allclose(raw, truncated)
        assert np.allclose(X.to_components(raw), X.to_components(truncated))
        assert np.allclose(truncated, X.truncate(truncated))

    def test_truncation_is_free_on_a_box(self, rng):
        """One component per grid point, so there is nothing to remove."""
        X = BoxLebesgue((32, 16), lengths=(1.0, 2.0))
        x = X.random(rng=rng)
        assert X.truncate(x) is not None
        assert np.allclose(X.truncate(x), x)
        assert X.dim == 32 * 16

    def test_require_module_names_the_capability(self):
        from pygeoinf2.algebra.spaces import EuclideanSpace

        require_module(Lebesgue(4))
        with pytest.raises(TypeError, match="pointwise multiplication"):
            require_module(EuclideanSpace(3))


class TestGradientDotProduct:
    def test_against_an_analytic_case(self):
        """grad(sin) . grad(cos) on a circle, where the answer is known."""
        X = BoxLebesgue((256,), lengths=(2.0 * np.pi,))
        t = X.grid_axes[0]
        f = X.project_function(np.sin)
        g = X.project_function(np.cos)
        exact = np.cos(t) * (-np.sin(t))
        got = X.gradient_dot_product(f, g)
        assert np.allclose(got, exact, atol=1e-9)

    def test_the_sign_is_not_free(self):
        """The negative control. v1 returns exactly minus this.

        Recorded because it is a real defect there, and because it is invisible
        in every constant-coefficient test: the gradient terms only appear
        multiplied by the gradient of a coefficient.
        """
        X = BoxLebesgue((256,), lengths=(2.0 * np.pi,))
        t = X.grid_axes[0]
        f, g = X.project_function(np.sin), X.project_function(np.cos)
        exact = np.cos(t) * (-np.sin(t))
        assert not np.allclose(-X.gradient_dot_product(f, g), exact, atol=1e-6)

    def test_it_is_symmetric(self, rng):
        X = BoxLebesgue((64,), lengths=(1.0,))
        f, g = X.random(rng=rng), X.random(rng=rng)
        assert np.allclose(X.gradient_dot_product(f, g), X.gradient_dot_product(g, f))

    def test_the_gradient_of_a_constant_vanishes(self, rng):
        X = BoxLebesgue((64,), lengths=(1.0,))
        one = X.project_function(lambda t: 1.0)
        g = X.random(rng=rng)
        assert np.allclose(X.gradient_dot_product(one, g), 0.0, atol=1e-9)


class TestBochnerIdentity:
    """The block that produces the Hessian trace and the curvature commutator.

    Checked on the sphere against a closed form: a degree-one harmonic is the
    restriction of a linear function, so ``Hess f == -f g_ab`` and therefore
    ``tr(Hess f Hess g) == 2 f g`` exactly.
    """

    @staticmethod
    def _hessian_trace(space, f, g):
        laplacian = space.laplacian
        products = space.gradient_dot_product
        block = -0.5 * laplacian(products(f, g))
        block = block + 0.5 * (products(f, laplacian(g)) + products(g, laplacian(f)))
        return block - space.gaussian_curvature * products(f, g)

    def test_against_the_closed_form_for_degree_one(self):
        X = Lebesgue(24)
        f = X.project_function(lambda p: np.cos(p[0]))
        g = X.project_function(lambda p: np.sin(p[0]) * np.cos(p[1]))
        exact = 2.0 * f * g
        got = self._hessian_trace(X, f, g)
        assert np.allclose(got, exact, atol=1e-8 * np.abs(exact).max())

    @pytest.mark.parametrize("factor", [0.0, 2.0, -1.0])
    def test_the_curvature_coefficient_is_pinned(self, factor):
        """The negative control. Self-adjointness does *not* pin this term."""
        X = Lebesgue(24)
        f = X.project_function(lambda p: np.cos(p[0]))
        g = X.project_function(lambda p: np.sin(p[0]) * np.cos(p[1]))
        laplacian, products = X.laplacian, X.gradient_dot_product
        block = -0.5 * laplacian(products(f, g))
        block = block + 0.5 * (products(f, laplacian(g)) + products(g, laplacian(f)))
        wrong = block - factor * X.gaussian_curvature * products(f, g)
        exact = 2.0 * f * g
        assert not np.allclose(wrong, exact, atol=1e-3 * np.abs(exact).max())


class TestFlexureAgainstIndependentComputations:
    def test_the_one_dimensional_beam(self):
        """In one flat dimension the operator is exactly ``(D w'')'' + rho w``.

        The two middle terms of the general formula cancel identically there,
        which is what makes this a clean check of everything else — including
        the gradient block, which does *not* cancel.
        """
        size, length = 512, 1.0
        X = BoxLebesgue((size,), lengths=(length,))
        t = X.grid_axes[0]
        rigidity = 1.0 + 0.5 * np.sin(2 * np.pi * t) + 0.2 * np.cos(6 * np.pi * t)
        buoyancy = 3.0
        w = np.sin(4 * np.pi * t) + 0.3 * np.cos(2 * np.pi * t)

        exact = (
            spectral_second_derivative(
                rigidity * spectral_second_derivative(w, length), length
            )
            + buoyancy * w
        )
        got = X.flexural_operator(rigidity, 0.25, buoyancy)(w)
        assert np.allclose(got, exact, rtol=1e-7, atol=1e-7 * np.abs(exact).max())

    def test_two_flat_dimensions(self):
        """Where ``tr(Hess D Hess w)`` is genuinely a sum over components."""
        size, length = 64, 1.0
        X = BoxLebesgue((size, size), lengths=(length, length))
        axis_x, axis_y = X.grid_axes
        xx, yy = np.meshgrid(axis_x, axis_y, indexing="ij")
        frequencies = 2.0 * np.pi * np.fft.fftfreq(size, d=length / size)
        kx, ky = np.meshgrid(frequencies, frequencies, indexing="ij")
        wavevectors = [kx, ky]

        def second(field, first_axis, second_axis):
            symbol = -wavevectors[first_axis] * wavevectors[second_axis]
            return np.real(np.fft.ifft2(symbol * np.fft.fft2(field)))

        def positive_laplacian(field):
            return np.real(np.fft.ifft2((kx**2 + ky**2) * np.fft.fft2(field)))

        rigidity = 1.0 + 0.3 * np.sin(2 * np.pi * xx) + 0.2 * np.cos(4 * np.pi * yy)
        poisson, buoyancy = 0.25, 2.0
        effective = rigidity * (1.0 - poisson)
        w = np.sin(6 * np.pi * xx) * np.cos(4 * np.pi * yy) + 0.4 * np.sin(
            2 * np.pi * yy
        )

        hessian_trace = sum(
            second(effective, a, b) * second(w, a, b) for a in (0, 1) for b in (0, 1)
        )
        exact = (
            positive_laplacian(rigidity * positive_laplacian(w))
            - positive_laplacian(effective) * positive_laplacian(w)
            + hessian_trace
            + buoyancy * w
        )
        got = X.flexural_operator(rigidity, poisson, buoyancy)(w)
        assert np.allclose(got, exact, rtol=1e-9, atol=1e-9 * np.abs(exact).max())

    @pytest.mark.parametrize("radius", [1.0, 2.5])
    def test_constant_coefficients_match_the_spectral_symbol(self, radius, rng):
        """``D lambda^2 - K D_eff lambda + rho``, which is where K first shows."""
        X = Lebesgue(20, radius=radius)
        rigidity, poisson, buoyancy = 1.7, 0.25, 3.0
        curvature = X.gaussian_curvature
        effective = rigidity * (1.0 - poisson)
        symbol = X.invariant_operator(
            lambda eigenvalue: rigidity * eigenvalue**2
            - curvature * effective * eigenvalue
            + buoyancy
        )
        w = X.random(rng=rng)
        assert np.allclose(
            X.flexural_operator(rigidity, poisson, buoyancy)(w), symbol(w)
        )


class TestFlexureStructure:
    def test_it_is_self_adjoint_with_a_varying_rigidity(self, rng):
        X = Lebesgue(16)
        rigidity = X.project_function(lambda p: 1.0 + 0.5 * np.cos(p[0]))
        A = X.flexural_operator(rigidity, 0.25, 2.0)
        check_operator(A, rng=rng)
        assert Traits.SELF_ADJOINT & A.traits

    def test_it_is_positive_definite_for_physical_parameters(self):
        """Which is what lets the inverse claim it and reach for CG."""
        X = Lebesgue(12)
        rigidity = X.project_function(lambda p: 1.0 + 0.5 * np.cos(p[0]))
        matrix = X.flexural_operator(rigidity, 0.25, 3.0).matrix(form="galerkin")
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        assert eigenvalues.min() > 0.0
        # the constant mode sees only the restoring force
        assert eigenvalues.min() == pytest.approx(3.0)

    def test_the_sobolev_version_acts_the_same(self, rng):
        X = Sobolev(16, 2.0, 0.2)
        rigidity = X.project_function(lambda p: 1.0 + 0.5 * np.cos(p[0]))
        A = X.flexural_operator(rigidity, 0.25, 2.0)
        check_operator(A, rng=rng)
        base = X.with_order(0.0)
        w = X.random(rng=rng)
        assert np.allclose(A(w), base.flexural_operator(rigidity, 0.25, 2.0)(w))


class TestFlexureInverse:
    def test_constant_coefficients_invert_exactly(self, rng):
        X = Lebesgue(16)
        rigidity, poisson, buoyancy = 1.7, 0.25, 3.0
        A = X.flexural_operator(rigidity, poisson, buoyancy)
        inverse = X.inverse_flexural_operator(rigidity, poisson, buoyancy)
        w = X.random(rng=rng)
        assert np.allclose(A(inverse(w)), w, rtol=1e-10)

    def test_a_varying_rigidity_inverts_by_preconditioned_cg(self, rng):
        X = BoxLebesgue((128,), lengths=(1.0,))
        t = X.grid_axes[0]
        rigidity = 1.0 + 0.5 * np.sin(2 * np.pi * t)
        A = X.flexural_operator(rigidity, 0.25, 3.0)
        inverse = X.inverse_flexural_operator(
            rigidity, 0.25, 3.0, baseline_rigidity=1.0
        )
        w = X.random(rng=rng)
        residual = X.subtract(A(inverse(w)), w)
        assert X.norm(residual) < 1e-6 * X.norm(w)

    def test_a_direct_solver_is_refused(self):
        from pygeoinf2.numerics.solvers import CholeskySolver

        X = BoxLebesgue((32,), lengths=(1.0,))
        rigidity = X.project_function(lambda t: 1.0 + 0.5 * np.sin(2 * np.pi * t))
        with pytest.raises(TypeError, match="IterativeSolver"):
            X.inverse_flexural_operator(rigidity, 0.25, 3.0, solver=CholeskySolver())

    def test_an_explicit_preconditioner_is_kept(self):
        """``with_preconditioner`` must not overrule a caller who set one."""
        from pygeoinf2.numerics.solvers import CGSolver
        from pygeoinf2.algebra.operators import LinearOperator

        X = BoxLebesgue((16,), lengths=(1.0,))
        chosen = LinearOperator.identity(X)
        solver = CGSolver(preconditioner=chosen)
        assert solver.with_preconditioner(LinearOperator.zero(X)) is solver
