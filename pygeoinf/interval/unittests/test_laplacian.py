import numpy as np
import unittest

from pygeoinf.interval import (
    Lebesgue,
    IntervalDomain,
    BoundaryConditions,
    Laplacian,
    Function,
)


def _safe_eval(func: Function, xs: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(func.evaluate(xs))
    except Exception:
        # fall back to scalar eval
        return np.asarray([func(float(xi)) for xi in xs])


def _l2_and_max_error(
    f_num: Function,
    f_ref: Function,
    a: float,
    b: float,
    n: int = 2001,
):
    xs = np.linspace(a, b, n)
    y_num = _safe_eval(f_num, xs)
    y_ref = _safe_eval(f_ref, xs)
    diff = y_num - y_ref

    # Trapezoidal L2 on [a,b]
    dx = (b - a) / (n - 1)
    l2_sq = np.sum(diff[:-1] ** 2 + diff[1:] ** 2) * 0.5 * dx
    l2 = np.sqrt(l2_sq)

    # Reference norm for relative error (avoid zero-divide)
    ref_sq = np.sum(y_ref[:-1] ** 2 + y_ref[1:] ** 2) * 0.5 * dx
    ref_norm = np.sqrt(ref_sq)

    max_err = float(np.max(np.abs(diff)))
    rel_l2 = l2 / (ref_norm + 1e-12)
    return l2, max_err, rel_l2


class TestLaplacianFromNotebook(unittest.TestCase):
    def setUp(self):
        # Domain
        self.a, self.b = 0.0, 1.0
        L = self.b - self.a
        self.L = L

        # Spaces per BC family (match notebook)
        self.space_dirichlet = Lebesgue(
            256, IntervalDomain(self.a, self.b), basis='sine'
        )
        self.space_neumann = Lebesgue(
            256, IntervalDomain(self.a, self.b), basis='cosine'
        )
        self.space_periodic = Lebesgue(
            257, IntervalDomain(self.a, self.b), basis='fourier'
        )

        # Operators (match notebook selections)
        self.L_dirichlet = Laplacian(
            self.space_dirichlet, BoundaryConditions('dirichlet'),
            method='spectral', dofs=256
        )
        # Neumann used FD in the notebook
        self.L_neumann = Laplacian(
            self.space_neumann, BoundaryConditions('neumann'),
            method='fd', dofs=256
        )
        self.L_periodic = Laplacian(
            self.space_periodic, BoundaryConditions('periodic'),
            method='spectral', dofs=256
        )

        # --- Inputs (Dirichlet) ---
        self.D1 = Function(
            self.space_dirichlet,
            evaluate_callable=lambda x: (x - self.a) * (self.b - x),
            name='D1_bump'
        )
        self.D2 = Function(
            self.space_dirichlet,
            evaluate_callable=lambda x: np.sin(np.pi * (x - self.a) / L),
            name='D2_sin_pi'
        )
        self.D3 = Function(
            self.space_dirichlet,
            evaluate_callable=lambda x: np.sin(2*np.pi * (x - self.a) / L),
            name='D3_sin_2pi'
        )
        self.D4 = Function(
            self.space_dirichlet,
            evaluate_callable=lambda x: ((x-self.a)**2) * ((self.b-x)**2),
            name='D4_poly'
        )
        self.D5 = Function(
            self.space_dirichlet,
            evaluate_callable=(
                lambda x: 5*np.sin(np.pi*(x-self.a)/L)
                - 7*np.sin(3*np.pi*(x-self.a)/L)
            ),
            name='D5_mix'
        )

        # --- Inputs (Neumann) ---
        self.N1 = Function(
            self.space_neumann,
            evaluate_callable=lambda x: np.ones_like(x),
            name='N1_const'
        )
        self.N2 = Function(
            self.space_neumann,
            evaluate_callable=lambda x: np.cos(np.pi*(x-self.a)/L),
            name='N2_cos_pi'
        )
        self.N3 = Function(
            self.space_neumann,
            evaluate_callable=lambda x: np.cos(2*np.pi*(x-self.a)/L),
            name='N3_cos_2pi'
        )
        self.N4 = Function(
            self.space_neumann,
            evaluate_callable=lambda x: ((x-self.a)**2)*((x-self.b)**2) + 1.0,
            name='N4_poly_plus_1'
        )
        self.N5 = Function(
            self.space_neumann,
            evaluate_callable=(
                lambda x: 2.0
                + 3.0*np.cos(np.pi*(x-self.a)/L)
                - np.cos(3*np.pi*(x-self.a)/L)
            ),
            name='N5_mix'
        )

        # --- Inputs (Periodic) ---
        self.P1 = Function(
            self.space_periodic,
            evaluate_callable=lambda x: np.ones_like(x),
            name='P1_const'
        )
        self.P2 = Function(
            self.space_periodic,
            evaluate_callable=lambda x: np.sin(2*np.pi*(x-self.a)/L),
            name='P2_sin_2pi'
        )
        self.P3 = Function(
            self.space_periodic,
            evaluate_callable=lambda x: np.cos(2*np.pi*(x-self.a)/L),
            name='P3_cos_2pi'
        )
        self.P4 = Function(
            self.space_periodic,
            evaluate_callable=(
                lambda x: np.sin(4*np.pi*(x-self.a)/L)
                + 2*np.cos(6*np.pi*(x-self.a)/L)
            ),
            name='P4_combo'
        )
        # Family P5: sin and cos modes n=0..4
        self.P5_sin = {
            n: Function(
                self.space_periodic,
                evaluate_callable=(
                    lambda x, n=n: np.sin(2*np.pi*n*(x-self.a)/L)
                    if n > 0 else np.zeros_like(x)
                ),
                name=f'P5_sin_{n}'
            )
            for n in range(0, 5)
        }
        self.P5_cos = {
            n: Function(
                self.space_periodic,
                evaluate_callable=(
                    lambda x, n=n: np.cos(2*np.pi*n*(x-self.a)/L)
                ),
                name=f'P5_cos_{n}'
            )
            for n in range(0, 5)
        }

        # Numeric outputs
        self.DIRICHLET_INPUTS = [self.D1, self.D2, self.D3, self.D4, self.D5]
        self.NEUMANN_INPUTS = [self.N1, self.N2, self.N3, self.N4, self.N5]
        self.PERIODIC_INPUTS = [self.P1, self.P2, self.P3, self.P4] + \
            list(self.P5_sin.values()) + list(self.P5_cos.values())

        self.DIRICHLET_NUM = [
            self.L_dirichlet(u) for u in self.DIRICHLET_INPUTS
        ]
        self.NEUMANN_NUM = [self.L_neumann(u) for u in self.NEUMANN_INPUTS]
        self.PERIODIC_NUM = [self.L_periodic(u) for u in self.PERIODIC_INPUTS]

        # Names for clarity/threshold logic
        self.DIRICHLET_NAMES = ['D1', 'D2', 'D3', 'D4', 'D5']

        # Analytic expected outputs
        self.D1_L = Function(self.space_dirichlet,
                             evaluate_callable=lambda x: 2.0 * np.ones_like(x),
                             name='D1_L')
        self.D2_L = Function(
            self.space_dirichlet,
            evaluate_callable=(
                lambda x: (np.pi / L)**2 * np.sin(np.pi * (x - self.a) / L)
            ),
            name='D2_L'
        )
        self.D3_L = Function(
            self.space_dirichlet,
            evaluate_callable=(
                lambda x: (2 * np.pi / L)**2
                * np.sin(2 * np.pi * (x - self.a) / L)
            ),
            name='D3_L'
        )
        self.D4_L = Function(
            self.space_dirichlet,
            evaluate_callable=(
                lambda x: -2*(x-self.a)**2
                + 8*(x-self.a)*(self.b-x)
                - 2*(self.b-x)**2
            ),
            name='D4_L'
        )
        self.D5_L = Function(
            self.space_dirichlet,
            evaluate_callable=(
                lambda x: 5*(np.pi / L)**2 * np.sin(np.pi * (x - self.a) / L)
                - 7*(3 * np.pi / L)**2 * np.sin(3 * np.pi * (x - self.a) / L)
            ),
            name='D5_L'
        )
        self.DIRICHLET_EXP = [
            self.D1_L, self.D2_L, self.D3_L, self.D4_L, self.D5_L
        ]

        self.N1_L = Function(self.space_neumann,
                             evaluate_callable=lambda x: np.zeros_like(x),
                             name='N1_L')
        self.N2_L = Function(
            self.space_neumann,
            evaluate_callable=(
                lambda x: (np.pi / L)**2 * np.cos(np.pi * (x - self.a) / L)
            ),
            name='N2_L'
        )
        self.N3_L = Function(
            self.space_neumann,
            evaluate_callable=(
                lambda x: (2 * np.pi / L)**2
                * np.cos(2 * np.pi * (x - self.a) / L)
            ),
            name='N3_L'
        )
        self.N4_L = Function(
            self.space_neumann,
            evaluate_callable=(
                lambda x: -2*(x-self.a)**2
                - 8*(x-self.a)*(x-self.b)
                - 2*(x-self.b)**2
            ),
            name='N4_L'
        )
        self.N5_L = Function(
            self.space_neumann,
            evaluate_callable=(
                lambda x: 0.0
                + 3*(np.pi / L)**2 * np.cos(np.pi * (x - self.a) / L)
                - (3 * np.pi / L)**2 * np.cos(3 * np.pi * (x - self.a) / L)
            ),
            name='N5_L'
        )
        self.NEUMANN_EXP = [
            self.N1_L, self.N2_L, self.N3_L, self.N4_L, self.N5_L
        ]

        self.P1_L = Function(self.space_periodic,
                             evaluate_callable=lambda x: np.zeros_like(x),
                             name='P1_L')
        self.P2_L = Function(
            self.space_periodic,
            evaluate_callable=(
                lambda x: (2 * np.pi / L)**2
                * np.sin(2 * np.pi * (x - self.a) / L)
            ),
            name='P2_L'
        )
        self.P3_L = Function(
            self.space_periodic,
            evaluate_callable=(
                lambda x: (2 * np.pi / L)**2
                * np.cos(2 * np.pi * (x - self.a) / L)
            ),
            name='P3_L'
        )
        self.P4_L = Function(
            self.space_periodic,
            evaluate_callable=(
                lambda x: (4 * np.pi / L)**2
                * np.sin(4 * np.pi * (x - self.a) / L)
                + 2 * (6 * np.pi / L)**2
                * np.cos(6 * np.pi * (x - self.a) / L)
            ),
            name='P4_L'
        )
        self.P5_sin_L = {
            n: Function(
                self.space_periodic,
                evaluate_callable=(
                    lambda x, n=n: (2 * np.pi * n / L)**2
                    * np.sin(2 * np.pi * n * (x - self.a) / L)
                    if n > 0 else np.zeros_like(x)
                ),
                name=f'P5_sin_L_{n}'
            )
            for n in range(0, 5)
        }
        self.P5_cos_L = {
            n: Function(
                self.space_periodic,
                evaluate_callable=(
                    lambda x, n=n: (2 * np.pi * n / L)**2
                    * np.cos(2 * np.pi * n * (x - self.a) / L)
                    if n > 0 else np.zeros_like(x)
                ),
                name=f'P5_cos_L_{n}'
            )
            for n in range(0, 5)
        }
        self.PERIODIC_EXP = [self.P1_L, self.P2_L, self.P3_L, self.P4_L] + \
            list(self.P5_sin_L.values()) + list(self.P5_cos_L.values())

    # Test methods are generated below per input/output pair


def _make_dirichlet_test(idx: int):
    def test_method(self: TestLaplacianFromNotebook):
        num = self.DIRICHLET_NUM[idx]
        ref = self.DIRICHLET_EXP[idx]
        name = self.DIRICHLET_NAMES[idx]
        l2, _max, rel = _l2_and_max_error(num, ref, self.a, self.b)
        if name in ('D2', 'D3', 'D5'):
            self.assertLess(rel, 5e-5, f"Dirichlet rel L2 too high: {rel}")
        else:
            self.assertLess(rel, 1.2e-1, f"Dirichlet rel L2 too high: {rel}")
    return test_method


def _make_neumann_test(idx: int):
    def test_method(self: TestLaplacianFromNotebook):
        num = self.NEUMANN_NUM[idx]
        ref = self.NEUMANN_EXP[idx]
        l2, _max, rel = _l2_and_max_error(num, ref, self.a, self.b)
        if idx == 0:
            # constant mode: absolute L2
            self.assertLess(l2, 2e-2, f"Neumann constant L2 err: {l2}")
        else:
            self.assertLess(rel, 1e-2, f"Neumann rel L2 too high: {rel}")
    return test_method


def _make_periodic_test(idx: int):
    def test_method(self: TestLaplacianFromNotebook):
        num = self.PERIODIC_NUM[idx]
        ref = self.PERIODIC_EXP[idx]
        l2, _max, rel = _l2_and_max_error(num, ref, self.a, self.b)
        self.assertLess(rel, 5e-5, f"Periodic rel L2 too high: {rel}")
    return test_method


# Dynamically attach one test per pair
for i in range(5):
    setattr(
        TestLaplacianFromNotebook,
        f"test_dirichlet_{i}",
        _make_dirichlet_test(i),
    )

for i in range(5):
    setattr(
        TestLaplacianFromNotebook,
        f"test_neumann_{i}",
        _make_neumann_test(i),
    )

for i in range(4 + 5 + 5):
    setattr(
        TestLaplacianFromNotebook,
        f"test_periodic_{i}",
        _make_periodic_test(i),
    )
