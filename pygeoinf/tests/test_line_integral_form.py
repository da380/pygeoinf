import numpy as np

from pygeoinf.line_integral_form import LineIntegralForm
from pygeoinf.hilbert_space import HilbertSpace
from pygeoinf.linear_forms import LinearForm


class SimpleFunctionWrapper:
    """A tiny callable wrapper representing f(x,y,z) = c0 + c1 * z."""

    def __init__(self, components: np.ndarray):
        self._c = np.asarray(components, dtype=float)

    @property
    def components(self) -> np.ndarray:
        return self._c

    def __call__(self, point):
        # point is (x,y,z)
        return float(self._c[0] + self._c[1] * point[2])

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return np.asarray([self(p) for p in points])


class SimpleFunctionSpace(HilbertSpace):
    """A tiny Hilbert space with two basis functions: 1 and z."""

    def __init__(self):
        self._dim = 2

    @property
    def dim(self) -> int:
        return self._dim

    def to_dual(self, x):
        # represent dual simply via LinearForm using components
        return LinearForm(self, components=self.to_components(x))

    def from_dual(self, xp):
        return self.from_components(xp.components)

    def to_components(self, x) -> np.ndarray:
        return np.asarray(x.components)

    def from_components(self, c: np.ndarray):
        return SimpleFunctionWrapper(c)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SimpleFunctionSpace) and self.dim == other.dim


def test_line_integral_constant_and_z():
    # path along z from 0 to 1
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    domain = SimpleFunctionSpace()

    form = LineIntegralForm(domain, path)

    # Components should be integrals of basis functions: int 1 ds = 1, int z ds = 1/2
    assert np.allclose(form.components, np.array([1.0, 0.5]), atol=1e-12)

    # Evaluate on f = 2 + 3*z -> integral = 2*1 + 3*0.5 = 3.5
    vec = domain.from_components(np.array([2.0, 3.0]))
    assert abs(form(vec) - 3.5) < 1e-12
