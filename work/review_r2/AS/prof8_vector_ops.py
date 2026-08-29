from common import *
from pygeoinf2 import EuclideanSpace
for n in (2000, 200000, 2000000):
    X = EuclideanSpace(n)
    x = np.ones(n); y = np.ones(n)
    r = interleave({"X.add": lambda: X.add(x, y), "x+y": lambda: x + y, "X.axpy": lambda: X.axpy(0.5, x, y), "np.add out": lambda: np.add(y, x, out=y), "X.scale": lambda: X.scale(2.0, x), "2*x": lambda: 2.0*x}, repeats=5, number=50 if n < 1e6 else 5)
    print(n, {a: f"{b*1e6:.1f} us" for a, b in r.items()})
