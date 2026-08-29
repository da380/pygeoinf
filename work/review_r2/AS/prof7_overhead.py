from common import *
from pygeoinf2 import LinearOperator, EuclideanSpace, Traits
from pygeoinf2.traits import close
from pygeoinf2.algebra.nodes import _Composition, _Sum
n = 10
X = EuclideanSpace(n)
M = np.eye(n)
mv = lambda x: M @ x
ops = [LinearOperator.from_callables(X, X, mv, adjoint=mv) for _ in range(8)]
x = np.ones(n)
raw = lambda: mv(mv(mv(mv(x))))
for k in (1, 2, 4, 8):
    C = ops[0]
    for o in ops[1:k]: C = C @ o
    adj = C.adjoint
    r = interleave({"raw matvec x k": lambda: [mv(x) for _ in range(k)], "C(x)": lambda: C(x), "C.adjoint(x)": lambda: adj(x), "C._adjoint_value": lambda: C._adjoint_value(x)}, repeats=5, number=2000)
    print(k, {a: f"{b*1e6:.2f} us" for a, b in r.items()})
# node construction and trait closure
r = interleave({"close(PD)": lambda: close(Traits.POSITIVE_DEFINITE), "A@B build": lambda: ops[0] @ ops[1], "A+B build": lambda: ops[0] + ops[1], "2*A build": lambda: 2.0*ops[0], "from_callables": lambda: LinearOperator.from_callables(X, X, mv, adjoint=mv)}, repeats=5, number=2000)
print({a: f"{b*1e6:.2f} us" for a, b in r.items()})
