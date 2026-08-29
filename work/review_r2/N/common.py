import sys, time, functools
sys.path.insert(0, "/home/david/dev/pygeoinf")
import numpy as np
from collections import Counter

class Counts:
    def __init__(self): self.c = Counter()
    def wrap_space(self, space, names=("inner_product","axpy","to_components","from_components","copy","scale_inplace")):
        for n in names:
            orig = getattr(space, n)
            def make(orig, n):
                def f(*a, **k):
                    self.c[n] += 1
                    return orig(*a, **k)
                return f
            object.__setattr__(space, n, make(orig, n))
        return space
    def patch_sh(self):
        import pyshtools.expand as ex
        for n in ("SHExpandDH", "MakeGridDH"):
            orig = getattr(ex, n)
            def make(orig, n):
                def f(*a, **k):
                    self.c[n] += 1
                    return orig(*a, **k)
                return f
            setattr(ex, n, make(orig, n))
    def reset(self): self.c.clear()
    def __repr__(self): return repr(dict(self.c))

def counting_operator(op, counts, key="apply"):
    from pygeoinf2.algebra.operators import LinearOperator
    def value(x):
        counts.c[key] += 1
        return op(x)
    def adjoint(y):
        counts.c[key+"_adj"] += 1
        return op.adjoint(y)
    return LinearOperator.from_callables(op.domain, op.codomain, value, adjoint=adjoint, traits=op.traits)

def timeit(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    return min(ts), r
