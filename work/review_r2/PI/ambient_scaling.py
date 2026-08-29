import sys, time
import numpy as np
sys.path.insert(0, "/home/david/dev/pygeoinf")
from pygeoinf2.algebra.spaces import EuclideanSpace
from pygeoinf2.probability.gaussian import GaussianMeasure
for n in (2000, 4000, 8000):
    mu = GaussianMeasure.from_standard_deviation(EuclideanSpace(n), 0.3)
    t = time.perf_counter(); mu.ambient_ball(level=0.9); print(f"ambient_ball dim {n}: {time.perf_counter()-t:.2f} s", flush=True)
    t = time.perf_counter(); mu.credible_set(level=0.9); print(f"credible_set dim {n}: {time.perf_counter()-t:.2f} s", flush=True)
