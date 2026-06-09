import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import Sobolev, plot


X = Sobolev.from_heat_kernel_prior(0.01, 2.0, 0.1, power_of_two=True)
print(f"Dimension of space    = {X.dim}")


A = X.invariant_automorphism(lambda k: k * k / (1 + 0.01 * k**4))

mu = X.heat_kernel_gaussian_measure(0.01)
u = mu.sample()


B = inf.LowRankEig.from_randomized(
    A, 10, method="variable", power=2, rtol=1e-3, measure=mu
)
print(f"Rank of decomposition = {B.rank}")


print(f"Estimated trace = {B.trace}")
print(f"True trace      = {A.trace}")
print(
    f"Estimated trace via Hutchinson = {inf.random_trace(A, 10, method="variable", rtol=1e-3)} "
)

print(f"Covariance norm (shortcut)        = {mu.hilbert_schmidt_norm()}")
print(
    f"Covariance norm (random sampling) = {np.sqrt(inf.random_trace(mu.covariance @ mu.covariance, 10, rtol=1e-3))}"
)

v = A(u)
w = B(u)


_, ax = plt.subplots()

plot(X, v, ax=ax, linestyle="-", color="red")
plot(X, w, ax=ax, linestyle="--", color="blue")

plt.show()
