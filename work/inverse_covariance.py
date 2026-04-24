import time
import pygeoinf as inf
from pygeoinf.symmetric_space import sphere

X = sphere.Sobolev(32, 2, 0.1)

mu = X.heat_kernel_gaussian_measure(0.01)

P = X.point_evaluation_operator(X.random_points(1000))

Y = P.codomain

nu = mu.affine_mapping(operator=P, inverse_solver=inf.MinResSolver())


u = nu.sample()
v = nu.covariance(u)


start_time = time.time()
w = nu.inverse_covariance(v)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")


print(Y.norm(w - u) / Y.norm(u))
