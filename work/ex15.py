from pygeoinf.symmetric_space.sphere import Sobolev

# Parameters
lmax = 32
order = 2
scale = 0.1
radius = 2

prior_order = 2.0
prior_scale = 0.2

npoints = 50
std = 0.01

# Setup Model Space and Forward Problem
model_space = Sobolev(lmax, order, scale, radius=radius)

points = model_space.random_points(1000)

A = model_space.point_evaluation_operator(points)


L, D, R = A.random_svd(10, rtol=1e-1)

A_approx = L @ D @ R

print(len(D.extract_diagonal()))


prior = model_space.point_value_scaled_heat_kernel_gaussian_measure(0.1)

model = prior.sample()


data = A(model)

data_approx = A_approx(model)

data_space = A.codomain

print(data_space.norm(data - data_approx) / data_space.norm(data))
