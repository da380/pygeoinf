import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev
import cartopy.crs as ccrs

lmax = 128
order = 2
scale = 0.1
radius = 2

prior_order = 1.5
prior_scale = 0.1

npoints = 10
std = 0.01


model_space = Sobolev(lmax, order, scale, radius=radius)

points = model_space.random_points(npoints)
lats = [lat for lat, _ in points]
lons = [lon for _, lon in points]
forward_operator = model_space.point_evaluation_operator(points)

model_prior_measure = model_space.point_value_scaled_sobolev_kernel_gaussian_measure(
    prior_order, prior_scale
)
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    forward_operator.codomain, std
)

constraint_operator = model_space.to_coefficient_operator(2, lmin=2)
constraint_value = np.array([2, 0, 0, 0, 0])

forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

constraint = inf.AffineSubspace.from_linear_equation(
    constraint_operator, constraint_value
)

inversion = inf.ConstrainedLinearBayesianInversion(
    forward_problem, model_prior_measure, constraint
)


model, data = forward_problem.synthetic_model_and_data(model_prior_measure)


fig1, ax1, im1 = model_space.plot(model)
fig1.colorbar(im1, orientation="horizontal", shrink=0.7)
ax1.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree())


model_posterior_measure = inversion.model_posterior_measure(
    data, inf.CholeskySolver(), inf.CholeskySolver()
)

model_posterior_expectation = model_posterior_measure.expectation

fig2, ax2, im2 = model_space.plot(model_posterior_expectation)
fig2.colorbar(im2, orientation="horizontal", shrink=0.7)
ax2.plot(lons, lats, "k^", markersize=5, transform=ccrs.PlateCarree())

plt.show()
