import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from pygeoinf.symmetric_space import sphere

X = sphere.Sobolev(32, 2, 0.1)


mu = X.heat_kernel_gaussian_measure(0.1)


nu = mu.with_sparse_approximation()


u = mu.sample()
v = mu.covariance(u)
w = nu.covariance(u)
z = nu.inverse_covariance(v)


_, axes = plt.subplots(
    2, 2, figsize=(16, 12), subplot_kw={"projection": ccrs.PlateCarree()}
)

sphere.plot(u, ax=axes[0, 0])
sphere.plot(v, ax=axes[0, 1])
sphere.plot(z, ax=axes[1, 0])
sphere.plot(w, ax=axes[1, 1])

axes[0, 0].set_title(r"Sample $u \sim \mu$")
axes[0, 1].set_title(r"Exact Covariance: $v = C u$")
axes[1, 0].set_title(r"Sparse Precision: $z = \tilde{C}^{-1} v$")
axes[1, 1].set_title(r"Sparse Covariance: $w = \tilde{C} u$")

plt.show()
