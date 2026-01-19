import matplotlib.pyplot as plt
from pygeoinf.symmetric_space.sphere import Sobolev


lmax = 64
order = 2
scale = 0.1

# X = Lebesgue(lmax)
X = Sobolev(lmax, order, scale)

mu = X.heat_kernel_gaussian_measure(0.1)

u = mu.sample()

# X.plot(u)
# plt.show()

A = X.to_coefficient_operator(X.lmax)
B = X.from_coefficient_operator(X.lmax)

A.check()
B.check()

v = A(u)

w = B(v)

fig1, ax1, im1 = X.plot(u)
fig1.colorbar(im1, orientation="horizontal", shrink=0.7)
fig2, ax2, im2 = X.plot(w - u)
fig2.colorbar(im2, orientation="horizontal", shrink=0.7)
plt.show()
