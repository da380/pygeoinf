import matplotlib.pyplot as plt

import pygeoinf as inf


X = inf.EuclideanSpace(4)

x = X.random()
xp = X.to_dual(x)

y = X.random()
yp = X.to_dual(y)

print(xp)
X.dual.ax(2, xp)
print(xp)
