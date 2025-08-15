import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
from pygeoinf.symmetric_space.line import Sobolev

X = Sobolev.from_sobolev_parameters(2,0.1)

print(X._sqrt_jac)
print(X._isqrt_jac)
print(X._sqrt_jac * X._isqrt_jac)

x1 = X.project_function(lambda x : x)

c1 = X.to_components(x1)

x2 = X.from_components(c1)

c2 = X.to_components(x2)

#print(c1-c2)



#fig, ax = X.plot(x1)
#X.plot(x2,fig, ax, linestyle = "--")
#X.plot(x3,fig, ax, linestyle = "-.")
plt.plot(c1)
plt.plot(c2,'--')
plt.show()