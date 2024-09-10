import numpy as np
from scipy.stats import norm
from pygeoinf.linalg import EuclideanSpace, LinearOperator, DirectLUSolver, DirectCholeskySolver, GMRESSolver

dimX = 100

g = norm().rvs(size=(dimX, dimX))
g = g @ g.T + np.identity(dimX)

X = EuclideanSpace(dimX, metric_tensor=g)

A = LinearOperator(X, X, mapping=lambda x : 2*x + x[0])

x1 = X.random()
x2 = X.random()
xp = X.dual.random()

#print(xp(A(x1)))
#print(A.dual(xp)(x1))
#print(X.inner_product(A(x1), x2))
#print(X.inner_product(x1, A.adjoint(x2)))

#solver = DirectLUSolver()
solver = GMRESSolver()

solver.set_operator(A)

x = X.random()
y = A(x)
z = solver(y)
print(np.linalg.norm(x-z) / np.linalg.norm(x))


b = norm().rvs(size=(dimX, dimX))
b = b @ b.T

B = LinearOperator.self_adjoint(X, mapping = A @ A.adjoint + 0.1 * X.identity())

cholesky_solver = DirectCholeskySolver()
cholesky_solver.set_operator(B)

x = X.random()
y = B(x)
z = cholesky_solver(y)
print(np.linalg.norm(x-z) / np.linalg.norm(x))


#print(xp(B(x1)))
#print(B.dual(xp)(x1))
#print(X.inner_product(B(x1), x2))
#print(X.inner_product(x1, B.adjoint(x2)))






  
