import pygeoinf as inf
import numpy as np


X = inf.EuclideanSpace(3)

A = inf.LinearOperator(X, X, lambda x: 2 * x)

f = inf.NonLinearForm(
    X,
    lambda x: 0.5 * X.inner_product(A(x), A(x)),
    gradient=A.adjoint @ A,
    hessian=lambda x: A.adjoint @ A,
)

x = X.random()
dx = X.random()

print(f(x))
print(f.gradient(x))
print(f.hessian(x))

print(f(x + dx))
print(
    f(x)
    + X.inner_product(f.gradient(x), dx)
    + 0.5 * X.inner_product(f.hessian(x)(dx), dx)
)

g = f + f

print(g(x))
print(g.gradient(x))
print(g.hessian(x))

h = inf.LinearForm(X, mapping=lambda x: x[0])

print(h(x))
print(h.gradient(x))
print(h.hessian(x))

k = f + h

print(k(x))
print(k.gradient(x))
print(k.hessian(x))

print(type(k.hessian(x)))
