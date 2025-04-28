from pygeoinf.hilbert import (
    EuclideanSpace,
    HilbertSpaceDirectSum,
    LinearOperator,
    BlockLinearOperator,
)


X = EuclideanSpace(2)
Y = EuclideanSpace(3)

Z = HilbertSpaceDirectSum([X, Y])

x = X.random()
y = Y.random()

# print(x)
# print(y)

I0 = Z.subspace_inclusion(0)
I1 = Z.subspace_inclusion(1)

z = Z.add(I0(x), I1(y))

# print(z)

print(Z.inner_product(z, z))

print(X.inner_product(x, x) + Y.inner_product(y, y))

A = X.identity_operator()
B = LinearOperator(Y, X, lambda y: y[: X.dim])
C = X.zero_operator(Y)
D = Y.zero_operator()

M = BlockLinearOperator([[A, B], [C, D]])
