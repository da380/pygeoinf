from pygeoinf.linalg import VectorSpace, EuclideanSpace, VectorSpaceDirectSum

X = EuclideanSpace(2)
Y = EuclideanSpace(3)

Z = VectorSpaceDirectSum([X, Y])

x = X.random()
y = Y.random()

print(x)
print(y)

c = Z.to_components([x, y])
z = Z.from_components(c)
print(z)

P = Z.projection(0)

I = Z.inclusion(0)

print(P.dual)
print(I)
print(I.dual)


z1 = Z.random()
z2 = Z.random()
print(z1)
print(z2)

z3 = Z.add(z1, z2)
print(z3)

z4 = Z.copy(z3)
z4 = Z.axpy(2, z3, Z.zero)
print(z4)
