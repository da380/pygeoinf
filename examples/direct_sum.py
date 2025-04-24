from pygeoinf.linalg import VectorSpace, EuclideanSpace, VectorSpaceDirectSum

X = EuclideanSpace(2)
Y = EuclideanSpace(3)

Z = VectorSpaceDirectSum([X, Y])
