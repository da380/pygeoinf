from pygeoinf import EuclideanSpace, HilbertSpaceChecks, BlockLinearOperator

X = EuclideanSpace(2)
Y = EuclideanSpace(3)

check = HilbertSpaceChecks(X)

print(check.all_checks_passed())

A = X.identity_operator()
B = Y.zero_operator(X)


C = BlockLinearOperator([[A, B]])

print(C)
print(C.adjoint)
