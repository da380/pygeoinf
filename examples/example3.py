from pygeoinf import EuclideanSpace, HilbertSpaceChecks

X = EuclideanSpace(2)

check = HilbertSpaceChecks(X)


print(check.passed_checks())

print(check.failed_checks())
