import pygeoinf as inf

X = inf.EuclideanSpace(2)

Y = inf.HilbertSpaceDirectSum([X, X])

Z = X


class OperatorTest(inf.LinearOperator):

    def __init__(self, domain):

        codomain = inf.HilbertSpaceDirectSum([domain, domain])

        def mapping(x):
            return [x, x]

        def formal_adjoint_mapping(y):
            [y1, y2] = y
            return self.domain.add(y1, y2)

        super().__init__(
            domain, codomain, mapping, formal_adjoint_mapping=formal_adjoint_mapping
        )


A = OperatorTest(Y)

print(A)

print(A.adjoint)
