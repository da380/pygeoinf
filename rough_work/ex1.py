import pygeoinf as inf

X = inf.EuclideanSpace(10)
Y = inf.EuclideanSpace(5)

A = inf.LinearOperator(X, Y, lambda x: x[: Y.dim])

mu = inf.GaussianMeasure.from_standard_deviation(X, 0.1)
nu = inf.GaussianMeasure.from_standard_deviation(Y, 2.0)

A.check(domain_measure=mu, codomain_measure=nu)
