from pygeoinf.symmetric_space.sphere import Sobolev

X = Sobolev(128, 2, 0.1)

points = X.random_points(1000)


A = X.point_evaluation_operator(points, matrix_free=True, parallel=True, n_jobs=10)
A.check()
