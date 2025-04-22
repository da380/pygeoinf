import matplotlib.pyplot as plt

from pygeoinf.linalg import EuclideanSpace, GaussianMeasure
from pygeoinf.sphere import Sobolev

X = Sobolev(64, 2, 0.1, vector_as_SHGrid=False)


mu = X.sobolev_gaussian_measure(3, 0.1, 1)

n = 2
samples = [mu.sample() for _ in range(n)]


nu = GaussianMeasure.from_samples(X, samples)

u = nu.sample()

# fig, ax = plt.subplots(1, 1)

out = X.plot(u)

plt.show()
