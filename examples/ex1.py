import matplotlib.pyplot as plt

import numpy as np

import pygeoinf as inf
from pygeoinf.symmetric_space_new.circle import Lebesgue

X = Lebesgue(128)

x1 = X.project_function(lambda th: 1)

x2 = X.project_function(lambda th: 1)

print(X.inner_product(x1, x2))
