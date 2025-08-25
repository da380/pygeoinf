import pyshtools as sh
import numpy as np


u = sh.SHCoeffs.from_zeros(10)

data = u.to_array()
print(data)
