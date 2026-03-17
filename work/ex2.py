import pyshtools as sh


u = sh.SHCoeffs.from_zeros(10)

data = u.to_array()
print(data)
