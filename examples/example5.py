import pyshtools as sh
import matplotlib.pyplot as plt

ulm = sh.SHCoeffs.from_zeros(lmax = 64)

ulm.coeffs[0,2,1] = 1

u = ulm.expand()


lat = 20
lon = 50 
print(sh.expand.MakeGridPoint(ulm.coeffs, lat, lon))




