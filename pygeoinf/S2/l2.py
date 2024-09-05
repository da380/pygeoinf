"""
This module defines ...
"""

import numpy as np
from pygeoinf import HilbertSpace
from pygeoinf.s2.spherical_harmonic_space import SphericalHarmonicSpace

if __name__ == "__main__":
    pass



class L2(SphericalHarmonicSpace, HilbertSpace):

    def __init__(self, lmax, /, *, elements_as_SHGrid = True, csphase = 1,
                 grid = "DH", extend = True): 
        
        SphericalHarmonicSpace.__init__(self,lmax, elements_as_SHGrid=elements_as_SHGrid, 
                                        csphase=csphase, grid=grid, extend=extend)

        inner_product = lambda x1, x2, : np.dot(x1,x2)
        HilbertSpace.__init__(self, self.dim, self.to_components, self.from_components, 
                              inner_product)

        

              






