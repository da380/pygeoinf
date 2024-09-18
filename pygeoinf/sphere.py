import numpy as np
from pyshtools import SHGrid, SHCoeffs



class SHToolsHelper:

    def __init__(self, lmax, /, *, grid = "DH", extend = True, 
                 normalization = "ortho", csphase=1):
        self._lmax = lmax
        self._grid = grid
        if self.grid == "DH2":
            self._sampling = 2
        else:
            self._sampling = 1
        self._extend = extend
        self._normalization = normalization
        self._csphase = csphase
        

    @property
    def lmax(self):
        return self._lmax

    @property
    def dim(self):
        return (self.lmax+1)**2

    @property
    def grid(self):
        return self._grid

    @property
    def extend(self):
        return self._extend

    @property
    def normalization(self):
        return self._normalization

    @property
    def csphase(self):
        return self._csphase    

    
    def spherical_harmonic_index(self, l, m):
        """Return the component index for given spherical harmonic degree and order."""
        if m >= 0:
            return int(l*(l+1)/2) + m
        else:
            offset = int((self.lmax + 1)*(self.lmax + 2) / 2)
            return offset + int((l - 1) * l / 2) - m - 1

    def to_components_from_coeffs(self, coeffs):
        """Return component vector from coefficient array."""
        c = np.empty((self.dim,1))        
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            c[i:j,0] = coeffs[0,l,:l+1] 
            i = j
        for l in range(1,self.lmax+1):
            j = i + l
            c[i:j,0] = coeffs[1,l,1:l+1] 
            i = j    
        return c
    
    def to_components_from_SHCoeffs(self, ulm):        
        """Return component vector from SHCoeffs object."""
        return self.to_components_from_coeffs(ulm.coeffs)

    
    def to_components_from_SHGrid(self, u):      
        """Return component vector from SHGrid object."""
        ulm = u.expand(normalization=self.normalization, csphase=self.csphase)  
        return self.to_components_from_SHCoeffs(ulm)


    def from_components_to_SHCoeffs(self,c):
        """Return SHCoeffs object from its component vector."""
        coeffs = np.zeros((2,self.lmax+1, self.lmax+1))
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            coeffs[0,l,:l+1] = c[i:j,0] 
            i = j
        for l in range(1,self.lmax+1): 
            j = i + l
            coeffs[1,l,1:l+1] = c[i:j,0]
            i = j    
        ulm = SHCoeffs.from_array(coeffs, normalization = self.normalization, csphase = self.csphase)
        return ulm
    
    def from_components_to_SHGrid(self,c):        
        """Return SHGrid object from its component vector."""
        ulm = self.from_components_to_SHCoeffs(c)
        return ulm.expand(grid=self.grid, extend=self.extend)
