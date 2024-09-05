"""
This module defined the class SphericalHarmonicSpace. 
"""

import numpy as np
from pyshtools import SHCoeffs, SHGrid
from pygeoinf import VectorSpace

if __name__ == "__main__":
    pass


def SphericalHarmonicSpace(lmax, /, *, elements_as_SHGrid = True,
                           csphase = 1, grid = "DH", extend = True):
    """
    Args:
        lmax (int): Truncation degree for the space. 
        elements_as_SHGrid (bool): True if elements of the space are
            SHGrid objects. If false, the elements are SHCoeffs objects.
        csphase (int): Equal to 1 if Condon-Shortley phase not used with 
            the definition of the spherical harmonics, and equal to -1
            if it is applied. Default value 1. 
        grid (str): Grid option from pyshtools. The options are "DH", "DH2"
            and "GLQ". 
        extend (bool): If true, spatial grids are extended to include the 
            both longitudes 0 and 360.
            
    Raises:
        ValueError: If lmax is negative. 
        ValueError: If csphase value is not 1 or -1.
        ValueError: If grid is not one of "DH", "DH2", "GLQ"            

    """

    if lmax < 0:
        raise ValueError("lmax must be a natural number.")            
    if csphase not in [-1,1]:            
        raise ValueError("invalid csphase choice")
    if grid not in ["DH", "DH2", "GLQ"]:            
        raise ValueError("invalid grid choice")        

    dim = (lmax+1)**2

    if elements_as_SHGrid:
        to_components = _to_components_from_SHGrid
        from_components = _from_components_to_SHGrid                        
    else:
        to_components = _to_components_from_SHCoeffs
        from_components = _from_components_to_SHCoeffs            
    
    return VectorSpace(dim, to_components, from_components,_zero)


    @property
    def lmax(self):
        """Truncation degree for the expansions."""
        return self._lmax            
    
    @property
    def grid(self):
        """Grid option for SHGrid objects."""
        return self._grid

    @property
    def extend(self):
        """Extend option for SHGrid objects."""
        return self._extend

    @property
    def normalization(self):
        """Normalisation option for SHCoeffs objects."""
        return self._normalization

    @property
    def csphase(self):
        """Condon-Shortley phase object for SHCoeffs objects."""
        return self._csphase

    def _zero_local(self):
        if self._elements_as_SHGrid:
            if self.grid == "DH2":
                grid = "DH"
                sampling = 2
            else:
                grid = self._grid
                sampling = 1
            return SHGrid.from_zeros(self.lmax, grid=grid, sampling=sampling,
                                     extend=self.extend)
        else:
            return SHCoeffs.from_zeros(self.lmax, normalization=self.normalization, 
                                       csphase=self.csphase)


    def expand(self, u):
        """
        Expand a SHGrid object into a SHCoeffs, or conversely.

        The transformations are done using the parameters set within the class. 

        Args:
            u (SHGrid or SHCoeffs): The object to be transformed. 

        Returns:
            SHCoeffs or SHGrid: The result of the transformation. 
        """
        if isinstance(u,SHGrid):
            return u.expand(normalization = self.normalization, csphase = self.csphase)
        elif isinstance(u,SHCoeffs):            
            return u.expand(grid = self.grid, extend = self.extend)

    def spherical_harmonic_index(self, l, m):
        """Return the component vector index for the spherical harmonic indices."""
        if m >= 0:
            return int(l*(l+1)/2) + m
        else:
            offset = int((self.lmax + 1)*(self.lmax + 2) / 2)
            return offset + int((l - 1) * l / 2) - m - 1

    def _to_components_from_coeffs(self, coeffs):
        # Flatten shtools coefficient format into contiguous vector. 
        c = np.empty(self.dim)        
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            c[i:j] = coeffs[0,l,:l+1] 
            i = j
        for l in range(1,self.lmax+1):
            j = i + l
            c[i:j] = coeffs[1,l,1:l+1]
            i = j    
        return c

    def _to_components_from_SHCoeffs(self, ulm):        
        # Map a SHCoeffs object to its components as a contiguous vector. 
        return self._to_components_from_coeffs(ulm.coeffs)
    
    def _to_components_from_SHGrid(self, u):        
        # Map a SHGrid object to its components.
        return self._to_components_from_SHCoeffs(self.expand(u))
    
    def _from_components_to_SHCoeffs(self,c):
        # Map components to a SHCoeffs object.
        coeffs = np.zeros((2,self.lmax+1, self.lmax+1))
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            coeffs[0,l,:l+1] = c[i:j] 
            i = j
        for l in range(1,self.lmax+1): 
            j = i + l
            coeffs[1,l,1:l+1] = c[i:j] 
            i = j    
        ulm = SHCoeffs.from_array(coeffs, normalization = self._normalization, csphase = self._csphase)
        return ulm

    def _from_components_to_SHGrid(self,c):
        # Map components to a SHGrid object. 
        return self.expand(self._from_components_to_SHCoeffs(c))

    