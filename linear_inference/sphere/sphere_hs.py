if __name__ == "__main__":
    pass

import numpy as np
from pyshtools import SHCoeffs, SHGrid
from scipy.sparse import diags 
from linear_inference.vector_space import LinearForm
from linear_inference.vector_space import HilbertSpace

# Implements the Sobolev space H^s on a sphere using a spherical harmonic basis. The inner product
# is defined in terms of the Laplace-Beltrami operator, \Delta, by:
#
# (u,v)_{H^s} = ( \Lambda^{s/2} u, \Lambda^{s/2} v)_{L^2}
#
# where \Lambda = 1 + \lambda^{2} \Delta and \lambda is the chosen length-scale. 
class SphereHS(HilbertSpace):
    
    def __init__(self, lmax, /, *, exponent = 0, radius = 1, length_scale = 1,
                 vectors_as_SHGrid = True, csphase = 1, grid = "DH", extend = True):    

        # Store the basic information. 
        self._lmax = lmax        
        self._exponent = exponent
        self._radius = radius  
        self._length_scale = length_scale / radius

        # Store SHTools options. 
        self._vectors_as_SHGrid = vectors_as_SHGrid   
        self._normalization = "ortho"
        if csphase in [-1,1]:
            self._csphase = csphase
        else:
            raise ValueError("invalid csphase choice")
        if grid in ["DH", "DH2", "GLQ"]:            
            self._grid = grid
            if grid == "DH2":
                self._sampling = 2
            else:
                self._sampling = 1            
        else:
            raise ValueError("invalid grid choice")        
        self._extend = extend

        # Construct the metric and its inverse. 
        sobolev_factor = lambda l : radius * radius * (1 + (length_scale)**2 * l * (l + 1))**exponent
        dim = (lmax+1)**2
        metric_values = np.zeros(dim)
        i = 0
        for l in range(lmax+1):
            j = i + l + 1
            metric_values[i:j] = sobolev_factor(l)
            i = j
        for l in range(1,lmax+1):
            j = i + l
            metric_values[i:j] = sobolev_factor(l)
            i = j        
        inverse_metric_values = np.reciprocal(metric_values)
        self._metric = diags([metric_values], [0])
        self._inverse_metric = diags([inverse_metric_values], [0])    
    
        # Set the mappings to and from components.                         
        if vectors_as_SHGrid:
            to_components = self._to_components_from_SHGrid
            from_components = self._from_components_to_SHGrid
        else:
            to_components = self._to_components_from_SHCoeffs
            from_components = self._from_components_to_SHCoeffs

        # Construct the base class. 
        super(SphereHS, self).__init__(dim, to_components, from_components, 
                                 self._inner_product, from_dual = self._from_dual, 
                                 to_dual = self._to_dual)

    # Return maximum degree. 
    @property
    def lmax(self):
        return self._lmax            

    # Return the sobolev exponent. 
    @property
    def exponent(self):
        return self._exponent

    # Return the length scale. 
    @property 
    def length_scale(self):
        return self._length_scale

    # Return the component index for the (l,m)th spherical harmonic coefficient
    def spherical_harmonic_index(self, l, m):
        if m >= 0:
            return int(l*(l+1)/2) + m
        else:
            offset = int((self.lmax + 1)*(self.lmax + 2) / 2)
            return offset + int((l - 1) * l / 2) - m - 1

    # Map a SHCoeffs object to its components as a contiguous vector. 
    def _to_components_from_SHCoeffs(self, ulm):        
        c = np.empty(self.dim)        
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            c[i:j] = ulm.coeffs[0,l,:l+1]    
            i = j
        for l in range(1,self.lmax+1):
            j = i + l
            c[i:j] = ulm.coeffs[1,l,1:l+1]
            i = j    
        return c

    # Map a SHGrid object to its components.
    def _to_components_from_SHGrid(self, u):
        ulm = u.expand(normalization = self._normalization, csphase = self._csphase)
        return self._to_components_from_SHCoeffs(ulm)

    # Map components to a SHCoeffs object.
    def _from_components_to_SHCoeffs(self,c):
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

    # Map components to a SHGrid object. 
    def _from_components_to_SHGrid(self,c):
        ulm = self._from_components_to_SHCoeffs(c)
        return ulm.expand(grid = self._grid, extend = self._extend)

    # Local definition of the inner product. 
    def _inner_product(self, u1, u2):
        c1 = self.to_components(u1)
        c2 = self.to_components(u2)
        return np.dot(self._metric @ c1, c2)

    # Local definition of mapping from the dual. 
    def _from_dual(self, up):        
        cp = self.dual.to_components(up)
        c = self._inverse_metric @ cp
        return self.from_components(c)

    # Local definition of mapping to the dual.
    def _to_dual(self, u):
        c = self.to_components(x)
        cp = self._metric @ c
        return LinearForm(self, components = cp)

    
        
# Implementation of the Lebesgue space L^{2} on a two-sphere. Obtained as a special case of H^{s} with exponent set to zero. 
# Note that with this value of s, the value of the length-scale does not matter. 
class SphereL2(SphereHS):

    def __init__(self, lmax, /, *, radius = 1, length_scale = 1,
                 vectors_as_SHGrid = True, csphase = 1, grid = "DH", extend = True):    
        super(L2,self).__init__(lmax, exponent = 0, radius = radius, length_scale = length_scale,
                 vectors_as_SHGrid = vectors_as_SHGrid, csphase = csphase, grid = grid, extend = extend):  










    


        



            


    


