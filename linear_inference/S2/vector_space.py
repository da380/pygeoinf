if __name__ == "__main__":
    pass

import numpy as np
from pyshtools import SHCoeffs, SHGrid
from scipy.sparse import diags 
from linear_inference.vector_space import LinearForm
from linear_inference.vector_space import HilbertSpace

# Implements the Sobolev space H^s on a two-sphere using a spherical harmonic basis. Vectors are by default 
# instances of the SHGrid class from pyshtools, but they can also take the form of SHCoeffs objects. 
# The space is characterised by its exponent, s, and a length-scale, mu. The inner product is defined by
#
# (u,v)_{H^{s}} = ( (1 + \mu^2 \Delta))^{-s/2} u, (1 + \mu^2 \Delta))^{-s/2} v)_{L^2} 
#
# with \Delta the Laplace beltrami operator. The following default values are assumed:
#
# vectors_as_SHGrid = True --> Parameter determines whether SHGrid or SHCoeffs is used for vectors. 
# radius = 1 --> Parameter sets the radius of the sphere. 
# grid = "DH" --> Parameter sets the grid used for the spherical harmonic transformations. Default is Driscol-Healy,
#                 but other options from pyshtools can be used (i.e., GLQ and DH2).abs
# normalization = "ortho" --> Parameter sets the normalisation convention used for the spherical harmonics. 
#                             Default is orthonormalised but other options availabe in pyshtools can be used. 
#
class HS(HilbertSpace):
    
    def __init__(self, lmax, exponent, length_scale, /, *, vectors_as_SHGrid = True, radius = 1, grid = "DH", normalization = "ortho"):    

        # Store the basic information. 
        self._lmax = lmax
        self._exponent = exponent
        self._length_scale = length_scale  
        self._vectors_as_SHGrid = vectors_as_SHGrid   
        self._radius = radius  
        self._grid = grid              
        self._normalization = normalization

        # Construct the metric and its inverse. 
        sobolev_factor = lambda l : radius * radius * (1 + (length_scale)**2 * l * (l + 1))**exponent
        dimension = (lmax+1)**2
        metric_values = np.zeros(dimension)
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
        super(HS, self).__init__(dimension, to_components, from_components, 
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
        c = np.empty(self.dimension)        
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
        ulm = u.expand(normalization = "ortho")
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
        ulm = SHCoeffs.from_array(coeffs, normalization = self._normalization)
        return ulm

    # Map components to a SHGrid object. 
    def _from_components_to_SHGrid(self,c):
        ulm = self._from_components_to_SHCoeffs(c)
        return ulm.expand(grid = self._grid)

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
class L2(HS):

    def __init__(self, lmax,  /, *, vectors_as_SHGrid = True, radius = 1, grid = "DH", normalization = "ortho"):    
        super(L2,self).__init__(lmax, 0, 0, 
                                           vectors_as_SHGrid = vectors_as_SHGrid, 
                                           radius = radius, 
                                           grid = grid, 
                                           normalization = normalization)










    


        



            


    


