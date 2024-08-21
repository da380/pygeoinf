import numpy as np
from pyshtools import SHCoeffs, SHGrid
from scipy.sparse import diags 
from linear_inference.vector_space import HilbertSpace

class Sobolev(HilbertSpace):
    
    def __init__(self, lmax, exponent, length_scale, /, *, grid = 'DH'):    
        self._lmax = lmax
        self._exponent = exponent
        self._length_scale = length_scale        
        self._grid = grid                
        self._dimension = (lmax+1)**2
        self._compute_metric()
        super(Sobolev, self).__init__(self._dimension, self._to_components, self._from_components, self._inner_product, from_dual = self._from_dual)

    # Return maximum degree. 
    @property
    def lmax(self):
        return self._lmax        

    @property 
    def dimension(self):
        return self._dimension

    @property
    def exponent(self):
        return self._exponent

    @property 
    def length_scale(self):
        return self._length_scale

    # Returns the index of the (l,m)th coefficient within the basis. 
    def component_index(self, l, m):
        if m >= 0:
            return int(l*(l+1)/2) + m
        else:
            offset = int((self.lmax + 1)*(self.lmax + 2) / 2)
            return offset + int((l - 1) * l / 2) - m - 1

    def _sobolev_factor(self, l):
        fac = 1 + (self.length_scale)**2 * l * (l + 1)
        return fac**(self.exponent)

    def _compute_metric(self):
        metric_values = np.zeros(self.dimension)
        i = 0
        for l in range(self.lmax+1):
            j = i + l + 1
            metric_values[i:j] = self._sobolev_factor(l)
            i = j
        for l in range(1,self.lmax+1):
            j = i + l
            metric_values[i:j] = self._sobolev_factor(l)
            i = j
        inverse_metric_values = np.reciprocal(metric_values)
        self._metric = diags([metric_values], [0])
        self._inverse_metric = diags([inverse_metric_values], [0])


    def _to_components(self, u):
        ulm = u.expand(normalization = "ortho")
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

    def _from_components(self,c):
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
        ulm = SHCoeffs.from_array(coeffs, normalization = "ortho")
        return ulm.expand()

    def _inner_product(self, u1, u2):
        c1 = self.to_components(u1)
        c2 = self.to_components(u2)
        return np.dots(self._metric @ c1, c2)

    def _from_dual(self, up):        
        cp = self.dual.to_components(up)
        c = self._inverse_metric @ cp
        return self.from_components(c)
        
                    
class Lebesgue(Sobolev):

    def __init__(self, lmax, /, *, grid = 'DH'):    
        super(Lebesgue,self).__init__(lmax, 0, 0, grid = grid)










    


        



            


    


