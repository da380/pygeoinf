"""
This module contains the definition of the VectorSpace class. 
"""


import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from pygeoinf.linear_form import LinearForm


if __name__ == "__main__":
    pass



class VectorSpace:
    """
    A class for real vector spaces. A vector space is represented in 
    terms of the following information:

    (1) The dimension of the space or of the approximating basis. 
    (2) A mapping from elements of the space to their components. This mapping
        serves to define the basis used for working with the space. 
    (3) A mapping from components to elements of the space. This must be the 
        the inverse of the mapping in point (2) though this is not checked. 

    Note that this class does *not* define the elements of the space. These 
    elements must be implemented elsewhere. This class representing elements
    of the space may have the vector space operations defined using standard
    overloads (+,-,*). If this is not the case, functions that implement the 
    operations can be provided, or default implementations (that work at a
    component level) can be used. 
    """
    def __init__(self, dim, to_components, from_components, /, * , 
                 operations_defined=True, axpy = None, dual_base = None):
        """
        Args:
            dim (int): Dimension of the space or of the approximating basis. 
            to_components: Callable object that implements the mapping from
                the vector space to an array of its components.
            from_components: Callable object that implements the mapping from
                an array of components to a vector. 
            operations_defined (bool): Set to true if elements of the space
                have vector (+,-) and scalar (*,/) overloads defined. 
            axpy: Callable object that implements the transformation
                y -> a*x + y for vectors x and y and scalar y.
            dual_base (bool): Used internally to record that object is the
                dual of another VectorSpace. 
        """
        self._dim = dim
        self._to_components = to_components
        self._from_components = from_components    
        self._dual_base = dual_base
        self._operations_defined = operations_defined
        self._axpy = axpy        
        

    @property    
    def dim(self):
        """Dimension of the space or of its approximating basis."""
        return self._dim

    @property
    def dual(self):
        """The dual of the vector space."""
        if self._dual_base is None:            
            return VectorSpace(self.dim, self._dual_to_components, 
                               self._dual_from_components, dual_base = self)
        else:
            return self._dual_base

    @property
    def zero(self):
        """The zero vector within the space."""
        return self.from_components(np.zeros(self.dim))

    def to_components(self,x):
        """Maps a vector to its components."""
        if isinstance(x, LinearForm) and x.store_components:
            return x.components        
        else:                
            return self._to_components(x)
    
    def from_components(self,c):
        """Maps an array of components to the vector"""
        return self._from_components(c)    
    
    def _dual_to_components(self,xp):
        # Default implement of the mapping of a dual vector to 
        # its components within the induced basis. 
        n = self.dim
        c = np.zeros(n)
        cp = np.zeros(n)
        for i in range(n):
            c[i] = 1
            cp[i] = xp(self.from_components(c))
            c[i] = 0
        return cp
     
    def _dual_from_components(self, cp):
        # Maps dual components to the dual vector. 
        return LinearForm(self, components = cp)

    def random(self, dist = norm()):
        """
        Returns a vector whose components are idd samples from the given distribution.

        Args:
            dist: The distribution from which samples are to be drawn. This is required
                to be a scipy.stats distribution which has a "rvs" method. 

        Note:
            This method should not generally be used to generate random elements of the 
            vector space. It is instead a quick method that can be useful for testing.         
        """
        return self.from_components(norm.rvs(size = self.dim))

    
    def axpy(self, a, x, y):
        """ Returns a * x + y with scalar, a, and vectors, x and y."""
        if self._operations_defined:
            return a * x + y
        else:
            if self._axpy is not None:
                return self._axpy(a,x,y)
            else:
                cy = self.to_components(y)
                cx = self.to_components(x)
                cy += a * cx
                return self.from_components(cy)




