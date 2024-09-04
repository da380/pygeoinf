"""
This module contains the definition of the LinearForm class.
"""

import numpy as np

if __name__ == "__main__":
    pass

class LinearForm:

    """
    Class for linear forms defined on a real vector space. 

    A linear form can be either be defined directly through a mapping, or in 
    terms of its components relative to the induced basis.
    """

    def __init__(self, domain, /, *, mapping=None, components=None,
                 store_components=False):
        """
        Args:
            domain (VectorSpace): Space over which the linear form is defined. 
            mapping (VectorSpace -> float | None): Callable object that implements
                the action of the linear norm. 
            components (ArrayLike | None):  Components of the form relative to the
                induced basis. 
            store_componets (bool): If true, component values are pre-computed and
                stored. If components are provided, value set to true.
        """
        
        self._domain = domain        
        self._components = components
        if mapping is None:
            assert components is not None
            assert components.size == domain.dim
            self._mapping = lambda x : np.dot(domain.to_components(x),
                                              components)
        if components is None:
            assert mapping is not None
            self._mapping = mapping
            if store_components:
                self._components = self._compute_components()                
            
    @property
    def domain(self):
        """Vector space over which the form is defined."""
        return self._domain    

    @property
    def store_components(self):
        """True is the components of the form are stored internally."""
        return (self._components is not None)

    @property
    def components(self):
        """Components of the form relative to the induced basis."""
        if self.store_components:
            return self._components            
        else:
            return self._compute_components()            

    def _compute_components(self):         
        # Computes componets of the form relative the the basis for its domain. 
        return self.domain.dual.to_components(self) 
                
    def __call__(self,x):
        """ Return action of the form on a vector."""
        return self._mapping(x)
    
    def __mul__(self, s):
        """Multiply form by a scalar."""
        if self.store_components:
            return LinearForm(self.domain, components = s * self.components)
        else:
            return LinearForm(self.domain, mapping = lambda x : s * self(x))

    def __rmul__(self,s):
        """Multiply form by a scalar."""        
        return self * s

    def __div__(self, s):
        """Divide form by a scalar."""                
        return self * (1/s)

    def __add__(self, other):
        """Add linear forms"""
        assert self.domain == other.domain        
        if self.store_components and other.store_components:
            return LinearForm(self.domain, components = self.components + other.components)
        else:
            return LinearForm(self.domain, mapping = lambda x : self(x) + other(x))

    def __sub__(self, other):
        """Subtract linear forms"""
        assert self.domain == other.domain
        if self.store_components and other.store_components:
            return LinearForm(self.domain, components = self.components - other.components)
        else:
            return LinearForm(self.domain, mapping = lambda x : self(x) - other(x))
        
    def __str__(self):
        """Print linea form using its components"""
        return self.domain.dual.to_components(self).__str__()


