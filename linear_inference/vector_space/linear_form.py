if __name__ == "__main__":
    pass

import numpy as np

# Class for linear forms on a vector space. 
class LinearForm:

    def __init__(self, domain, /, *,  mapping = None, components = None, components_stored = False):
        self._domain = domain        
        self._components = components
        if mapping is None:
            assert components is not None
            assert components.size == domain.dimension
            self._mapping = lambda x : np.dot(domain.to_components(x),components)            
        if components is None:
            assert mapping is not None
            self._mapping = mapping
            if components_stored:
                self._components = self.compute_components()                
        
    # Return the domain of the linear form.
    @property
    def domain(self):
        return self._domain    

    # Compute the components of the form relative to the induced basis. 
    def compute_components(self):         
        return self.domain.dual.to_components(self) 

    # Return true if the components have been stored. 
    @property
    def components_stored(self):
        return (self._components is not None)

    # Return the compents of the form relative to the induced basis. 
    @property
    def components(self):
        if self.components_stored:
            return self._components            
        else:
            return self.compute_components()            
            
    # Return action of the form on a vector. 
    def __call__(self,x):
        return self._mapping(x)

    # Overloads to make LinearForm a vector space. 
    def __mul__(self, s):
        return LinearForm(self.domain, lambda x : s * self(x))

    def __rmul__(self,s):
        return self * s

    def __div__(self, s):
        return self * (1/s)

    def __add__(self, other):
        assert self.domain == other.domain        
        return LinearForm(self.domain, lambda x : self(x) + other(x))

    def __sub__(self, other):
        assert self.domain == other.domain        
        return LinearForm(self.domain, lambda x : self(x) - other(x))         

    def __matmul__(self, other):
        return self(other)

    def __str__(self):
        return self.domain.dual.to_components(self).__str__()


