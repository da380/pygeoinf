# Class for linear forms on a vector space. 
class LinearForm:

    def __init__(self, domain, mapping):
        self._domain = domain    
        self._mapping = mapping    

    # Return the domain of the linear form.
    @property
    def domain(self):
        return self._domain    

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
