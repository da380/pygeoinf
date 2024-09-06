"""
This module defined the VectorSpace class along with a function 
that returns n-dimensional real vector space with its standard
basis as an instance of this class. 
"""

import numpy as np
from scipy.stats import norm
from pygeoinf.utils import run_randomised_checks




class VectorSpace:
    """
    A class for real vector spaces. To define an instance, the
    user needs to provide the following:

        (1) The dimension of the space, or the dimension of the 
            finite-dimensional approximating space. 
        (2) A mapping from elements of the space to their components. 
            These components must be expressed as numpy arrays with
            shape (dim,1) with dim the spaces dimension. 
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but 
            this requirement is not automatically checked. 

    Note that this class does *not* define elements of the 
    vector space. These must be pre-defined separately. It 
    is also assumed that the usual vector operations are 
    available for this latter space. 
    """

    def __init__(self, dim, to_components, from_components):
        """
        Args:
            dim (int): The dimension of the space, or of the 
                finite-dimensional approximating space. 
            to_components (callable):  A functor that maps vectors
                to their components. 
            from_components (callable): A functor that maps components
                to vectors. 
        """
        self._dim = dim
        self._to_components = to_components
        self._from_components = from_components
    
    @property
    def dim(self):
        """The dimension of the space."""
        return self._dim

    def to_components(self,x):
        """Maps vectors to components."""        
        return self._to_components(x)

    def from_components(self,c):
        """Maps components to vectors."""
        return self._from_components(c)

    def _random_components(self):
        # Generates a random set of components drawn 
        # from a standard Gaussian distribution. 
        return norm().rvs(size = (self.dim,1))

    def random(self):
        """
        Returns a random vector whose components have been 
        drawn from a standard Gaussian distribution. 
        """
        return self.from_components(self._random_components())

    def check(self, /, *,trials = 1, rtol = 1e-9):
        """
        Returns true is checks on the space have been passed. 

        Args:
            trials (int): The number of random instances of each check performed.
            rtol (float): The relative tolerance used within numerical checks. 

        Returns:
            bool: True if all checks have passed. 

        Notes:
            The purpose of this function is to check that the functions 
            provided to set up the vector space are consistent. Specifically,
            it checks that the functions to_components and from_components 
            are mutual inverses, and that they are linear. This is done by 
            computing their actions on randomly generated components and 
            associated vectors. Such tests cannot be conclusive but are 
            better than nothing.  
        """
        checks = (self._mutual_inverse, self._linearity)
        return run_randomised_checks(checks, trials, rtol)
                        
    def _mutual_inverse(self, rtol):
        c1 = self._random_components()
        c2 = self.to_components(self.from_components(c1))        
        return np.linalg.norm(c1-c2) < rtol * np.linalg.norm(c1)

    def _linearity(self, rtol):
        x1 = self.random()
        x2 = self.random()
        x = x1 + x2
        c1 = self.to_components(x1)
        c2 = self.to_components(x2)
        c = self.to_components(x)
        return np.linalg.norm(c1 + c2 - c) < rtol * np.linalg.norm(c)


def standard_vector_space(dim):
    """Returns n-dimensional real space with the standard basis."""    
    return VectorSpace(dim, lambda x : x.reshape(dim,1), lambda x : x.reshape(dim,))



