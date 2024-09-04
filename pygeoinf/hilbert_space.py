"""
This module contains the definition of the HilbertSpace class. 
"""

if __name__ == "__main__":
    pass

import numpy as np
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from pygeoinf.linear_form import LinearForm
from pygeoinf.vector_space import VectorSpace


class HilbertSpace(VectorSpace):
    """
    A class for real Hilbert spaces. A Hilbert space is represented in 
    terms of the following information:

    (1) The dimension of the space or of the approximating basis. 
    (2) A mapping from elements of the space to their components. This mapping
        serves to define the basis used for working with the space. 
    (3) A mapping from components to elements of the space. This must be the 
        the inverse of the mapping in point (2) though this is not checked. 
    (4) The inner product for the space, this being a bilinear mapping from
        the space to the real numbers. 
    (5) A mapping from a dual vector to its representation in the space. 
    (6) A mapping from the space to the dual vector it cannonically defines. 

    Note that this class does *not* define the elements of the space. These 
    elements must be implemented elsewhere. This class representing elements
    of the space may have the vector space operations defined using standard
    overloads (+,-,*). If this is not the case, functions that implement the 
    operations can be provided, or default implementations (that work at a
    component level) can be used.         

    If mappings to and from the dual space are not provided, they are 
    generated using default component-level implementations. Such 
    methods inversion of dense linear systems, and so will be inefficient 
    for high-dimensional spaces. 
    """    
    
    def __init__(self, dim,  to_components, from_components,
                 inner_product, /, *,  from_dual = None, 
                 to_dual = None, operations_defined=True, 
                 axpy=None, dual_base = None):                 
        """
        Args:
            dim (int): Dimension of the space or of the approximating basis. 
            to_components: Callable object that implements the mapping from
                the vector space to an array of its components.
            from_components: Callable object that implements the mapping from
                an array of components to a vector. 
            inner_product: Callable object that implements the inner product
                on the space, this being a bilinear mapping from the space 
                to the real numbers.
            to_dual: Callable object that implements the mapping from a dual 
                vector to its representation within the space. 
            from_dual: Callable object that implements the mapping from a 
                vector to the cannonically defined dual vector. 
            operations_defined (bool): Set to true if elements of the space
                have vector (+,-) and scalar (*,/) overloads defined. 
            axpy: Callable object that implements the transformation
                y -> a*x + y for vectors x and y and scalar y.
            dual_base (bool): Used internally to record that object is the
                dual of another VectorSpace.         
        """

        # Form the underlying vector space. 
        super(HilbertSpace,self).__init__(dim, to_components, from_components,
                                          operations_defined=operations_defined,
                                          axpy=axpy)

        # Set the inner
        self._inner_product = inner_product

        # Set the mapping from the dual space.         
        if from_dual is None:                        
            self._form_and_factor_metric()
            self._from_dual = lambda xp :  self._from_dual_default(xp)
        else:
            self._from_dual = from_dual

        # Set the mapping to the dual space. 
        if to_dual is None:
            self._to_dual = self._to_dual_default
        else:
            self._to_dual = to_dual

        # Store the base space (which may be none).
        self._dual_base = dual_base


    @staticmethod 
    def from_vector_space(space, inner_product, /, *,  from_dual = None,
                          to_dual = None):
        """
        Constructs a Hilbert space from a vector space given the inner product. 

        Args:
            space: The underlying vector space. 
            inner_product: Callable object that implements the inner product
                on the space, this being a bilinear mapping from the space 
                to the real numbers.
            to_dual: Callable object that implements the mapping from a dual 
                vector to its representation within the space. 
            from_dual: Callable object that implements the mapping from a 
                vector to the cannonically defined dual vector. 
            operations_defined (bool): Set to true if elements of the space
                have vector (+,-) and scalar (*,/) overloads defined. 
            axpy: Callable object that implements the transformation
                y -> a*x + y for vectors x and y and scalar y.       

            Returns:
                HilbertSpace: The Hilbert space formed.         
        """
        return HilbertSpace(space.dim, space.to_components, space.from_components,
                            inner_product, from_dual = from_dual, to_dual = to_dual, 
                            operations_defined=space.operations_defined, axpy=space.axpy)

    
    @property
    def dual(self):
        """The dual of the vector space."""
        if self._dual_base is None:            
            return HilbertSpace(self.dim,
                                self._dual_to_components,
                                self._dual_from_components, 
                                self._dual_inner_product,
                                from_dual = self.to_dual,
                                to_dual = self.from_dual,                                
                                dual_base = self)
        else:
            return self._dual_base        
    
    @property
    def to_vector_space(self):
        """The underlying vector space."""
        return VectorSpace(self.dim,self.to_components, self.from_components,
                           operations_defined=self._operations_defined, 
                           axpy=self.axpy, dual_base = self._dual_base)
                           
    def inner_product(self, x1, x2):
        """Returns the inner product of two vectors."""
        return self._inner_product(x1, x2)

    def norm(self, x):
        """ Returns the norm of a vector."""
        return np.sqrt(self.inner_product(x,x))

    def _form_and_factor_metric(self):
        # Construct the Cholesky factorisation of the metric.   
        metric = np.zeros((self.dim, self.dim))
        c1 = np.zeros(self.dim)
        c2 = np.zeros(self.dim)
        for i in range(self.dim):
            c1[i] = 1
            x1 = self.from_components(c1)
            metric[i,i] = self.inner_product(x1,x1)
            for j in range(i+1,self.dim):
                c2[j] = 1
                x2 = self.from_components(c2)                
                metric[i,j] = self.inner_product(x1,x2)          
                metric[j,i] = metric[i,j]
                c2[j] = 0
            c1[i] = 0                      
        self._metric_factor = cho_factor(metric)        
    
    def _from_dual_default(self,xp):    
        # Default implementation for the representation of a dual vector. 
        cp = self.dual.to_components(xp)
        c = cho_solve(self._metric_factor,cp)        
        return self.from_components(c)
    
    def from_dual(self, xp):        
        """Return the representation of a dual vector."""
        return self._from_dual(xp)
    
    def to_dual(self, x):        
        """Return the dual vector cannonically defined by a vector."""
        return self._to_dual(x)
    
    def _to_dual_default(self,x):
        # Default implementation of the mapping to the dual space. 
        return LinearForm(self, mapping = lambda y : self.inner_product(x,y))    

    def _dual_inner_product(self, xp1, xp2):
        # Inner product on the dual space. 
        return self.inner_product(self.from_dual(xp1),self.from_dual(xp2))



    