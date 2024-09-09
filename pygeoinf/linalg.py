"""
This module defined the VectorSpace class along with a function 
that returns n-dimensional real vector space with its standard
basis as an instance of this class. 
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm



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

    def __init__(self, dim, to_components, from_components, /, *, base = None):
        """
        Args:
            dim (int): The dimension of the space, or of the 
                finite-dimensional approximating space. 
            to_components (callable):  A functor that maps vectors
                to their components. 
            from_components (callable): A functor that maps components
                to vectors. 
            base (VectorSpace | None): Set to none for an original space, 
                and to the base space when forming the dual. 
        """
        self._dim = dim
        self._to_components = to_components
        self._from_components = from_components
        self._base = base
    
    @property
    def dim(self):
        """The dimension of the space."""
        return self._dim    

    @property
    def dual(self):
        """The dual of the vector space."""
        if self._base is None:
            return VectorSpace(self.dim, self._dual_to_components, self._dual_from_components, base = self)
        else:            
            return self._base

    def to_components(self,x):
        """Maps vectors to components."""        
        return self._to_components(x)

    def from_components(self,c):
        """Maps components to vectors."""
        return self._from_components(c)

    def random(self):
        """Returns a random vector with components drwn from a standard Gaussian distribution."""
        return self.from_components(norm().rvs(size = (self.dim,1)))

    def identity(self):
        return LinearOperator(self, self, mapping=lambda x: x)


    def _dual_to_components(self, xp):
        # Mapping to components for a dual space. Note that components are 
        # always column vectors, but a linear forms matrix representation is
        # a row vector. 
        return xp.matrix.reshape(self.dim,1)

    def _dual_from_components(self,cp):
        # Mapping from components for a dual space. Note that components are 
        # always column vectors, but a linear forms matrix representation is
        # a row vector. 
        return LinearForm(self, matrix = cp.reshape(1,self.dim))





class LinearOperator:
    """
    Class for linear operators between two vector spaces. To define an 
    instance, the user must provide the following:

        (1) The domain of the operator as an instance of VectorSpace. 
        (2) The codomain of the operator as an instance of VectorSpace.     

    To define the action of the operator they can provide either:

        (a) A functor that represents the action of the operator. 
        (b) The matrix representation for the operator relative to the 
            basis for the domain and codomain. This matrix can be a dense
            numpy matrix, a scipy sparse matrix, or another object that 
            behaves in the same way. 

    In addition, the user can supply a functor that implement the action of
    the operators dual. If the matrix representation is supplied, a default 
    implemention of the dual mapping is used. 

    Linear operators form an algebra over the reals in the usual way. Overloads
    for the relevant operators are provided. In all cases, these operations are
    lazily implemented. 
    """

    def __init__(self, domain, codomain, /, *, mapping = None,
                 dual_mapping = None, matrix = None, base = None):
        """
        Args:
            domain (VectorSpace): The domain of the operator. 
            codomain (VectorSpace): The codomain of the operator. 
            mapping (callable | None): A functor that implements the 
                action of the operator. 
            dual_mapping (callable | None): A functor that implements 
                the action of the dual operator. 
            adjoint_mapping (callable | None): A functor that implements
                the action of the adjoint operator. 
            matrix (MatrixLike | None): The matrix representation of the 
                operator relative to the bases for its domain and codomain.        
        """
        self._domain = domain
        self._codomain = codomain        
        self._matrix = matrix        
        self._base = base

        if matrix is None:
            self._mapping = mapping
            if dual_mapping is None:
                self._dual_mapping = self._dual_mapping_default
            else:
                self._dual_mapping = dual_mapping            
        else:
            self._mapping = self._mapping_from_matrix                    
            self._dual_mapping = self._dual_mapping_from_matrix                        
        self._adjoint_mapping = self._adjoint_mapping_default
        

    @staticmethod
    def self_dual(domain, mapping):
        """
        Returns a self-dual operator. 

        Args:
            domain (VectorSpace): The domain of the operator. 
            mapping (callable): A functor implementing the action of the operator. 
        """
        return LinearOperator(domain, domain.dual, mapping=mapping, dual_mapping=mapping)


    @staticmethod
    def self_adjoint(domain,  mapping):
        """
        Returns a self-adjoint operator.

        Args:
            domain (HilbertSpace): The domain of the operator.             
            mapping (callable): A functor implementing the action of the operator. 

        Raises:
            ValueError: If the domain is not a Hilbert space. 
        """
        if not isinstance(domain, HilbertSpace):
            raise ValueError("Domain is not a Hilbert space.")
        dual_mapping = lambda yp : domain.to_dual(mapping(domain.from_dual(yp)))
        operator = LinearOperator(domain, domain, mapping=mapping, 
                                  dual_mapping=dual_mapping)
        operator.set_adjoint_mapping(mapping)
        return operator

    @property
    def domain(self):
        """Domain of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain of the operator."""
        return self._codomain

    @property
    def matrix(self):
        """Matrix representation of the operator."""
        if self._matrix is None:            
            return self._compute_matrix()            
        else:            
            return self._matrix    

    @property
    def dual(self):
        """The dual of the operator."""
        if self._base is None:            
            return LinearOperator(self.codomain.dual, self.domain.dual,
                              mapping=self._dual_mapping, dual_mapping=self, base = self)                      
        else:
            return self._base

    @property
    def adjoint(self):
        """The adjoint of the operator."""
        return LinearOperator(self.codomain, self.domain, mapping=self._adjoint_mapping)

    def set_adjoint_mapping(self, adjoint_mapping):
        """Set the value of the adjoint mapping directly."""
        self._adjoint_mapping = adjoint_mapping


    def store_matrix(self):
        """Call to compute and store the operators matrix representation."""
        if self._matrix is None:
            self._matrix = self._compute_matrix()

    def _mapping_from_matrix(self,x):
        # Sets the mapping from the assigned matrix.        
        cx = self.domain.to_components(x)
        cy = self.matrix @ cx                
        return self.codomain.from_components(cy)

    def _dual_mapping_from_matrix(self,yp):
        # Action of the dual mapping via the matrix representation. 
        return self.domain.dual.from_components(self.matrix.T @ self.codomain.dual.to_components(yp))

    def _dual_mapping_default(self, yp):
        # Default implementation of the dual mapping. 
        return LinearForm(self.domain, mapping = lambda x: yp(self(x)))       

    def _adjoint_mapping_default(self, y):
        if not (isinstance(self.domain,HilbertSpace) 
                and isinstance(self.codomain,HilbertSpace)):
            raise ValueError("Adjoints defined only for operators on Hilbert spaces.")
        return self.domain.from_dual(self.dual(self.codomain.to_dual(y)))
    
    def _compute_matrix(self):                
        # Compute the matrix representation through.
        matrix = np.zeros((self.codomain.dim, self.domain.dim))              
        cx = np.zeros((self.domain.dim,1))                        
        for i in range(self.domain.dim):
            cx[i,0] = 1
            x = self.domain.from_components(cx)
            y = self(x)                
            matrix[:,i] = self.codomain.to_components(y)[:,0]
            cx[i,0] = 0
        return matrix            

    def __call__(self, x):
        """Action of the operator on a vector."""
        return self._mapping(x)

    def __neg__(self):
        """Negative of the operator."""
        return LinearOperator(self.domain, self.codomain, mapping=lambda x : -self(x), 
                              dual_mapping=lambda yp : -self.dual(yp))

    def __mul__(self, a):
        """Multiply by a scalar."""
        return LinearOperator(self.domain, self.codomain, mapping=lambda x: a * self(x), 
                              dual_mapping=lambda yp : a * self.dual(yp))

    def __rmul__(self, a):
        """Multiply by a scalar."""
        return self * a

    def __truediv__(self, a):
        """Divide by scalar."""
        return self * (1/a)

    def __add__(self, other):
        """Add another operator."""
        if self.domain != other.domain or self.codomain != other.codomain:
            raise ValueError("Operators cannot be added.")
        return LinearOperator(self.domain, self.codomain, mapping=lambda x : self(x) + other(x), 
                              dual_mapping=lambda yp : self.dual(yp) + other.dual(yp))

    def __sub__(self, other):
        """Subtract another operator."""
        if self.domain != other.domain or self.codomain != other.codomain:
            raise ValueError("Operators cannot be subtracted.")
        return LinearOperator(self.domain, self.codomain, mapping=lambda x : self(x) - other(x), 
                              dual_mapping=lambda yp : self.dual(yp) - other.dual(yp))                              

    def __matmul__(self, other):
        """Compose with another operator."""
        if self.domain != other.codomain:
            raise ValueError("Operators cannot be composed")
        return LinearOperator(other.domain, self.codomain, mapping=lambda x : self(other(x)), 
                              dual_mapping=lambda yp : other.dual(self.dual(yp)))


    def __str__(self):
        """Print the operator as its matrix representation."""
        return self.matrix.__str__()




# Global definition of the real numbers as a VectorSpace. 
_REAL = VectorSpace(1, lambda x : np.array([[x]]), lambda c : c[0,0])




class LinearForm(LinearOperator):
    """
    Class for linear forms on a vector space. 

    The vector space is represented by a VectorSpace object. Linear forms
    are (continuous) linear mappings from the to the real numbers. These
    mappings form a vector space known that is *dual* to the original.

    A linear form can be specified either:

        (1) In terms of a functor that performs its action on a vector. 
        (2) Its matrix-representation relative to the basis for the space.

    Specification of a form in terms of its matrix representation offers
    computational advatages in many cases. A form specified in terms 
    of a mapping can (as with a general linear operator) compute and 
    store its matrix representation.
    """

    def __init__(self, domain, /, *, mapping = None, matrix = None):
        """
        Args:
            domain (VectorSpace): Domain of the linear form. 
            mapping (callable | None): A functor that performs the action
                of the linear form on a vector. 
            matrix (MatrixLike | None): The matrix representation of the 
                form, this having shape (1,dim) with dim the dimension of
                the domain. 
        """
        super().__init__(domain, _REAL, mapping = mapping, matrix=matrix )



class HilbertSpace(VectorSpace):
    """
    Class for Hilbert spaces. To define an instance, the user provides:

        (1) The dimension of the space, or the dimension of the 
            finite-dimensional approximating space. 
        (2) A mapping from elements of the space to their components. 
            These components must be expressed as numpy arrays with
            shape (dim,1) with dim the spaces dimension. 
        (3) A mapping from components back to the vectors. This
            needs to be the inverse of the mapping in (2), but 
            this requirement is not automatically checked. 
        (4) The inner product on the space. 

    Note that this space inherits from VectorSpace. 

    The user can also provide either of the following:
        
        (a) The mapping from the space to its dual. 
        (b) The mapping from a dual vector to its representation 
            within the space. 

    The existence of such mappings follows from the Riesz representation
    theorem. If (a) is not provided, then then cannonical form of this
    mapping is used, with a vector, x, being mapped to the dual vector 
    that acts on a vector, y, by y -> (x,y).

    If (b) is not provided, the metric tensor for the space relative 
    to its basis is computed along with its Cholesky factor. In this 
    case the mapping in (a) is replaced by one using dual components
    that is more efficient. 

    Within high-dimensional spaces, custom mappings to and from the 
    dual space should be provided for the user for better efficiency.  

    A HilbertSpace can be constructed directly, or from an already 
    formed VectorSpace using the static method "from_vector_space". 
    The latter option is often preferable as a custom definition of 
    the functor "to_dual" requires that the dual space be available.
    """

    def __init__(self, dim, to_components, from_components, inner_product,
                 /, *, to_dual = None, from_dual = None, base = None):
        """
        Args:
            dim (int): The dimension of the space, or of the 
                finite-dimensional approximating space. 
            to_components (callable):  A functor that maps vectors
                to their components. 
            from_components (callable): A functor that maps components
                to vectors.         
            inner_product (callable): A functor the implements the inner 
                product on the space. 
            to_dual (callable | None): A funcator that maps a vector 
                to the cannonically associated dual vector. 
            from_dual (callable | None): A functor that maps a dual vector
                to its representation on the space. 
            base (VectorSpace | None): Set to none for an original space, 
                and to the base space when forming the dual.                 
        """
        super().__init__(dim, to_components, from_components)
        self._inner_product = inner_product

        if from_dual is None:
            self._metric_tensor = self.calculate_metric_tensor()
            self._metric_tensor_factor = cho_factor(self.metric_tensor)
            self._from_dual = self._from_dual_default
        else:
            self._metric_tensor = None
            self._from_dual = from_dual            

        if to_dual is None:
            if self._metric_tensor is None:
                self._to_dual = self._to_dual_default
            else:                                
                self._to_dual = self._to_dual_default_with_metric
        else:
            self._to_dual = to_dual

        self._base = base


    @staticmethod
    def from_vector_space(space, inner_product, /, *, to_dual = None, from_dual = None):
        """
        Form a HilbertSpace from a VectorSpace by providing additional structures. 

        Args:
            space (VectorSpace): The vector space to form as base for the Hilbert space. 
            inner_product (callable): A functor the implements the inner 
                product on the space. 
            to_dual (callable | None): A funcator that maps a vector 
                to the cannonically associated dual vector. 
            from_dual (callable | None): A functor that maps a dual vector
                to its representation on the space.             
        """
        return HilbertSpace(space.dim, space.to_components, space.from_components, 
                            inner_product, to_dual=to_dual, from_dual=from_dual)

    @property
    def vector_space(self):
        """The underlying vector space."""
        return VectorSpace(self.dim, self.to_components, self.from_components)
    
    @property
    def dual(self):
        """The dual of the Hilbert space."""
        if self._base is None:
            return HilbertSpace(self.dim, self._dual_to_components, self._dual_from_components, 
                                self._dual_inner_product, to_dual=self.from_dual, 
                                from_dual=self.to_dual, base = self)
        else:            
            return self._base

    @property
    def metric_tensor(self):
        """The metric tensor for the space."""
        if self._metric_tensor is None:
            return self.calculate_metric_tensor()
        else:
            return self._metric_tensor
        
    def inner_product(self, x1, x2):
        """Return the inner product of two vectors."""
        return self._inner_product(x1,x2)

    def norm(self,x):
        """Return the norm of a vector."""
        return np.sqrt(self.inner_product(x,x))
        
    def to_dual(self, x):
        """Map a vector to cannonically associated dual vector."""
        return self._to_dual(x)

    def from_dual(self, xp):
        """Map a dual vector to its representation in the space."""
        return self._from_dual(xp)

    def calculate_metric_tensor(self):
        """Return the space's metric tensor as a numpy matrix."""
        metric_tensor = np.zeros((self.dim, self.dim))
        c1 = np.zeros((self.dim,1))
        c2 = np.zeros((self.dim,1))
        for i in range(self.dim):
            c1[i,0] = 1
            x1 = self.from_components(c1)            
            metric_tensor[i,i] = self.inner_product(x1,x1)
            for j in range(i+1,self.dim):
                c2[j,0] = 1
                x2 = self.from_components(c2)
                metric_tensor[i,j] = self.inner_product(x1,x2)
                metric_tensor[j,i] = metric_tensor[i,j]
                c2[j,0] = 0
            c1[i,0] = 0
        return metric_tensor
    
    def _to_dual_default(self,x):
        return LinearForm(self, mapping = lambda y : self.inner_product(x,y))

    def _to_dual_default_with_metric(self, x):
        cp = self.metric_tensor @ self.to_components(x)
        return self.dual.from_components(cp)

    def _from_dual_default(self, xp):
        cp = self.dual.to_components(xp)
        c = cho_solve(self._metric_tensor_factor, cp)
        return self.from_components(c)

    def _dual_inner_product(self, xp1, xp2):
        return self.inner_product(self.from_dual(xp1), self.from_dual(xp2))            


class EuclideanSpace(HilbertSpace):

    def __init__(self, dim):        
        space = VectorSpace(dim, self._to_components_local, self._from_components_local)        
        super().__init__(dim, self._to_components_local, self._from_components_local,
                         self._inner_product_local, to_dual=self._to_dual_local, 
                         from_dual=self._from_dual_local)
                         
    def _to_components_local(self, x):
        return x.reshape(self.dim,1)

    def _from_components_local(self, c):
        return c.reshape(self.dim,)

    def _inner_product_local(self, x1, x2):
        return np.dot(x1, x2)

    def _to_dual_local(self, x):
        return self.dual.from_components(self.to_components(x))

    def _from_dual_local(self, xp):
        return self.from_components(self.dual.to_components(xp))
            



        


