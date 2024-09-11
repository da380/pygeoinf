"""
This module defined the VectorSpace class along with a function 
that returns n-dimensional real vector space with its standard
basis as an instance of this class. 
"""

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
from scipy.stats import norm, multivariate_normal
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from scipy.sparse.linalg import gmres, bicg, bicgstab, cg



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

    @property
    def zero(self):
        return self.from_components(np.zeros((self.dim,1)))

    def to_components(self,x):
        """Maps vectors to components."""        
        return self._to_components(x)

    def from_components(self,c):
        """Maps components to vectors."""
        return self._from_components(c)


    def basis_vector(self,i):
        """Return the ith basis vector."""
        c = np.zeros((self.dim,1))
        c[i,0] = 1
        return self.from_components(c)

    def random(self):
        """Returns a random vector with components drwn from a standard Gaussian distribution."""
        return self.from_components(norm().rvs(size = (self.dim,1)))

    def identity(self):
        """Returns identity operator on the space."""
        if isinstance(self, HilbertSpace):
            return LinearOperator(self, self, mapping=lambda x: x, dual_mapping = lambda yp : yp, 
                                  adjoint_mapping=lambda y : y)            
        else:
            return LinearOperator(self, self, mapping=lambda x: x, dual_mapping = lambda yp : yp)            


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
        (b) The matrix representation for the operator. 
        (c) The Galerkin matrix representation for operators 
            between Hilbert spaces. 

    For options (b) and (c) the action of the operator's dual and 
    (where appropriate) adjoint are deduced internally. Note that 
    the matrices can either dense or in the scipy sparse format. 

    For option (a) the dual and adjoint can be deduced internally 
    but in an inefficient manner, or they can be supplied directly. 

    Linear operators form an algebra over the reals in the usual way. Overloads
    for the relevant operators are provided. In all cases, these operations are
    lazily implemented. 
    """

    def __init__(self, domain, codomain, /, *, matrix = None,
                 galerkin_matrix = None, mapping = None,
                 dual_mapping = None, adjoint_mapping = None,  
                 dual_base = None, adjoint_base = None):
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
        # Store basic information. 
        self._domain = domain
        self._codomain = codomain        
        self._dual_base = dual_base
        self._adjoint_base = adjoint_base

        # Set defaults that may be overwritten below. 
        self._matrix = None
        self._galerkin_matrix = None
        self._adjoint_mapping = None

        # Set mappings from the matrix representation.  
        if matrix is not None:
            self._matrix = matrix
            if galerkin_matrix is not None:
                raise ValueError("Cannot specify matrix representation and Galerkin representation")
            if mapping is not None:
                raise ValueError("Cannot specify matrix representation and mapping.")
            if dual_mapping is not None:
                raise ValueError("Cannot specify matrix representation and dual mapping.")
            if adjoint_mapping is not None:
                raise ValueError("Cannot specify matrix representation and adjoint mapping.")
            self._mapping = self._mapping_from_matrix
            self._dual_mapping = self._dual_mapping_from_matrix
            if self.hilbert_operator:
                self._adjoint_mapping = self._adjoint_mapping_from_dual
            else:
                self._adjoint_mapping = None
        
        if galerkin_matrix is not None:
            if not self.hilbert_operator:
                raise ValueError("Galerkin representation only available for Hilbert space operators")
            self._galerkin_matrix = galerkin_matrix
            if matrix is not None:
                raise ValueError("Cannot specify Galerkin representation and matrix representation")
            if mapping is not None:
                raise ValueError("Cannot specify Galerkin representation and mapping.")
            if dual_mapping is not None:
                raise ValueError("Cannot specify Galerkin representation and dual mapping.")
            if adjoint_mapping is not None:
                raise ValueError("Cannot specify Galerkin representation and adjoint mapping.")
            self._mapping = self._mapping_from_galerkin_matrix
            self._dual_mapping = self._dual_mapping_from_galerkin_matrix
            self._adjoint_mapping = self._adjoint_mapping_from_galerkin_matrix

        # Set mapping directly.
        if mapping is not None:            
            self._mapping = mapping
            if dual_mapping is None:                            
                if self.hilbert_operator:
                    if adjoint_mapping is None:
                        self._dual_mapping = self._dual_mapping_default
                        self._adjoint_mapping = self._adjoint_mapping_from_dual
                    else:
                        self._adjoint_mapping = adjoint_mapping
                        self._dual_mapping = self._dual_mapping_from_adjoint
                else:                    
                    self._dual_mapping = self._dual_mapping_default
                    self._adjoint_mapping = None
            else:
                self._dual_mapping = dual_mapping
                if self.hilbert_operator:
                    if adjoint_mapping is None:
                        self._adjoint_mapping = self._adjoint_mapping_from_dual
                    else:
                        self._adjoint_mapping = adjoint_mapping
            
    @staticmethod
    def self_dual(domain, mapping):
        """Returns a self-dual operator in terms of its domain and mapping."""
        return LinearOperator(domain, domain.dual, mapping=mapping, dual_mapping=mapping)

    @staticmethod
    def self_adjoint(domain, mapping):
        """Returns a self-adjoint operator in terms of its domain and mapping."""
        return LinearOperator(domain, domain, mapping=mapping, adjoint_mapping=mapping)

    @property
    def domain(self):
        """Domain of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain of the operator."""
        return self._codomain

    @property
    def hilbert_operator(self):
        """True is operator maps between Hilbert spaces."""
        return isinstance(self.domain,HilbertSpace) and isinstance(self.codomain,HilbertSpace)

    @property
    def matrix(self):
        """Matrix representation of the operator."""
        if self._matrix is None:            
            return self._compute_matrix()            
        else:            
            return self._matrix    

    @property
    def galerkin_matrix(self):
        """Matrix representation of the operator."""
        if self._galerkin_matrix is None:            
            return self._compute_galerkin_matrix()
        else:            
            return self._galerkin_matrix

    @property
    def dual(self):
        """The dual of the operator."""
        if self._dual_base is None:            
            return LinearOperator(self.codomain.dual, self.domain.dual,
                                  mapping=self._dual_mapping, dual_mapping=self,
                                  dual_base = self)                      
        else:
            return self._dual_base

    @property
    def adjoint(self):
        """The adjoint of the operator."""
        if not self.hilbert_operator:
            raise NotImplementedError("Adjoint not defined for the operator.")
        if self._adjoint_base is None:
            return LinearOperator(self.codomain, self.domain, mapping=self._adjoint_mapping,
                                  adjoint_mapping=self._mapping, adjoint_base=self)
        else:
            return self._adjoint_base


    @property
    def scipy_linear_operator(self):
        """Returns the operator in scipy.sparse.LinearOperator form."""
        matvec = lambda c : self.codomain.to_components(self(self.domain.from_components(c)))
        rmatvec = lambda c : self.domain.dual.to_components(self.dual(self.codomain.dual.from_components(c)))
        return ScipyLinOp((self.codomain.dim, self.domain.dim), matvec=matvec, rmatvec=rmatvec)


    def _mapping_from_matrix(self,x):
        # Sets the mapping from the matrix representation. 
        cx = self.domain.to_components(x)
        cy = self.matrix @ cx
        return self.codomain.from_components(cy)

    def _mapping_from_galerkin_matrix(self, x):
        # Sets the mapping from the Galerkin representation.
        cx = self.domain.to_components(x)
        cyp = self.galerkin_matrix @ cx
        yp = self.codomain.dual.from_components(cyp)
        return self.codomain.from_dual(yp)

    def _dual_mapping_default(self, yp):
        # Default implementation of the dual mapping. 
        return LinearForm(self.domain, mapping=lambda x : yp(self(x)))       

    def _dual_mapping_from_matrix(self,yp):
        # Action of the dual mapping via the matrix representation. 
        cyp = self.codomain.dual.to_components(yp)
        cxp = self.matrix.T @ cyp
        return self.domain.dual.from_components(cxp)

    def _dual_mapping_from_galerkin_matrix(self, yp):
        # Action of the dual mapping via the matrix representation. 
        y = self.codomain.from_dual(yp)
        cy = self.codomain.to_components(y)
        cxp = self.galerkin_matrix.T @ cy
        return  self.domain.dual.from_components(cxp)

    def _dual_mapping_from_adjoint(self, yp):
        # Dual mapping in terms of the adjoint. 
        y = self.codomain.from_dual(yp)
        x = self._adjoint_mapping(y)
        return self.domain.to_dual(x)

    def _adjoint_mapping_from_galerkin_matrix(self, y):
        # Action of the adjoint from the Galerkin representation. 
        cy = self.codomain.to_components(y)
        cxp = self.galerkin_matrix.T @ cy
        xp = self.domain.dual.from_components(cxp)
        return self.domain.from_dual(xp)

    def _adjoint_mapping_from_dual(self, y):
        # Adjoing mapping in terms of the dual.
        yp = self.codomain.to_dual(y)
        xp = self._dual_mapping(yp)
        return self.domain.from_dual(xp)

    def _compute_matrix(self):                
        # Compute the matrix representation.
        matrix = np.zeros((self.codomain.dim, self.domain.dim))              
        cx = np.zeros((self.domain.dim,1))                        
        for i in range(self.domain.dim):
            cx[i,0] = 1
            y = self(self.domain.from_components(cx))            
            matrix[:,i] = self.codomain.to_components(y)[:,0]
            cx[i,0] = 0
        return matrix            

    def store_matrix(self):
        """Compute and store the operator's matrix representation."""
        if self._matrix is None:
            self._matrix = self._compute_matrix()

    def _compute_galerkin_matrix(self):
        # Computes the Galerkin matrix representation. 
        if not self.hilbert_operator:
            raise NotImplementedError("Galerkin matrix only defined for Hilbert space operators.")
        matrix = np.zeros((self.codomain.dim, self.domain.dim))              
        cx = np.zeros((self.domain.dim,1))     
        for i in range(self.domain.dim):
            cx[i,0] = 1            
            y = self(self.domain.from_components(cx))
            yp = self.codomain.to_dual(y)
            matrix[:,i] = self.codomain.dual.to_components(yp)[:,0]
            cx[i,0] = 0
        return matrix            

    def store_galerkin_matrix(self):
        """Compute and store the operator's matrix representation."""
        if self._galerkin_matrix is None:
            self._galerkin_matrix = self._compute_galerkin_matrix()

    def __call__(self, x):
        """Action of the operator on a vector."""
        if self._mapping is None:
            raise NotImplementedError("Mapping has not been set.")
        return self._mapping(x)

    def __neg__(self):
        """Negative of the operator."""
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x : -self(x)
        dual_mapping = lambda yp : -self.dual(yp)
        if self.hilbert_operator:
            adjoint_mapping = lambda y : -self.adjoint(y)            
        return LinearOperator(domain, codomain, mapping=mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)

    def __mul__(self, a):
        """Multiply by a scalar."""
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x : a * self(x)
        dual_mapping = lambda yp : a * self.dual(yp)
        if self.hilbert_operator:
            adjoint_mapping = lambda y : a * self.adjoint(y)
        else:
            adjoint_mapping = None            
        return LinearOperator(domain, codomain, mapping=mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)        
        
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
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x :  self(x) + other(x)
        dual_mapping = lambda yp : self.dual(yp) + other.dual(yp)
        if self.hilbert_operator:        
            adjoint_mapping = lambda y : self.adjoint(y) + other.adjoint(y)
        else:
            adjoint_mapping = None                        
        return LinearOperator(domain, codomain, mapping=mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)        
        
    def __sub__(self, other):
        """Subtract another operator."""
        if self.domain != other.domain or self.codomain != other.codomain:
            raise ValueError("Operators cannot be subtracted.")
        domain = self.domain
        codomain = self.codomain
        mapping = lambda x :  self(x) - other(x)
        dual_mapping = lambda yp : self.dual(yp) - other.dual(yp)
        if self.hilbert_operator:
            adjoint_mapping = lambda y : self.adjoint(y) - other.adjoint(y)
        else:
            adjoint_mapping = None                                    
        return LinearOperator(domain, codomain, mapping=mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)                           

    def __matmul__(self, other):
        """Compose with another operator."""
        if self.domain != other.codomain:
            raise ValueError("Operators cannot be composed")
        domain = other.domain
        codomain = self.codomain
        mapping = lambda x :  self(other(x))
        dual_mapping = lambda yp : other.dual(self.dual(yp))
        if self.hilbert_operator:
            adjoint_mapping = lambda y : other.adjoint(self.adjoint(y))
        else:
            adjoint_mapping = None                        
        return LinearOperator(domain, codomain, mapping=mapping, dual_mapping=dual_mapping,
                              adjoint_mapping=adjoint_mapping)                           

        
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


def euclidean_space(dim, /, *, metric_tensor = None):
    """
    Returns Euclidean space implemented as a HilbertSpace instance. 

    Args:
        dim (int): Dimension of the space. 
        metric_tensor (ArrayLike | None): The metric tensor that defines
            the inner product on the space. If none is provided, this is 
            taken to the identity matrix, and hence the standard Euclidean
            inner product is used. 

    Notes:
        To work with a non-standard inner product, the Cholesky factor of the 
        metric tensor is formed. This implementation uses dense matrices, and 
        will not be efficient in high-dimensional spaces. An alternative 
        implementation that uses pre-conditioned iterative solvers would be 
        preferable in such cases. 
    """

    to_components = lambda x : x.reshape(dim,1)
    from_components = lambda c : c.reshape(dim,)
    space = VectorSpace(dim, to_components, from_components)

    if metric_tensor is None:        
        inner_product = lambda x1, x2 : np.dot(x1, x2)
        to_dual = lambda x:  space.dual.from_components(space.to_components(x))
        from_dual = lambda xp : space.from_components(space.dual.to_components(xp))        
    else:
        factor = cho_factor(metric_tensor)
        inner_product = lambda x1, x2 : np.dot(metric_tensor @ x1, x2)
        to_dual = lambda x : space.dual.from_components(metric_tensor @space.to_components(x))
        from_dual = lambda xp : space.from_components(cho_solve(factor, space.dual.to_components(xp)))

    return HilbertSpace.from_vector_space(space, inner_product,to_dual=to_dual, from_dual=from_dual)

            
class GaussianMeasure:
    """
    Class for Gaussian measures on a Hilbert space.  
    """

    def __init__(self, domain, covariance, / , *, expectation = None,
                 sample = None, sample_using_matrix = False):  
        """
        Args:
            domain (HilbertSpace): The Hilbert space on which the measure is defined. 
            covariance (callable): A functor representing the covariance operator. 
            expectation (Vector): The expectation value of the measure. If none is provided, set equal to zero.  
            sample (callable): A functor that returns a random sample from the measure.         
        """
        self._domain = domain              
        self._covariance = covariance
        if expectation is None:
            self._expectation = self.domain.zero
        else:
            self._expectation = expectation
        if sample is None:
            self._sample_defined = False
        else:
            self._sample = sample
            self._sample_defined = True
        if sample is None and sample_using_matrix:
            dist = multivariate_normal(mean = self.expectation, cov= self.covariance.matrix)
            self._sample = lambda : self.domain.from_components(dist.rvs())
            self._sample_defined = True


    @staticmethod
    def from_factored_covariance(factor, /, *,  expectation = None):
            
        covariance  = factor @ factor.adjoint
        sample = lambda : factor(norm().rvs(size = factor.domain.dim))
        if expectation is not None:
            sample = lambda : expectation + sample()        
        return GaussianMeasure(factor.codomain, covariance, expectation = expectation, sample = sample)    

    @property
    def domain(self):
        """The Hilbert space the measure is defined on."""
        return self._domain
    
    @property
    def covariance(self):
        """The covariance operator as an instance of LinearOperator."""
        return LinearOperator.self_adjoint(self.domain, self._covariance)
    
    @property
    def expectation(self):
        """The expectation of the measure."""
        return self._expectation

    @property
    def sample_defined(self):
        """True if the sample method has been implemented."""
        return self._sample_defined
    
    def sample(self):
        """Returns a random sample drawn from the measure."""
        if self.sample_defined:        
            return self._sample()
        else:
            raise NotImplementedError

    def affine_mapping(self, /, *,  operator = None, translation = None):
        """
        Returns the push forward of the measure under an affine mapping.

        Args:
            operator (LinearOperator): The operator part of the mapping.
            translation (vector): The translational part of the mapping.

        Returns:
            Gaussian Measure: The transformed measure defined on the 
                codomain of the operator.

        Raises:
            ValueError: If the domain of the operator domain is not 
                the domain of the measure.

        Notes:
            If operator is not set, it defaults to the identity.
            It translation is not set, it defaults to zero. 
        """
        assert operator.domain == self.domain        
        if operator is None:
            covariance = self.covariance
        else:
            covariance = operator @ self.covariance @ operator.adjoint        
        expectation = operator(self.expectation)
        if translation is not None:
            expectation = expectation + translation
        if self.sample_defined:
            if translation is None:
                sample = lambda : operator(self.sample())
            else: 
                sample = lambda  : operator(self.sample()) + translation                                        
        else:
            sample = None
        return GaussianMeasure(operator.codomain, covariance, expectation = expectation, sample = sample)            

    def __neg__(self):
        """Negative of the measure."""
        if self.sample_defined:
            sample = lambda : -self.sample()
        else:
            sample = None    
        return GaussianMeasure(self.domain, self.covariance, expectation=-self.expectation, sample=sample)
    
        
    def __mul__(self, alpha):
        """Multiply the measure by a scalar."""
        covariance = LinearOperator.self_adjoint(self.domain,lambda x : alpha*2 * self.covariance(x))
        expectation = alpha * self.expectation
        if self.sample_defined:
            sample = lambda : alpha * self.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample)

    def __rmul__(self, alpha):
        """Multiply the measure by a scalar."""
        return self * alpha
    
    def __add__(self, other):
        """Add two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        expectation = self.expectation + other.expectation
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() + other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample) 

    def __sub__(self, other):
        """Subtract two measures on the same domain."""
        assert self.domain == other.domain
        covariance = self.covariance + other.covariance
        expectation = self.expectation - other.expectation
        if self.sample_defined and other.sample_defined:
            sample  = lambda : self.sample() - other.sample()
        else:
            sample = None
        return GaussianMeasure(self.domain, covariance, expectation = expectation, sample = sample)     


class LinearSolver(LinearOperator, ABC):
    """
    Abstract base class for linear solvers. 

    Each linear solver must have a method called "set_operator" which takes a 
    LinearOperator as its sole argument. Once this routine has been called, the 
    LinearSolver then works as a LinearOperator whose action is the inverse of 
    the operator from which it was formed. 

    Any parameters needed to set up the linear solver should be passed within 
    the constuctor for the LinearSolver class. 
    """
    
    @abstractmethod    
    def set_operator(self, operator):
        """Set the linear operator whose inverse is to be formed."""
        pass

    def _check_dimensions(self, operator):
        # Check that the operator can have a well-defined inverse.
        if operator.domain.dim != operator.codomain.dim:
            raise ValueError("Domain and codomain must have the same dimensions.")


class Preconditioner(LinearSolver):
    """
    Abstract base class for preconditioners for linear systems. This class is
    functionally identical to the LinearSolver class, but the intention is that
    instances of this class do not need to provide exact implementations of the 
    inverse mapping.
    """
    pass


class DirectLUSolver(LinearSolver):
    """
    Linear solver class based on direct LU decomposition of the matrix representation. 
    """

    def __init__(self, /, *, overwrite = False, check_finite = False):
        """
        Args:
            overwrite (bool): Overwrite the matrix when forming the LU
                decomposition, or the rhs during backsubstitution. 
            check_finite (bool): Check matirx and rhs values are finite.                            
        """
        self._overwrite = overwrite
        self._check_finite = check_finite

    def set_operator(self, operator):    
        """ Set the operator."""
        self._check_dimensions(operator)
        self._factor = lu_factor(operator.matrix,  
                                 overwrite_a=self._overwrite, 
                                 check_finite=self._check_finite)        
        super().__init__(operator.codomain, operator.domain, 
                         mapping=self._mapping_local,
                         dual_mapping=self._dual_mapping_local)
        
    def _mapping_local(self, y):
        # Implement action of the inverse operator via back-substitution. 
        cy = self.domain.to_components(y)
        cx = lu_solve(self._factor, cy, 
                      overwrite_b=self._overwrite,
                      check_finite=self._check_finite)
        return self.codomain.from_components(cx)

    def _dual_mapping_local(self, xp):
        # Implement action of the dual of the inverse operator via back-substitution. 
        cxp = self.codomain.dual.to_components(xp)
        cyp = lu_solve(self._factor, cxp, trans=1,
                      overwrite_b=self._overwrite,
                      check_finite=self._check_finite)
        return self.domain.dual.from_components(cyp)
        

class DirectCholeskySolver(LinearSolver):
    """Linear solver class based on direct Cholesky decomposition of the 
       Galerkin matrix representation. Method applicable only for self-adjoint
       and positive-definite operators. 
    """

    def __init__(self, /, *, lower = False, overwrite = False, check_finite = False):
        """
        Args:
            lower (bool): Form lower triangular factorisation. 
            overwrite (bool): Overwrite the matrix when forming the LU
                decomposition, or the rhs during backsubstitution. 
            check_finite (bool): Check matirx and rhs values are finite.                            
        """
        self._lower = lower
        self._overwrite = overwrite
        self._check_finite = check_finite

    def set_operator(self, operator):    
        """ Set the operator."""
        self._check_dimensions(operator)
        if not operator.hilbert_operator:
            raise ValueError("Cholesky decomposition applicable only for operators between Hilbert spaces.")
        self._factor = cho_factor(operator.galerkin_matrix, 
                                  lower=self._lower, 
                                  overwrite_a=self._overwrite, 
                                  check_finite=self._check_finite)        
        super().__init__(operator.codomain, operator.domain,
                         mapping=self._mapping_local,
                         adjoint_mapping=self._mapping_local)

    def _mapping_local(self, y):
        # Implement the action of the operator via back-substitution. 
        cyp = self.domain.dual.to_components(self.domain.to_dual(y))
        cxp = cho_solve(self._factor, cyp, overwrite_b=self._overwrite,
                        check_finite=self._check_finite)
        return self.codomain.from_components(cxp)


class DiagonalPreconditioner(Preconditioner):

    def __init__(self, /, *, bandwidth = 0):
        pass


    def set_operator(self, operator):
        self._check_dimensions(operator)

        n = operator.domain.dim
        diagonal_values = np.zeros(n)
        cx = np.zeros((n,1))
        for i in range(n):
            cx[i,0] = 1
            x = operator.domain.from_components(cx)
            y = operator(x)
            diagonal_values[i] = 1/y[i]
            cx[i,0] = 0

        matrix = diags((diagonal_values))
        super().__init__(operator.codomain, operator.domain, 
                         matrix=matrix)

    
class GMRESSolver(LinearSolver):

    def __init__(self, /, *, x0 = None, preconditioner=None,
                 rtol=1e-05, atol=0.0, restart=None, maxiter=None, 
                 callback=None, callback_type=None):
        self._x0 = x0
        self._preconditioner = preconditioner
        self._rtol = rtol
        self._atol = atol
        self._restart = restart
        self._maxiter = maxiter
        self._callback = callback
        self._callback_type = callback_type
            


    def set_operator(self, operator):
        self._check_dimensions(operator)
        self._operator = operator
        super().__init__(operator.codomain, operator.domain, mapping = self._mapping_local)
        

    def _mapping_local(self, y):
        cy = self.domain.to_components(y)
        if self._x0 is None:
            c0 = None
        else:        
            c0 = self.codomain.to_components(self._x0)

        if self._preconditioner is None:
            M = None
        else:
            self._preconditioner.set_operator(self._operator)
            M = self._preconditioner.scipy_linear_operator
        cx, _ =  gmres(self._operator.scipy_linear_operator, cy, c0, rtol=self._rtol, 
                       atol=self._atol, restart=self._restart, maxiter=self._maxiter,
                       M=M, callback=self._callback, callback_type=self._callback_type)
        return self.codomain.from_components(cx)


class BICGSolver(LinearSolver):

    def __init__(self, /, *, x0 = None, preconditioner=None,
                 rtol=1e-05, atol=0.0, maxiter=None, callback=None):
        self._x0 = x0
        self._preconditioner = preconditioner
        self._rtol = rtol
        self._atol = atol        
        self._maxiter = maxiter
        self._callback = callback        

    
    def set_operator(self, operator):
        self._check_dimensions(operator)
        self._operator = operator
        super().__init__(operator.codomain, operator.domain, mapping = self._mapping_local)
        

    def _mapping_local(self, y):
        cy = self.domain.to_components(y)
        if self._x0 is None:
            c0 = None
        else:        
            c0 = self.codomain.to_components(self._x0)

        if self._preconditioner is None:
            M = None
        else:
            self._preconditioner.set_operator(self._operator)
            M = self._preconditioner.scipy_linear_operator
        cx, _ =  bicg(self._operator.scipy_linear_operator, cy, c0, rtol=self._rtol, 
                      atol=self._atol, maxiter=self._maxiter, M=M, callback=self._callback)
        return self.codomain.from_components(cx)
            

class BICGStabSolver(LinearSolver):

    def __init__(self, /, *, x0 = None, preconditioner=None,
                 rtol=1e-05, atol=0.0, maxiter=None, callback=None):
        self._x0 = x0
        self._preconditioner = preconditioner
        self._rtol = rtol
        self._atol = atol        
        self._maxiter = maxiter
        self._callback = callback        

    
    def set_operator(self, operator):
        self._check_dimensions(operator)
        self._operator = operator
        super().__init__(operator.codomain, operator.domain, mapping = self._mapping_local)
        

    def _mapping_local(self, y):
        cy = self.domain.to_components(y)
        if self._x0 is None:
            c0 = None
        else:        
            c0 = self.codomain.to_components(self._x0)

        if self._preconditioner is None:
            M = None
        else:
            self._preconditioner.set_operator(self._operator)
            M = self._preconditioner.scipy_linear_operator
        cx, _ =  bicgstab(self._operator.scipy_linear_operator, cy, c0, rtol=self._rtol, 
                      atol=self._atol, maxiter=self._maxiter, M=M, callback=self._callback)
        return self.codomain.from_components(cx)


class CGSolver(LinearSolver):

    def __init__(self, /, *, x0 = None, preconditioner=None,
                 rtol=1e-05, atol=0.0, maxiter=None, callback=None):
        self._x0 = x0
        self._preconditioner = preconditioner
        self._rtol = rtol
        self._atol = atol        
        self._maxiter = maxiter
        self._callback = callback        

    
    def set_operator(self, operator):
        self._check_dimensions(operator)
        self._operator = operator
        super().__init__(operator.codomain, operator.domain, mapping = self._mapping_local)
        

    def _mapping_local(self, y):
        cy = self.domain.to_components(y)
        if self._x0 is None:
            c0 = None
        else:        
            c0 = self.codomain.to_components(self._x0)

        if self._preconditioner is None:
            M = None
        else:
            self._preconditioner.set_operator(self._operator)
            M = self._preconditioner.scipy_linear_operator
        cx, _ =  cg(self._operator.scipy_linear_operator, cy, c0, rtol=self._rtol, 
                      atol=self._atol, maxiter=self._maxiter, M=M, callback=self._callback)
        return self.codomain.from_components(cx)

            



        


