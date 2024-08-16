from LinearInference.Operator import Linear, SelfAdjoint
import numpy as np
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.sparse.linalg import cg


# Solver based on LU decomposition of the matrix representation.  
class LU(Linear):
    def __init__(self,A):
        assert A.Domain.Dimension == A.CoDomain.Dimension
        factor = lu_factor(A.AsDense)        
        mapping = lambda x : A.Domain.FromComponents(lu_solve(factor, A.CoDomain.ToComponents(x)))
        dualMapping = lambda x : A.CoDomain.Dual.FromComponents(lu_solve(factor, A.Domain.Dual.ToComponents(x),trans = 1))
        super(LU,self).__init__(A.CoDomain, A.Domain, mapping, dualMapping)


# Solver for self-adjoint operators based on Cholesky decomposition. 
class Cholesky(SelfAdjoint):
    def __init__(self,A):
        assert A.Domain == A.CoDomain
        factor = cho_factor(A.AsDense)        
        mapping = lambda x : A.Domain.FromComponents(cho_solve(factor, A.Domain.ToComponents(x)))        
        super(Cholesky,self).__init__(A.Domain,mapping)


# Conjugate gradient solver for self-adjoint operators. 
class CG(SelfAdjoint):
    def __init__(self,A):
        assert A.Domain == A.CoDomain    
        mapping = lambda x : A.Domain.FromComponents(cg(A.AsSciPyLinearOperator, A.Domain.ToComponents(x))[0])
        super(CG,self).__init__(A.Domain,mapping)
    
