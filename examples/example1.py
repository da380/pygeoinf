import numpy as np
from scipy.stats import norm
import pygeoinf.linalg as la

dimX = 4

g = norm().rvs((dimX, dimX))
g =  g @ g.T +   np.identity(dimX)
X = la.euclidean_space(dimX, metric_tensor=g)

x = X.random()

xp = la.LinearForm(X, mapping=lambda x : x[0])

print(X.dual.to_components(xp))




#A = la.LinearOperator(X, X, lambda x : 2 * x + x[0])



#A = A + A.adjoint




#solver = la.MatrixSolverBICSTAB()
#solver = la.MatrixSolverGMRES()
#solver = la.MatrixSolverLU()

#solver.operator = A

#B = solver.inverse_operator

#print(A.dual @ B.dual)

#solver.operator = A
#B = solver.inverse_operator




#solver = la.MatrixSolverCholesky(galerkin=True)
#solver = la.MatrixSolverCG(galerkin=True)
#B = solver.inverse(A)

#x = X.random()
#y = A(x)
#z = B(y)

#print(np.linalg.norm(x-z) / np.linalg.norm(x))




#x1 = X.random()
#x2 = X.random()
#xp = X.dual.random()

#B = la.DirectLUSolver(galerkin=False)
#B = la.DirectCholeskySolver(galerkin=True)

#B.set_operator(A)

#print(B @ A)

#print(xp(A(x1)))
#print(A.dual(xp)(x1))
#print(X.inner_product(A(x1), x2))
#print(X.inner_product(x1, A.adjoint(x2)))

#B = la.DirectLUSolver(use_galerkin=True)

#B.set_operator(A)

#print(B)

"""


#solver = DirectCholeskySolver()
#solver = GMRESSolver()
#solver = BICGSolver()
#solver = BICGStabSolver()
#solver = CGSolver()


solver.set_operator(A)


x = X.random()
y = A(x)
z = solver(y)
print(np.linalg.norm(x-z) / np.linalg.norm(x))

"""



  
