import numpy as np
from scipy.stats import norm

from linear_inference.vector_space import VectorSpace, LinearForm, LinearOperator
from linear_inference.hilbert_space import HilbertSpace



dimension = 4

A = norm.rvs(size = (dimension, dimension))
A = A.T @ A
to_components = lambda x : x
from_components = lambda x : x
inner_product = lambda x1 , x2 : np.dot(A @ x1,x2)


X = HilbertSpace(dimension, to_components, from_components, inner_product)

xp = LinearForm(X, lambda x : x[0])

x = X.from_dual(xp)

yp = X.to_dual(x)

print(yp)



    


























































