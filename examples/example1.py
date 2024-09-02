import numpy as np
from scipy.stats import norm
from linear_inference.euclidean import L2, GaussianMeasure


X = L2(10)

covariance = np.identity(X.dimension)

mu = GaussianMeasure(X, covariance)

print(mu.mean)


'''
# Set up the first Hilbert space. 
m = 3
gX = norm.rvs(size = (m,m))
gX = gX.T @ gX + 0.1 * np.identity(m)
X = HilbertSpace(m, lambda x : x, lambda x : x,  (lambda x1, x2, : np.dot(gX @ x1, x2)))

# Set up the second Hilbert space. 
n = 2
gY = norm.rvs(size = (n,n))
gY = gY.T @ gY + 0.1 * np.identity(n)
Y = HilbertSpace(n, lambda x : x, lambda x : x, (lambda y1, y2, : np.dot(gY @ y1, y2)) )

# Define the linear mapping between the two. 
mapping = lambda x : x[:n]
dual_mapping = lambda yp : (lambda x : yp(x[:n]))
A = LinearOperator(X, Y, mapping, dual_mapping= dual_mapping)

# Check the adjoint identity. 
x = X.random()
y = Y.random()
print(Y.inner_product(y, A(x)))
print(X.inner_product(A.adjoint(y), x))

'''
















    


























































