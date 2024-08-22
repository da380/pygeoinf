from linear_inference.vector_space import LinearForm
from linear_inference.euclidean import EuclideanSpace

# Set up standard n-dimensional Euclidean space. 
n = 5
X = EuclideanSpace(n)


# Form a space of the same dimension but with a non-standard metric. 
Y = EuclideanSpace.with_random_metric(n)

# Generate two random vectors. 
x1 = X.random()
x2 = X.random()

# Print their inner products in X and Y
print(X.inner_product(x1,x2))
print(Y.inner_product(x1,x2))


# Define a linear form on the space
xp = LinearForm(X, mapping = lambda x : x[0] - 2 * x[2])

# Print its representation in X and in Y
x = X.from_dual(xp)
y = Y.from_dual(xp)

print(x)
print(y)

# Check that the representations work. 
print(xp(x1) - X.inner_product(x, x1))
print(xp(x1) - Y.inner_product(y, x1))





