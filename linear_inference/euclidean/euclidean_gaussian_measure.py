if __name__ == "__main__":
    pass


from linear_inference.gaussian_measure import GaussianMeasure
from linear_inference.euclidean import EuclideanSpace
from linear_inference.vector_space import LinearOperator
from scipy.stats import multivariate_normal

# Returns a gaussian measure on standard euclidean space. A covariance matrix
# is provided, and optionally a mean value. 
class EuclideanGaussianMeasure(GaussianMeasure):

    def __init__(self, cov, / , *, mean = None):
        m,n = cov.shape
        assert m == n 
        domain = EuclideanSpace(n)
        covariance = LinearOperator.self_adjoint_operator(domain, lambda x : cov @ x)
        if mean is None:
            dist = multivariate_normal(cov = cov)
            mean_ = None
        else:
            dist = multivariate_normal(mean = mean, cov = cov)
            mean_ = lambda : mean
        sample = lambda  : dist.rvs()
        super(EuclideanGaussianMeasure,self).__init__(covariance, mean = mean, sample = sample)

        


