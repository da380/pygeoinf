if __name__ == "__main__":
    pass


from linear_inference.gaussian_measure import GaussianMeasure
from linear_inference.euclidean import EuclideanSpace
from linear_inference.vector_space import LinearOperator
from scipy.stats import multivariate_normal

class EuclideanGaussianMeasure(GaussianMeasure):

    def __init__(self, domain, cov, / , *, xbar = None):
        assert isinstance(domain, EuclideanSpace)
        covariance = LinearOperator.self_adjoint_operator(domain, lambda x : cov @ x)
        if xbar is None:
            dist = multivariate_normal(cov = cov)
            mean = None
        else:
            dist = multivariate_normal(mean = xbar, cov = cov)
            mean = lambda : xbar
        sample = lambda  : dist.rvs()
        super(EuclideanGaussianMeasure,self).__init__(covariance, mean = mean, sample = sample)

        


