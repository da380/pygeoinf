if __name__ == "__main__":
    pass

from linear_inference.vector_space import HilbertSpace, LinearOperator
from linear_inference.gaussian_measure import GaussianMeasure


class ForwardProblem:

    def __init__(self, forward_operator, /, *, error_measure = None):
        assert forward_operator.codomain == error_measure.domain
        self._forward_operator = forward_operator
        self._error_measure = error_measure

    # Return the forward operator. 
    @property
    def forward_operator(self):
        return self._forward_operator

    # Return the model space. 
    @property
    def model_space(self):
        return self.forward_operator.domain

    # Return the data space.
    @property
    def data_space(self):
        return self.forward_operator.codomain

    # Returns true is error measure has been set. 
    @property
    def error_measure_set(self):
        return self._error_measure is not None

    # Return the induced measure on the data space for given model. 
    @property
    def data_measure(self, model):     
        assert self.error_measure_set   
        return self._error_measure(translation = self.forward_operator(model))

    # Return synthetic data for given model. By default, a sample from the error_distribution
    # is included if this has been defined. 
    def synthetic_data(self, model, /, *, include_data_errors = True):
        if self.error_measure_set and include_data_errors:
            return self.data_measure(model).sample()
        else:
            return self.forward_operator(model)


        

