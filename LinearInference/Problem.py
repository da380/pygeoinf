from LinearInference.Hilbert import Space
from LinearInference.Operator import Linear, SelfAdjoint
from LinearInference.Gaussian import Measure

'''
Class for computing the Bayesian solution of an linear inverse problem with Gaussian priors and errors. 
To form an instance, the user must provide:

priorModelMeasure: The prior model measure as an instance of the class Gaussian.Measure. Information 
                        on the model space is contained within this measure. 

errorMeasure:      The measure for data errors as an instance of the class Gaussian.Measure. Information 
                   on the data space is contained within this measure. 

forwardOperator:   The forward operator as an instance of the class Operator.Linear. The adjoint of the 
                   forward operator is accessed through this operator. 

solver:            An instance of the class Operator.Linear that implements the mapping (A Q A* + R)^{-1}
                   where A is the forward operator, A* its adjoint, Q the prior model covariance, and 
                   R the data covariance. 

'''
class InverseProblem:

    def __init__(self, priorModelMeasure, errorMeasure, forwardOperator, solver):
        assert forwardOperator.Domain == priorModelMeasure.Space
        assert forwardOperator.CoDomain == errorMeasure.Space
        self._priorModelMeasure = priorModelMeasure
        self._errorMeasure = errorMeasure
        self._forwardOperator = forwardOperator
        self._solver = solver
        
    
    @property
    def PriorModelMeasure(self):
        return self._priorModelMeasure

    @property
    def ModelSpace(self):
        return self.PriorModelMeasure.Space

    @property
    def ErrorMeasure(self):
        return self._errorMeasure

    @property
    def DataSpace(self):
        return self.ErrorMeasure.Space

    @property
    def ForwardOperator(self):
        return self._forwardOperator

    @property
    def Solver(self):
        return self._solver

    # Return the prior data Measure.
    @property
    def PriorDataMeasure(self):
        return self.PriorModelMeasure.Affine(self.ForwardOperator) + self.ErrorMeasure

    # Return synthetic data from a given model.
    def SyntheticData(self,model):
        return self.ForwardOperator(model) + self.ErrorMeasure.Sample()

    # Return synthetic data from a random model.
    def RandomSyntheticData(self):
        return self.PriorDataMeasure.Sample()

    # Return the posterior model covariance operator. 
    def PosteriorModelCovariance(self):
        Q = self.PriorMeasure.Covariance
        A = self.ForwardOperator
        As = A.Adjoint
        R = self.ErrorMeasure.Covariance
        T = self.Solver
        return Q - Q @ As @ T @ A @ Q

    # Return the posterior model mean given data. 
    def PosteriorModelMean(self, data):
        model0 = self.PriorModelMeasure.Mean
        data0 = self.PriorDataMeasure.Mean
        Q = self.PriorMeasure.Covariance
        A = self.ForwardOperator
        As = A.Adjoint
        R = self.ErrorMeasure.Covariance
        T = self.Solver
        return model0 + (Q @ As @ T)(data - data0)



'''
Class for computing the Bayesian solution of an linear inference problem with Gaussian priors and errors. 
To form an instance, the user must provide:

priorModelMeasure: The prior model measure as an instance of the class Gaussian.Measure. Information 
                        on the model space is contained within this measure. 

errorMeasure:      The measure for data errors as an instance of the class Gaussian.Measure. Information 
                   on the data space is contained within this measure. 

forwardOperator:   The forward operator as an instance of the class Operator.Linear.

propertyOperator:  The property operator as an instance of the class Operator.Linear. Information on the 
                   property space is contained within this operator. 

solver:            An instance of the class Operator.Linear that implements the mapping (A Q A* + R)^{-1}
                   where A is the forward operator, A* its adjoint, Q the prior model covariance, and 
                   R the data covariance. 

'''
class InferenceProblem(InverseProblem):

    def __init__(self, priorModelMeasure, errorMeasure, forwardOperator, solver, propertyOperator):    
        super(InferenceProblem,self).__init__(priorModelMeasure, errorMeasure, forwardOperator, solver)
        assert propertyOperator.Domain == self.ModelSpace
        self._propertyOperator == propertyOperator


    @property
    def PropertyOperator(self):
        return self._propertyOperator

    @property
    def PropertySpace(self):
        return self.PropertyOperator.CoDomain

    # Return prior property measure. 
    @property
    def PriorPropertyMeasure(self):
        return self.PriorModelMeasure.Affine(self.PropertyOperator)

    # Return posterior property covariance. 
    @ property
    def PosteriorPropertyCovariane(self):
        B = self.PropertyOperator
        return B @ self.PosteriorModelCovariance @ B.Adjoint

    # Return posterior property mean given data.
    def PosteriorPropertyMean(self,data):
        return self.PropertyOperator(self.PosteriorModelMean(data))












    





    