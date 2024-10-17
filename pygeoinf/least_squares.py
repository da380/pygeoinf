import numpy as np
import pygeoinf.linalg as la


class LeastSquaresProblem:

    def __init__(self, forward_operator):
        self._forward_operator = forward_operator

    @property
    def model_space(self):
        return self._forward_operator.domain

    @property
    def data_space(self):
        return self._forward_operator.codomain

    def normal_operator(self, damping):
        return la.LinearOperator.self_adjoint(self.model_space, mapping=self._forward_operator.adjoint @ self._forward_operator + damping * self.model_space.identity())
        # return self.model_space.identity()

    def normal_solver(self, damping, solver):
        return solver(self.normal_operator(damping)) @ self._forward_operator.adjoint
