"""
A module defining the base architecture for dynamical systems on Hilbert spaces.
"""

from abc import ABC, abstractmethod

from .hilbert_space import HilbertSpace
from .nonlinear_operators import NonLinearOperator
from .linear_operators import LinearOperator


class DynamicalSystem(ABC):
    """
    Base abstract class for a dynamical system on a Hilbert space.
    Represents the generally non-linear, non-autonomous system du/dt = F(t, u).
    """

    def __init__(self, state_space: HilbertSpace):
        self.state_space = state_space

    @property
    def is_autonomous(self) -> bool:
        """
        Indicates if the system's governing operator is time-independent.
        Defaults to False for a general dynamical system.
        """
        return False

    @abstractmethod
    def dynamical_rule(self, t: float) -> NonLinearOperator:
        """
        Returns the non-linear operator F(t, .) mapping the state to its derivative.

        Args:
            t: The current time.

        Returns:
            A pygeoinf NonLinearOperator mapping state_space to state_space.
        """
        pass


class AutonomousDynamicalSystem(DynamicalSystem):
    """
    A concrete implementation of a non-linear, autonomous system.
    Represents du/dt = F(u), where the operator F does not change with time.
    """

    def __init__(self, state_space: HilbertSpace, operator: NonLinearOperator):
        super().__init__(state_space)
        self._operator = operator

    @property
    def is_autonomous(self) -> bool:
        return True

    def dynamical_rule(self, t: float) -> NonLinearOperator:
        """
        Returns the static non-linear operator.

        Args:
            t: The current time (ignored, but kept for interface compliance).
        """
        return self._operator


class LinearDynamicalSystem(DynamicalSystem):
    """
    Abstract base class for a linear, non-autonomous dynamical system.
    Represents du/dt = L(t)u.

    This enforces a stricter return type (LinearOperator) so integrators
    can optimize Jacobian evaluations and matrix operations.
    """

    @abstractmethod
    def dynamical_rule(self, t: float) -> LinearOperator:
        """
        Returns the linear operator L(t) mapping the state to its derivative.

        Args:
            t: The current time.

        Returns:
            A pygeoinf LinearOperator mapping state_space to state_space.
        """
        pass


class AutonomousLinearSystem(LinearDynamicalSystem):
    """
    A concrete implementation of a linear, autonomous system.
    Represents du/dt = Lu, where the linear operator L is constant.
    """

    def __init__(self, state_space: HilbertSpace, operator: LinearOperator):
        super().__init__(state_space)
        self._operator = operator

    @property
    def is_autonomous(self) -> bool:
        return True

    def dynamical_rule(self, t: float) -> LinearOperator:
        """
        Returns the static linear operator.

        Args:
            t: The current time (ignored, but kept for interface compliance).
        """
        return self._operator
