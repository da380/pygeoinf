"""
Boundary condition specifications for function spaces on intervals.

This module provides boundary condition classes and utilities for L² spaces,
Sobolev spaces, and FEM solvers on interval domains.
"""


class BoundaryConditions:
    """
    Boundary condition specifications for function spaces on intervals.

    This class provides a unified interface for all boundary condition types
    used across L² spaces, Sobolev spaces, and FEM solvers.
    """

    def __init__(self, bc_type: str, **kwargs):
        """
        Initialize boundary conditions.

        Args:
            bc_type: Type of boundary condition. One of:
                - 'dirichlet':
                    left (float, optional): Value at left boundary
                        (default 0.0)
                    right (float, optional): Value at right boundary
                        (default 0.0)
                - 'neumann':
                    left (float, optional): Derivative at left boundary
                        (default 0.0)
                    right (float, optional): Derivative at right boundary
                        (default 0.0)
                - 'robin':
                    left_alpha (float): Coefficient for u(a)
                    left_beta (float): Coefficient for u'(a)
                    left_value (float): Value at left boundary
                    right_alpha (float): Coefficient for u(b)
                    right_beta (float): Coefficient for u'(b)
                    right_value (float): Value at right boundary
                - 'periodic':
                    (no additional parameters)
            **kwargs: See above for valid keyword arguments for each type.
        """
        self.type = bc_type
        self._params = kwargs
        self._validate()

    def _validate(self):
        """Validate boundary condition parameters."""
        valid_types = {
            'dirichlet', 'neumann', 'robin', 'periodic',
            'mixed_dirichlet_neumann', 'mixed_neumann_dirichlet'
        }

        if self.type not in valid_types:
            raise ValueError(
                f"Invalid boundary condition type '{self.type}'. "
                f"Valid types: {valid_types}"
            )

        # Type-specific validation
        if self.type == 'dirichlet':
            # Default values if not provided
            self._params.setdefault('left', 0.0)
            self._params.setdefault('right', 0.0)

        elif self.type == 'neumann':
            self._params.setdefault('left', 0.0)
            self._params.setdefault('right', 0.0)

        elif self.type == 'robin':
            required = [
                'left_alpha', 'left_beta', 'left_value',
                'right_alpha', 'right_beta', 'right_value'
            ]
            for param in required:
                if param not in self._params:
                    raise ValueError(
                        f"Robin boundary conditions require '{param}'"
                    )

        elif self.type == 'periodic':
            # No additional parameters needed
            pass

        elif self.type == 'mixed_dirichlet_neumann':
            self._params.setdefault('left', 0.0)  # Dirichlet at left
            self._params.setdefault('right', 0.0)  # Neumann at right

        elif self.type == 'mixed_neumann_dirichlet':
            self._params.setdefault('left', 0.0)  # Neumann at left
            self._params.setdefault('right', 0.0)  # Dirichlet at right

    @property
    def is_homogeneous(self) -> bool:
        """Check if boundary conditions are homogeneous."""
        if self.type == 'dirichlet':
            return (self._params.get('left', 0) == 0 and
                    self._params.get('right', 0) == 0)
        elif self.type == 'neumann':
            return (self._params.get('left', 0) == 0 and
                    self._params.get('right', 0) == 0)
        elif self.type == 'periodic':
            return True  # Periodic BCs are considered homogeneous
        elif self.type == 'mixed_dirichlet_neumann':
            return (self._params.get('left', 0) == 0 and
                    self._params.get('right', 0) == 0)
        elif self.type == 'mixed_neumann_dirichlet':
            return (self._params.get('left', 0) == 0 and
                    self._params.get('right', 0) == 0)
        elif self.type == 'robin':
            return (self._params.get('left_value', 0) == 0 and
                    self._params.get('right_value', 0) == 0)
        else:
            return False

    def get_parameter(self, name: str, default=None):
        """Get a boundary condition parameter."""
        return self._params.get(name, default)

    @classmethod
    def dirichlet(cls, left_value: float = 0,
                  right_value: float = 0) -> 'BoundaryConditions':
        """
        Dirichlet boundary conditions: u(a) = left_value, u(b) = right_value.

        Args:
            left_value: Value at left boundary
            right_value: Value at right boundary
        """
        return cls('dirichlet', left=left_value, right=right_value)

    @classmethod
    def neumann(cls, left_derivative: float = 0,
                right_derivative: float = 0) -> 'BoundaryConditions':
        """
        Neumann boundary conditions:
        u'(a) = left_derivative, u'(b) = right_derivative.

        Args:
            left_derivative: Derivative value at left boundary
            right_derivative: Derivative value at right boundary
        """
        return cls('neumann', left=left_derivative, right=right_derivative)

    @classmethod
    def robin(cls, left_alpha: float, left_beta: float, left_value: float,
              right_alpha: float, right_beta: float,
              right_value: float) -> 'BoundaryConditions':
        """
        Robin boundary conditions: αu + βu' = value at boundaries.

        Args:
            left_alpha, left_beta, left_value: Left boundary coefficients
            right_alpha, right_beta, right_value: Right boundary coefficients
        """
        return cls('robin',
                   left_alpha=left_alpha, left_beta=left_beta,
                   left_value=left_value, right_alpha=right_alpha,
                   right_beta=right_beta, right_value=right_value)

    @classmethod
    def periodic(cls) -> 'BoundaryConditions':
        """
        Periodic boundary conditions: u(a) = u(b), u'(a) = u'(b).
        """
        return cls('periodic')

    @classmethod
    def mixed_dirichlet_neumann(cls, left_value: float = 0,
                                right_derivative: float = 0) -> 'BoundaryConditions':
        """
        Mixed Dirichlet-Neumann boundary conditions:
        u(a) = left_value, u'(b) = right_derivative.

        Args:
            left_value: Value at left boundary
            right_derivative: Derivative value at right boundary
        """
        return cls('mixed_dirichlet_neumann', left=left_value, right=right_derivative)

    @classmethod
    def mixed_neumann_dirichlet(cls, left_derivative: float = 0,
                                right_value: float = 0) -> 'BoundaryConditions':
        """
        Mixed Neumann-Dirichlet boundary conditions:
        u'(a) = left_derivative, u(b) = right_value.

        Args:
            left_derivative: Derivative value at left boundary
            right_value: Value at right boundary
        """
        return cls('mixed_neumann_dirichlet', left=left_derivative, right=right_value)

    def __str__(self) -> str:
        """String representation."""
        if self.type == 'periodic':
            return f"{self.type}"
        else:
            params_str = ', '.join(f"{k}={v}" for k, v in self._params.items())
            return f"{self.type}({params_str})"

    def __repr__(self) -> str:
        """Representation."""
        return f"BoundaryConditions('{self.type}', {self._params})"

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if isinstance(other, BoundaryConditions):
            return (self.type == other.type and
                    self._params == other._params)
        return False
