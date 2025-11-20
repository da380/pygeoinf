"""Hat function provider for finite element methods."""

import numpy as np
from .base import IndexedFunctionProvider


class HatFunctionProvider(IndexedFunctionProvider):
    """
    Provider for hat functions (piecewise linear basis functions).

    Hat functions are continuous, piecewise linear functions that form
    a basis for finite element methods. Each function is 1 at one node
    and 0 at all other nodes.

    For homogeneous hat functions, the boundary nodes are omitted,
    satisfying homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, space, homogeneous=False, n_nodes=None):
        """
        Initialize the hat function provider.

        Args:
            space: Lebesgue instance (contains domain information)
            homogeneous: If True, omit boundary nodes (homogeneous Dirichlet)
            n_nodes: Number of nodes. If None, uses space.dim + boundary
                     adjustment
        """
        super().__init__(space)
        self._cache = {}
        self.homogeneous = homogeneous

        # Determine number of nodes
        if n_nodes is None:
            if homogeneous:
                # For homogeneous: space.dim interior nodes + 2 boundary nodes
                self.n_nodes = self.space.dim + 2
            else:
                # For non-homogeneous: space.dim total nodes
                # (including boundary)
                self.n_nodes = self.space.dim
        else:
            self.n_nodes = n_nodes

        # Create node coordinates
        a, b = self.space.function_domain.a, self.space.function_domain.b
        self.nodes = np.linspace(a, b, self.n_nodes)
        self.h = (b - a) / (self.n_nodes - 1)  # Node spacing

    def get_function_by_index(self, index: int):
        """
        Get hat function for given index.

        Args:
            index: Index of the hat function

        Returns:
            Function: Hat function that is 1 at node[effective_index] and 0
                      elsewhere
        """
        if index not in self._cache:
            # Determine which node this function corresponds to
            if self.homogeneous:
                # Skip first boundary node: effective_index = index + 1
                effective_index = index + 1
                node_position = self.nodes[effective_index]
            else:
                effective_index = index
                node_position = self.nodes[effective_index]

            def hat_func(x):
                """Piecewise linear hat function."""
                x = np.asarray(x)
                result = np.zeros_like(x, dtype=float)

                # Hat function is non-zero only in [x_{i-1}, x_{i+1}]
                left_node = effective_index - 1
                right_node = effective_index + 1

                if left_node >= 0:
                    # Left piece: linear from 0 to 1
                    left_x = self.nodes[left_node]
                    mask_left = (x >= left_x) & (x <= node_position)
                    if np.any(mask_left):
                        result[mask_left] = (x[mask_left] - left_x) / self.h

                if right_node < self.n_nodes:
                    # Right piece: linear from 1 to 0
                    right_x = self.nodes[right_node]
                    mask_right = (x >= node_position) & (x <= right_x)
                    if np.any(mask_right):
                        result[mask_right] = (right_x - x[mask_right]) / self.h

                # Handle the case where x is exactly at the node
                mask_exact = np.isclose(
                    x, node_position, rtol=1e-14, atol=1e-14
                )
                result[mask_exact] = 1.0

                return result

            from ..functions import Function

            # Create function name
            if self.homogeneous:
                name = f"hat_hom_{index}(x={node_position:.3f})"
            else:
                name = f"hat_{index}(x={node_position:.3f})"

            func = Function(
                self.space,
                evaluate_callable=hat_func,
                name=name
            )

            self._cache[index] = func

        return self._cache[index]

    def get_nodes(self):
        """
        Get the node coordinates.

        Returns:
            np.ndarray: Array of node coordinates
        """
        return self.nodes.copy()

    def get_active_nodes(self):
        """
        Get the coordinates of nodes corresponding to basis functions.

        For homogeneous hat functions, this excludes boundary nodes.

        Returns:
            np.ndarray: Array of active node coordinates
        """
        if self.homogeneous:
            return self.nodes[1:-1].copy()  # Exclude boundary nodes
        else:
            return self.nodes.copy()  # All nodes are active
