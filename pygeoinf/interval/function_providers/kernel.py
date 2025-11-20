"""Kernel provider for loading kernel data from files."""

import os
import numpy as np
from typing import Optional
from scipy.interpolate import interp1d
from .base import IndexedFunctionProvider


class KernelProvider(IndexedFunctionProvider):
    """
    Provider for kernel functions loaded from data files.

    This provider reads kernel sensitivity data from files and creates
    interpolated Function objects. Useful for loading pre-computed
    sensitivity kernels.
    """

    def __init__(self, space, kernel_type: str = "rho",
                 kernel_data_dir: Optional[str] = None):
        """
        Initialize kernel provider.

        Args:
            space: Lebesgue instance (contains domain information)
            kernel_type: Type of kernel (e.g., "rho", "vpv", "vph")
            kernel_data_dir: Directory containing kernel data files
        """
        super().__init__(space)
        self._kernel_type = kernel_type
        self._data_dir = kernel_data_dir
        self._data_list = self._get_data_list()

    def get_function_by_index(self, index: int):
        """
        Get kernel function by index from data files.

        Args:
            index: Index of the kernel function

        Returns:
            Function: Interpolated kernel function
        """
        from ..functions import Function

        if self._data_dir is None:
            raise ValueError("kernel_data_dir must be provided")
        if index < 0 or index >= len(self._data_list):
            raise IndexError(
                f"Invalid index. Maximum is {len(self._data_list) - 1}"
            )
        mode = self._data_list[index]
        filename = f"{self._kernel_type}-sens_{mode}_iso.dat"
        filepath = os.path.join(self._data_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Kernel file not found: {filepath}")
        data = np.loadtxt(filepath)
        radius = data[:, 0]
        values = data[:, 1]

        interp_func = interp1d(
            radius, values, bounds_error=False, fill_value=0.0
        )
        return Function(self.space, evaluate_callable=interp_func,
                        name=f"kernel_{self._kernel_type}_{mode}")

    def _get_data_list(self):
        """Get a list of all kernel data files."""
        if self._data_dir is None:
            return []
        filepath = os.path.join(self._data_dir, 'data_list_SP12RTS')
        if not os.path.isfile(filepath):
            return []
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
