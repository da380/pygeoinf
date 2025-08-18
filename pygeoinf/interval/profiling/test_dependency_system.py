"""
Working example of the dependency resolution system.
"""

import numpy as np
from pygeoinf.linear_solvers import CholeskySolver

# Assuming we can import the functions (fix import issues first)
# from pli_profiling import run_property_posterior


def create_test_parameters():
    """Create a minimal working parameter set."""

    parameters = {
        # Spatial setup
        'N': 10,  # Smaller for testing
        'N_d': 20,
        'N_p': 5,
        'endpoints': (0, 1),
        'basis_type': 'sine',

        # Mapping setup
        'integration_method_G': 'trapz',
        'integration_method_T': 'trapz',
        'n_points_G': 100,  # Smaller for testing
        'n_points_T': 100,

        # Truth and measurement setup (using callables)
        'm_bar_callable': lambda x: np.sin(2 * np.pi * x),
        'true_data_noise': 0.05,
        'assumed_data_noise': 0.05,

        # Prior setup (using callables)
        'alpha': 0.1,
        'K': 50,  # Smaller for testing
        'm_0_callable': lambda x: np.zeros_like(x),

        # Solver
        'solver': CholeskySolver(),
    }

    return parameters


def test_dependency_system():
    """Test the dependency resolution system."""

    print("=== Testing Dependency Resolution System ===")

    parameters = create_test_parameters()

    print("Created parameters:")
    for key, value in parameters.items():
        if callable(value):
            print(f"  {key}: <callable>")
        else:
            print(f"  {key}: {value}")

    print("\nTo test the system:")
    print("1. Fix any remaining import issues in pli_profiling.py")
    print("2. Run: results = run_property_posterior(parameters)")
    print("3. The system should automatically execute:")
    print("   - setup_spatial_spaces")
    print("   - setup_mappings")
    print("   - _setup_truths_and_measurement")
    print("   - setup_prior_measure")
    print("   - create_problems")
    print("   - compute_property_posterior")

    print("\nExpected output: Dict with timing results from property posterior computation")


if __name__ == "__main__":
    test_dependency_system()
