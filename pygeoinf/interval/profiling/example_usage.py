"""
Example usage of the dependency resolution system for PLI profiling.
"""

import numpy as np
from pli_profiling import run_property_posterior, run_model_posterior, run_full_analysis
from pygeoinf.interval.functions import Function
from pygeoinf.linear_solvers import CholeskySolver


def create_example_parameters():
    """Create a complete parameter set for testing."""

    # Create example true model function
    def m_bar_callable(x):
        return np.exp(-((x - 0.5)/0.5)**2) * np.sin(5 * np.pi * x) + x

    # Create example prior mean function
    def m_0_callable(x):
        return x

    # We'll need to create the spaces first to make proper Function objects
    # For now, we'll use callables and let the system handle the conversion

    parameters = {
        # Spatial setup
        'N': 30,
        'N_d': 100,
        'N_p': 10,
        'endpoints': (0, 1),
        'basis_type': 'sine',

        # Mapping setup
        'integration_method_G': 'trapz',
        'integration_method_T': 'trapz',
        'n_points_G': 1000,
        'n_points_T': 1000,

        # Truth and measurement setup
        'true_data_noise': 0.1,
        'assumed_data_noise': 0.1,

        # Prior setup
        'alpha': 0.1,
        'K': 500,

        # Solver
        'solver': CholeskySolver(),
    }

    return parameters, m_bar_callable, m_0_callable


def example_property_posterior():
    """Example: Run only property posterior computation."""
    print("=== Property Posterior Example ===")

    parameters, m_bar_callable, m_0_callable = create_example_parameters()

    # We need to handle the function objects specially since they need spaces
    # This is a limitation of the current system - we'll address it below

    print("Parameters needed for property posterior:")
    for key, value in parameters.items():
        print(f"  {key}: {value}")

    print("\nNote: m_bar and m_0 functions need to be created after spaces are set up.")
    print("See the enhanced version below for a complete solution.")


def example_with_function_creation():
    """Example showing how to handle Function objects properly."""
    print("\n=== Enhanced Example with Function Creation ===")

    # Step 1: Create basic parameters
    parameters, m_bar_callable, m_0_callable = create_example_parameters()

    # Step 2: We need a way to inject Function creation into the pipeline
    # This suggests we need to enhance the system slightly

    print("This demonstrates the need for a function factory pattern")
    print("to handle Function objects that depend on spaces.")


if __name__ == "__main__":
    example_property_posterior()
    example_with_function_creation()

    print("\nTo use the system:")
    print("1. Import: from pli_profiling import run_property_posterior")
    print("2. Create parameter dict with all needed values")
    print("3. Call: results = run_property_posterior(parameters)")
    print("4. The system will automatically run all dependencies!")
