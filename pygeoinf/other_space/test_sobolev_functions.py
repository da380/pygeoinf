#!/usr/bin/env python3
"""
Test script to verify the create_standard_sobolev method creates SobolevFunction instances.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygeoinf.other_space.interval_space import Sobolev
from pygeoinf.other_space.sobolev_functions import SobolevFunction

def test_create_standard_sobolev():
    """Test that create_standard_sobolev creates SobolevFunction instances."""

    # Test parameters
    order = 1.0
    scale = 1.0
    dim = 5
    interval = (0, 1)

    print("Testing create_standard_sobolev method...")

    # Test different basis types
    for basis_type in ['fourier', 'sine', 'chebyshev']:
        print(f"\nTesting basis type: {basis_type}")

        try:
            # Create the space
            space = Sobolev.create_standard_sobolev(
                order, scale, dim, interval=interval, basis_type=basis_type
            )

            print(f"  Space created successfully")
            print(f"  Space dimension: {space.dim}")
            print(f"  Space interval: {space.interval}")
            print(f"  Space order: {space.order}")

            # Try to get basis functions
            try:
                basis_functions = space.get_basis_functions()
                print(f"  Number of basis functions: {len(basis_functions)}")

                # Check that they are SobolevFunction instances
                for i, func in enumerate(basis_functions):
                    if not isinstance(func, SobolevFunction):
                        print(f"  ERROR: Basis function {i} is not a SobolevFunction instance")
                        return False
                    else:
                        print(f"  Basis function {i}: {func.name}")

                        # Test evaluation at a point (if s > 0.5)
                        if order > 0.5:
                            try:
                                test_point = 0.5
                                value = func.evaluate(test_point)
                                print(f"    f({test_point}) = {value}")
                            except Exception as e:
                                print(f"    Warning: Could not evaluate at {test_point}: {e}")

                print(f"  SUCCESS: All basis functions are SobolevFunction instances")

            except Exception as e:
                print(f"  ERROR: Could not get basis functions: {e}")
                return False

        except Exception as e:
            print(f"  ERROR: Could not create space: {e}")
            return False

    return True

if __name__ == "__main__":
    success = test_create_standard_sobolev()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
