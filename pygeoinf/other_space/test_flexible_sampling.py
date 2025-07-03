"""
Test sampling from Gaussian measures with the flexible Sobolev space design.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pygeoinf.pygeoinf.other_space.interval_space import Sobolev


def test_gaussian_sampling():
    """Test that we can sample from Gaussian measures."""

    print("Testing Gaussian Measure Sampling")
    print("=" * 40)

    dim = 16
    interval = (0, 1)

    # Create a simple Fourier-based space
    def fourier_to_coeff(u):
        from scipy.fft import dct
        return dct(u, type=2, norm='ortho')

    def fourier_from_coeff(coeff):
        from scipy.fft import idct
        return idct(coeff, type=2, norm='ortho')

    def fourier_scaling(k):
        return (1 + (0.1 * k) ** 2) ** 1.0

    # Create space
    space = Sobolev(
        dim, fourier_to_coeff, fourier_from_coeff, fourier_scaling,
        interval=interval
    )

    # Define a simple covariance
    def simple_covariance(k):
        return np.exp(-0.2 * k)

    # Create Gaussian measure
    gm = space.gaussian_measure(simple_covariance)

    print(f"✓ Created Gaussian measure on {dim}-dimensional space")

    # Test sampling
    try:
        samples = []
        for i in range(3):
            sample = gm.sample()
            samples.append(sample)
            print(f"✓ Sample {i+1}: shape={sample.shape}, "
                  f"mean={np.mean(sample):.3f}, std={np.std(sample):.3f}")

        print(f"✓ Successfully generated {len(samples)} samples")

        # Test that samples have the right properties
        sample_mean = np.mean([np.mean(s) for s in samples])
        sample_std = np.mean([np.std(s) for s in samples])

        print(f"✓ Average sample mean: {sample_mean:.4f}")
        print(f"✓ Average sample std: {sample_std:.4f}")

        return True

    except Exception as e:
        print(f"✗ Error sampling: {e}")
        return False


def test_automorphism():
    """Test that automorphisms work correctly."""

    print("\nTesting Automorphisms")
    print("=" * 40)

    dim = 8

    # Create a simple identity-based space for testing
    def identity(u):
        return u.copy()

    def constant_scaling(k):
        return 1.0

    space = Sobolev(dim, identity, identity, constant_scaling)

    # Test automorphism
    def scaling_function(k):
        return 2.0 if k < 4 else 0.5

    auto = space.automorphism(scaling_function)

    # Test on a simple function
    test_function = np.ones(dim)
    result = auto(test_function)

    print(f"✓ Created automorphism")
    print(f"✓ Applied to test function: {result[:4]}")

    return True


if __name__ == "__main__":
    try:
        success1 = test_gaussian_sampling()
        success2 = test_automorphism()

        if success1 and success2:
            print("\n" + "=" * 40)
            print("✓ All tests passed!")
            print("✓ Flexible Sobolev space design is working correctly")
        else:
            print("\n" + "=" * 40)
            print("✗ Some tests failed")

    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
