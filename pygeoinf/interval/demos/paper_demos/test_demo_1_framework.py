#!/usr/bin/env python
"""
Quick test script to verify the Demo 1 sweep framework installation.

This script runs a minimal experiment to ensure all components are
properly installed and working.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from demo_1_config import (
            Demo1Config, get_fast_config, get_standard_config
        )
        from demo_1_experiment import Demo1Experiment, run_single_experiment
        from demo_1_sweep import Demo1Sweep, create_noise_sensitivity_sweep
        from demo_1_analysis import load_sweep_results, plot_parameter_sensitivity
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration creation and validation."""
    print("\nTesting configuration...")
    try:
        from demo_1_config import Demo1Config, get_fast_config

        # Create default config
        config = Demo1Config()
        assert config.N == 100
        assert config.s_vp == 2.0

        # Test derived properties
        assert config.k_vp == 1.0 / config.length_scale_vp

        # Test pre-defined configs
        fast_config = get_fast_config()
        assert fast_config.N == 50

        # Test copy
        modified = config.copy(N=150)
        assert modified.N == 150
        assert config.N == 100  # Original unchanged

        # Test validation
        try:
            bad_config = Demo1Config(s_vp=0.1)  # Should fail (s < 0.5)
            print("✗ Validation failed to catch invalid s_vp")
            return False
        except ValueError:
            pass  # Expected

        print("✓ Configuration tests passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_experiment_setup():
    """Test that an experiment can be set up (without running)."""
    print("\nTesting experiment setup...")
    try:
        from demo_1_config import get_fast_config
        from demo_1_experiment import Demo1Experiment
        import tempfile

        config = get_fast_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment = Demo1Experiment(config, Path(tmpdir))

            # Test setup phases
            experiment.setup_model_space()
            assert experiment.M_model is not None
            assert len(experiment.M_model.spaces) == 2  # [[functions], [scalars]]

            experiment.create_operators()
            assert experiment.G is not None
            assert experiment.T is not None

            experiment.generate_data()
            assert experiment.u_true is not None
            assert experiment.d_obs is not None

            print("✓ Experiment setup tests passed")
            return True
    except Exception as e:
        print(f"✗ Experiment setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sweep_creation():
    """Test that a sweep can be created."""
    print("\nTesting sweep creation...")
    try:
        from demo_1_config import get_fast_config
        from demo_1_sweep import Demo1Sweep
        import tempfile

        config = get_fast_config()
        sweep_params = {
            's_vp': [1.5, 2.0],
            'noise_level': [0.01, 0.05]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep = Demo1Sweep(
                base_config=config,
                sweep_params=sweep_params,
                output_dir=Path(tmpdir),
                name="test_sweep",
                parallel=False
            )

            # Check parameter combinations
            assert len(sweep.param_combinations) == 4  # 2 × 2

            # Check that configs can be created
            test_config = sweep._create_config({'s_vp': 2.0, 'noise_level': 0.01}, 0)
            assert test_config.s_vp == 2.0
            assert test_config.noise_level == 0.01

            print("✓ Sweep creation tests passed")
            return True
    except Exception as e:
        print(f"✗ Sweep creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_minimal_experiment():
    """Run a complete minimal experiment."""
    print("\nTesting full minimal experiment...")
    print("(This may take 30-60 seconds...)")
    try:
        from demo_1_config import get_fast_config
        from demo_1_experiment import run_single_experiment
        import tempfile

        # Use very small configuration for speed
        config = get_fast_config()
        config.N = 20  # Very small
        config.N_d = 10
        config.N_p = 5
        config.integration_N = 50
        config.kl_truncation_vp = 10
        config.kl_truncation_vs = 10
        config.kl_truncation_rho = 10
        config.parallel = False

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_single_experiment(config, Path(tmpdir) / "test_run")

            # Check that results contain expected keys
            assert 'metrics' in results
            assert 'timings' in results
            assert 'config' in results

            # Check metrics
            assert 'vp_rel_l2_error' in results['metrics']
            assert 'vs_rel_l2_error' in results['metrics']
            assert 'rho_rel_l2_error' in results['metrics']

            # Check timings
            assert 'total' in results['timings']

            # Check files were created
            output_dir = Path(tmpdir) / "test_run"
            assert (output_dir / 'config.json').exists()
            assert (output_dir / 'metrics.json').exists()
            assert (output_dir / 'timings.json').exists()
            assert (output_dir / 'figures').exists()

            print("✓ Full experiment test passed")
            print(f"  Metrics: vp_error={results['metrics']['vp_rel_l2_error']:.4f}")
            print(f"  Time: {results['timings']['total']:.2f}s")
            return True
    except Exception as e:
        print(f"✗ Full experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Demo 1 Sweep Framework - Installation Test")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Experiment Setup", test_experiment_setup),
        ("Sweep Creation", test_sweep_creation),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))

    # Optional: run full experiment test
    print("\n" + "="*60)
    response = input("Run full minimal experiment test? (yes/no): ")
    if response.lower() == 'yes':
        try:
            passed = test_full_minimal_experiment()
            results.append(("Full Experiment", passed))
        except Exception as e:
            print(f"✗ Full experiment crashed: {e}")
            results.append(("Full Experiment", False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(p for _, p in results)
    print("="*60)
    if all_passed:
        print("All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("  python demo_1_example_sweep.py 9  # Run fast example")
        print("  python demo_1_example_sweep.py 1  # Run single experiment")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
