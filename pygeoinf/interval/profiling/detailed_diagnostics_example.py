"""
Example: Using Detailed Diagnostics to Trace Non-PD Covariance Issues

This script demonstrates how to use the new detailed diagnostic functions
to understand where positive-definiteness is lost in the PLI pipeline.
"""

import numpy as np
from robust_testing import run_custom_robustness_test, test_block
from pygeoinf.linear_solvers import LUSolver

def analyze_covariance_breakdown():
    """Analyze a problematic configuration step by step."""

    print("🔍 DETAILED COVARIANCE BREAKDOWN ANALYSIS")
    print("=" * 60)

    # Known problematic configuration
    problematic_config = {
        'N': 25, 'N_d': 50, 'N_p': 20,
        'endpoints': (0, 1), 'basis_type': 'sine',
        'integration_method_G': 'trapz', 'integration_method_T': 'trapz',
        'n_points_G': 200, 'n_points_T': 200,
        'alpha': 0.001,  # Very small regularization
        'K': 15, 'true_data_noise': 0.1, 'assumed_data_noise': 0.1,
        'm_bar_callable': lambda x: np.sin(2 * np.pi * x),
        'm_0_callable': lambda x: np.zeros_like(x),
        'solver': LUSolver()
    }

    print("\n📋 Configuration:")
    for key, value in problematic_config.items():
        if not callable(value):
            print(f"  {key}: {value}")

    print("\n🔬 STEP-BY-STEP ANALYSIS:")

    # Step 1: Test model posterior detailed
    print("\n1. Model Posterior Analysis:")
    result_model = test_block('compute_model_posterior_detailed', problematic_config)

    if result_model.success and result_model.results:
        model_diag = result_model.results['compute_model_posterior_detailed']

        print(f"   ✅ Model posterior computation: SUCCESS")
        print(f"   Normal operator PD: {model_diag.get('normal_op_is_positive_definite', 'N/A')}")
        print(f"   Normal operator min eig: {model_diag.get('normal_op_min_eigenvalue', 'N/A'):.2e}")
        print(f"   Model posterior PD: {model_diag.get('model_post_is_positive_definite', 'N/A')}")
        print(f"   Model posterior min eig: {model_diag.get('model_post_min_eigenvalue', 'N/A'):.2e}")
        print(f"   Solver time: {model_diag.get('solver_time', 'N/A'):.3f}s")

    else:
        print(f"   ❌ Model posterior computation: FAILED")
        print(f"   Error: {result_model.error_message}")
        return

    # Step 2: Test property posterior detailed
    print("\n2. Property Posterior Analysis:")
    result_property = test_block('compute_property_posterior_detailed', problematic_config)

    if result_property.success:
        print(f"   ✅ Property posterior computation: SUCCESS")
    else:
        prop_diag = result_property.results['compute_property_posterior_detailed'] if result_property.results else {}

        print(f"   ❌ Property posterior computation: FAILED")
        print(f"   Error: {result_property.error_message}")

        # Analyze where the failure occurred
        print(f"\n   🔍 Failure Analysis:")
        print(f"   Property operator rank: {prop_diag.get('property_operator_rank', 'N/A')}")
        print(f"   Property operator condition: {prop_diag.get('property_operator_condition', 'N/A'):.2e}")
        print(f"   Property operator shape: {prop_diag.get('property_operator_shape', 'N/A')}")
        print(f"   Property posterior min eig: {prop_diag.get('property_post_min_eigenvalue', 'N/A'):.2e}")

        # Check if explicit computation worked
        if prop_diag.get('explicit_computation_successful', False):
            print(f"   Explicit P @ C_model @ P^T min eig: {prop_diag.get('explicit_min_eigenvalue', 'N/A'):.2e}")
            print(f"   Difference between explicit and affine: {prop_diag.get('explicit_vs_affine_diff_norm', 'N/A'):.2e}")

        # Model posterior diagnostics (inherited)
        print(f"\n   📊 Inherited Model Diagnostics:")
        print(f"   Model posterior was PD: {prop_diag.get('model_post_is_positive_definite', 'N/A')}")
        print(f"   Model posterior min eig: {prop_diag.get('model_post_min_eigenvalue', 'N/A'):.2e}")

    print("\n" + "=" * 60)

    # Step 3: Compare with simpler configuration
    print("\n🔄 COMPARISON WITH SAFER CONFIGURATION:")

    safer_config = problematic_config.copy()
    safer_config.update({
        'alpha': 0.1,  # Larger regularization
        'N_p': 10      # Smaller property dimension
    })

    result_safe = test_block('compute_property_posterior_detailed', safer_config)
    print(f"Safer config success: {result_safe.success}")

    if result_safe.success and result_safe.results:
        safe_diag = result_safe.results['compute_property_posterior_detailed']
        print(f"Property posterior PD: {safe_diag.get('property_post_is_positive_definite', 'N/A')}")
        print(f"Property operator rank: {safe_diag.get('property_operator_rank', 'N/A')}")

    print("\n💡 CONCLUSIONS:")
    print("- Model posterior computation succeeds → prior/data/forward operator are OK")
    print("- Property posterior fails → issue is in property mapping P @ C_model @ P^T")
    print("- Check property operator rank vs dimension for injectivity")
    print("- Consider regularization or rank-deficient property operators")


def batch_analysis_example():
    """Example of batch analysis with detailed diagnostics."""

    print("\n" + "=" * 60)
    print("🏭 BATCH ANALYSIS WITH DETAILED DIAGNOSTICS")
    print("=" * 60)

    # Create test configurations
    base_config = {
        'endpoints': (0, 1), 'basis_type': 'sine',
        'integration_method_G': 'trapz', 'integration_method_T': 'trapz',
        'n_points_G': 200, 'n_points_T': 200,
        'K': 15, 'true_data_noise': 0.1, 'assumed_data_noise': 0.1,
        'm_bar_callable': lambda x: np.sin(2 * np.pi * x),
        'm_0_callable': lambda x: np.zeros_like(x),
        'solver': LUSolver()
    }

    # Vary alpha and N_p
    test_configs = []
    for alpha in [0.001, 0.01, 0.1]:
        for N_p in [5, 15, 25]:
            config = base_config.copy()
            config.update({
                'N': 20, 'N_d': 40, 'N_p': N_p, 'alpha': alpha
            })
            test_configs.append(config)

    print(f"Testing {len(test_configs)} configurations with detailed diagnostics...")

    # Run batch test
    results_df = run_custom_robustness_test(
        configurations=test_configs,
        target_block='compute_property_posterior_detailed',
        verbose=False
    )

    print(f"\n📊 BATCH RESULTS:")
    print(f"Success rate: {results_df['success'].mean():.1%}")

    # Analyze failure patterns
    failed = results_df[~results_df['success']]
    if len(failed) > 0:
        print(f"Failed configurations: {len(failed)}")
        print("\nFailure pattern analysis:")
        print("Alpha values in failures:", sorted(failed['param_alpha'].unique()))
        print("N_p values in failures:", sorted(failed['param_N_p'].unique()))

        # Look at specific diagnostic columns if available
        diag_cols = [c for c in results_df.columns if 'property_operator_rank' in c]
        if diag_cols:
            print(f"Property operator ranks in failures: {failed[diag_cols[0]].unique()}")

    print("\n✅ Batch analysis complete!")


if __name__ == "__main__":
    analyze_covariance_breakdown()
    batch_analysis_example()
