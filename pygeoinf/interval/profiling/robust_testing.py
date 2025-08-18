"""
Enhanced PLI Profiler - Phase 1: Robust Testing Framework
Goal: Find scenarios where PLI breaks and understand failure modes

This framework now supports testing any dependency block, not just compute_property_posterior!

Available blocks:
- setup_spatial_spaces: Basic spatial domain setup
- setup_mappings: Integration/mapping operators
- _setup_truths_and_measurement: Data and measurement setup
- setup_prior_measure: Prior measure construction
- create_problems: Forward problem and Bayesian inference setup
- compute_model_posterior: Model posterior computation
- compute_property_posterior: Property posterior computation (default)

Quick usage examples:

1. Test a single block:
   from robust_testing import test_block, get_available_blocks

   # See available blocks
   print(get_available_blocks())

   # Test specific block
   result = test_block('compute_model_posterior', parameters)

2. Run custom test suite on any block:
   from robust_testing import run_custom_robustness_test

   results_df = run_custom_robustness_test(
       configurations=[config1, config2, ...],
       target_block='compute_model_posterior'
   )

3. Backward compatibility (defaults to compute_property_posterior):
   tester = RobustTester()
   result = tester.run_single_test(parameters)  # same as before
"""

import numpy as np
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time
import pandas as pd


class FailureType(Enum):
    """Classification of failure types"""
    NUMERICAL_INSTABILITY = "numerical_instability"
    MEMORY_ERROR = "memory_error"
    CONVERGENCE_FAILURE = "convergence_failure"
    DIMENSION_MISMATCH = "dimension_mismatch"
    PARAMETER_INVALID = "parameter_invalid"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class TestResult:
    """Comprehensive test result with failure analysis"""
    parameters: Dict[str, Any]
    success: bool
    execution_time: float
    memory_peak: Optional[float]
    results: Optional[Dict[str, Any]]
    failure_type: Optional[FailureType]
    failure_stage: Optional[str]
    error_message: Optional[str]
    stacktrace: Optional[str]
    warnings: List[str]


class ParameterGenerator:
    """Generate comprehensive parameter combinations for testing"""

    @staticmethod
    def generate_dimension_sweep():
        """Test different dimension combinations"""
        configs = []

        # Small dimensions (baseline)
        configs.extend([
            {'N': n, 'N_d': n_d, 'N_p': n_p}
            for n in [5, 10, 20]
            for n_d in [10, 20, 50]
            for n_p in [5, 10, 15]
        ])

        # Large dimensions (stress test)
        configs.extend([
            {'N': n, 'N_d': n_d, 'N_p': n_p}
            for n in [50, 100, 200]
            for n_d in [100, 200, 500]
            for n_p in [10, 25, 50]
        ])

        # Edge cases
        configs.extend([
            {'N': 1, 'N_d': 2, 'N_p': 1},  # Minimal case
            {'N': 500, 'N_d': 1000, 'N_p': 100},  # Very large
            {'N': 50, 'N_d': 51, 'N_p': 50},  # Barely determined
        ])

        return configs

    @staticmethod
    def generate_noise_sweep():
        """Test different noise levels"""
        return [
            {'true_data_noise': tn, 'assumed_data_noise': an}
            for tn in [0.001, 0.01, 0.1, 0.5, 1.0]  # Very low to very high
            for an in [0.001, 0.01, 0.1, 0.5, 1.0]
        ]

    @staticmethod
    def generate_prior_sweep():
        """Test different prior configurations"""
        return [
            {'alpha': alpha, 'K': k}
            for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]  # Very smooth to very rough
            for k in [5, 20, 50, 100, 500]  # Few to many KL modes
        ]

    @staticmethod
    def generate_integration_sweep():
        """Test different integration settings"""
        return [
            {
                'integration_method_G': method_g,
                'integration_method_T': method_t,
                'n_points_G': points_g,
                'n_points_T': points_t
            }
            for method_g in ['trapz', 'simpson']
            for method_t in ['trapz', 'simpson']
            for points_g in [50, 100, 500, 1000]
            for points_t in [50, 100, 500, 1000]
        ]

    @staticmethod
    def generate_pathological_cases():
        """Generate known problematic scenarios"""
        return [
            # Very noisy data
            {'true_data_noise': 2.0, 'assumed_data_noise': 0.01},
            # Mismatched noise assumptions
            {'true_data_noise': 0.01, 'assumed_data_noise': 1.0},
            # Extreme prior correlations
            {'alpha': 0.0001},  # Nearly singular
            {'alpha': 1000.0},  # Nearly white noise
            # Too few KL modes
            {'K': 1, 'N': 100},
            # Dimension mismatches (will these break gracefully?)
            {'N': 100, 'N_d': 5, 'N_p': 50},  # Underdetermined
        ]


class RobustTester:
    """Comprehensive testing with failure analysis"""

    def __init__(self, timeout_seconds: float = 300):
        self.timeout_seconds = timeout_seconds
        self.results: List[TestResult] = []

    def run_comprehensive_test_suite(self, base_parameters: Dict[str, Any], target_block: str = 'compute_property_posterior') -> List[TestResult]:
        """Run all test categories"""

        print("🧪 Starting Comprehensive PLI Test Suite...")
        print(f"📋 Target block: {target_block}")

        # Generate test cases
        test_configs = []

        # Dimension tests
        for dim_config in ParameterGenerator.generate_dimension_sweep()[:20]:  # Limit for demo
            test_configs.append({**base_parameters, **dim_config})

        # Noise tests
        for noise_config in ParameterGenerator.generate_noise_sweep()[:10]:
            test_configs.append({**base_parameters, **noise_config})

        # Prior tests
        for prior_config in ParameterGenerator.generate_prior_sweep()[:10]:
            test_configs.append({**base_parameters, **prior_config})

        # Pathological cases
        for path_config in ParameterGenerator.generate_pathological_cases():
            test_configs.append({**base_parameters, **path_config})

        print(f"Generated {len(test_configs)} test configurations")

        # Run tests
        for i, config in enumerate(test_configs):
            print(f"Running test {i+1}/{len(test_configs)}...")
            result = self.run_single_test(config, target_block=target_block)
            self.results.append(result)

            if not result.success:
                print(f"❌ FAILED: {result.failure_type} at {result.failure_stage}")
                print(f"   Error: {result.error_message}")
            else:
                print(f"✅ SUCCESS: {result.execution_time:.2f}s")

        return self.results

    def run_custom_test_suite(
        self,
        configurations: List[Dict[str, Any]],
        save_to_file: Optional[str] = None,
        verbose: bool = True,
        target_block: str = 'compute_property_posterior'
    ) -> pd.DataFrame:
        """
        Run tests on user-provided configurations and return results as DataFrame.

        Parameters:
        -----------
        configurations : List[Dict[str, Any]]
            List of parameter dictionaries to test
        save_to_file : Optional[str]
            If provided, save the DataFrame to this file path (CSV format)
        verbose : bool
            Whether to print progress messages
        target_block : str
            Which block to test (e.g., 'compute_property_posterior', 'compute_model_posterior', etc.)

        Returns:
        --------
        pd.DataFrame
            Results with columns for parameters, success, timing, errors, etc.
        """

        if verbose:
            print(f"🧪 Running Custom Test Suite with {len(configurations)} configurations...")
            print(f"📋 Target block: {target_block}")

        # Clear previous results
        self.results = []

        # Run tests
        for i, config in enumerate(configurations):
            if verbose:
                print(f"Running test {i+1}/{len(configurations)}...")

            result = self.run_single_test(config, target_block=target_block)
            self.results.append(result)

            if verbose:
                if not result.success:
                    print(f"❌ FAILED: {result.failure_type} at {result.failure_stage}")
                    print(f"   Error: {result.error_message}")
                else:
                    print(f"✅ SUCCESS: {result.execution_time:.2f}s")

        # Convert to DataFrame
        df = self._results_to_dataframe()

        # Save to file if requested
        if save_to_file:
            df.to_csv(save_to_file, index=False)
            if verbose:
                print(f"📁 Results saved to {save_to_file}")

        if verbose:
            success_rate = (df['success'].sum() / len(df)) * 100
            print(f"🔍 Test Suite Complete: {success_rate:.1f}% success rate")

        return df

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert test results to a pandas DataFrame for analysis"""

        rows = []

        for result in self.results:
            # Start with basic info
            row = {
                'success': result.success,
                'execution_time': result.execution_time,
                'failure_type': result.failure_type.value if result.failure_type else None,
                'failure_stage': result.failure_stage,
                'error_message': result.error_message,
                'warnings_count': len(result.warnings),
                'memory_peak': result.memory_peak
            }

            # Add all parameters as separate columns
            for param_name, param_value in result.parameters.items():
                row[f'param_{param_name}'] = param_value

            # Add results if available
            if result.results:
                for result_name, result_value in result.results.items():
                    # Handle different types of results
                    if isinstance(result_value, (int, float, bool)):
                        row[f'result_{result_name}'] = result_value
                    elif isinstance(result_value, np.ndarray):
                        # Store array statistics
                        row[f'result_{result_name}_mean'] = np.mean(result_value)
                        row[f'result_{result_name}_std'] = np.std(result_value)
                        row[f'result_{result_name}_min'] = np.min(result_value)
                        row[f'result_{result_name}_max'] = np.max(result_value)
                    else:
                        # Convert to string for complex objects
                        row[f'result_{result_name}'] = str(result_value)

            rows.append(row)

        return pd.DataFrame(rows)

    def run_single_test(self, parameters: Dict[str, Any], target_block: str = 'compute_property_posterior') -> TestResult:
        """Run a single test with comprehensive error handling"""

        start_time = time.time()
        warnings = []

        try:
            # Import here to avoid circular imports
            from pli_profiling import run_with_dependencies

            # Add timeout handling (simplified - real implementation would use signal)
            results = run_with_dependencies(target_block, parameters)

            execution_time = time.time() - start_time

            # Check for numerical issues in results
            if self._has_numerical_issues(results):
                warnings.append("Potential numerical instability detected in results")

            return TestResult(
                parameters=parameters,
                success=True,
                execution_time=execution_time,
                memory_peak=None,  # Would implement memory tracking
                results=results,
                failure_type=None,
                failure_stage=None,
                error_message=None,
                stacktrace=None,
                warnings=warnings
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Classify the failure type
            failure_type = self._classify_failure(e)
            failure_stage = self._extract_failure_stage(traceback.format_exc())

            # Try to extract diagnostics from CovarianceMatrixError
            diagnostics_results = None
            if 'CovarianceMatrixError' in str(type(e)) and hasattr(e, 'diagnostics'):
                # Use getattr to safely access the diagnostics attribute
                diagnostics = getattr(e, 'diagnostics', None)
                if diagnostics:
                    diagnostics_results = {target_block: diagnostics}

            return TestResult(
                parameters=parameters,
                success=False,
                execution_time=execution_time,
                memory_peak=None,
                results=diagnostics_results,  # Include diagnostics if available
                failure_type=failure_type,
                failure_stage=failure_stage,
                error_message=str(e),
                stacktrace=traceback.format_exc(),
                warnings=warnings
            )

    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception"""
        error_msg = str(exception).lower()

        # Check for our custom covariance matrix error
        if 'CovarianceMatrixError' in str(type(exception)):
            return FailureType.NUMERICAL_INSTABILITY
        elif isinstance(exception, (np.linalg.LinAlgError, ValueError)) and 'singular' in error_msg:
            return FailureType.NUMERICAL_INSTABILITY
        elif isinstance(exception, MemoryError):
            return FailureType.MEMORY_ERROR
        elif 'convergence' in error_msg or 'iteration' in error_msg:
            return FailureType.CONVERGENCE_FAILURE
        elif 'shape' in error_msg or 'dimension' in error_msg:
            return FailureType.DIMENSION_MISMATCH
        elif isinstance(exception, (ValueError, TypeError)) and 'parameter' in error_msg:
            return FailureType.PARAMETER_INVALID
        else:
            return FailureType.UNKNOWN

    def _extract_failure_stage(self, stacktrace: str) -> str:
        """Extract which stage/function the failure occurred in"""
        stages = [
            'setup_spatial_spaces',
            'setup_mappings',
            '_setup_truths_and_measurement',
            'setup_prior_measure',
            'create_problems',
            'compute_model_posterior',
            'compute_property_posterior'
        ]

        for stage in stages:
            if stage in stacktrace:
                return stage

        return "unknown_stage"

    def _has_numerical_issues(self, results: Dict[str, Any]) -> bool:
        """Check results for numerical instability signs"""
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value) or value < 0:
                    return True
        return False

    def generate_failure_report(self) -> Dict[str, Any]:
        """Generate comprehensive failure analysis report"""
        failures = [r for r in self.results if not r.success]
        successes = [r for r in self.results if r.success]

        # Failure analysis
        failure_by_type = {}
        for failure in failures:
            ftype = failure.failure_type.value if failure.failure_type else "unknown"
            if ftype not in failure_by_type:
                failure_by_type[ftype] = []
            failure_by_type[ftype].append(failure)

        # Success analysis
        success_times = [r.execution_time for r in successes]

        return {
            'summary': {
                'total_tests': len(self.results),
                'successes': len(successes),
                'failures': len(failures),
                'success_rate': len(successes) / len(self.results) if self.results else 0
            },
            'failure_analysis': {
                'by_type': {ftype: len(failures) for ftype, failures in failure_by_type.items()},
                'by_stage': self._analyze_failure_stages(failures)
            },
            'performance': {
                'avg_success_time': np.mean(success_times) if success_times else 0,
                'max_success_time': np.max(success_times) if success_times else 0,
                'min_success_time': np.min(success_times) if success_times else 0
            },
            'problematic_parameters': self._identify_problematic_parameters(failures)
        }

    def _analyze_failure_stages(self, failures: List[TestResult]) -> Dict[str, int]:
        """Analyze which stages fail most often"""
        stage_counts = {}
        for failure in failures:
            stage = failure.failure_stage or "unknown"
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        return stage_counts

    def _identify_problematic_parameters(self, failures: List[TestResult]) -> Dict[str, Any]:
        """Identify parameter ranges that frequently cause failures"""
        # This would analyze parameter distributions in failed vs successful cases
        # For now, simplified version
        return {"analysis": "Would analyze parameter correlations with failures"}


def run_custom_robustness_test(
    configurations: List[Dict[str, Any]],
    save_to_file: Optional[str] = None,
    timeout_seconds: float = 300,
    verbose: bool = True,
    target_block: str = 'compute_property_posterior'
) -> pd.DataFrame:
    """
    Convenience function to run robustness tests on custom configurations.

    Parameters:
    -----------
    configurations : List[Dict[str, Any]]
        List of parameter dictionaries to test
    save_to_file : Optional[str]
        If provided, save the DataFrame to this file path (CSV format)
    timeout_seconds : float
        Timeout for individual tests
    verbose : bool
        Whether to print progress messages
    target_block : str
        Which block to test (e.g., 'compute_property_posterior', 'compute_model_posterior', etc.)

    Returns:
    --------
    pd.DataFrame
        Results DataFrame with test outcomes and analysis

    Example:
    --------
    configs = [
        {'N': 10, 'N_d': 20, 'N_p': 5, 'alpha': 0.1},
        {'N': 20, 'N_d': 40, 'N_p': 10, 'alpha': 0.5},
        {'N': 50, 'N_d': 100, 'N_p': 25, 'alpha': 1.0},
    ]

    results_df = run_custom_robustness_test(
        configs,
        save_to_file='my_test_results.csv',
        target_block='compute_model_posterior'
    )
    """

    tester = RobustTester(timeout_seconds=timeout_seconds)
    return tester.run_custom_test_suite(
        configurations=configurations,
        save_to_file=save_to_file,
        verbose=verbose,
        target_block=target_block
    )


# Example usage function
def run_robustness_testing(target_block: str = 'compute_property_posterior'):
    """Example of how to run comprehensive robustness testing"""

    # Base parameters that work
    base_params = {
        'endpoints': (0, 1),
        'basis_type': 'sine',
        'integration_method_G': 'trapz',
        'integration_method_T': 'trapz',
        'n_points_G': 1000,
        'n_points_T': 1000,
        'm_bar_callable': lambda x: np.sin(2 * np.pi * x),
        'm_0_callable': lambda x: np.zeros_like(x),
        'solver': 'LUSolver()'  # String for now, would need actual instance
    }

    tester = RobustTester(timeout_seconds=60)
    results = tester.run_comprehensive_test_suite(base_params, target_block=target_block)

    # Generate report
    report = tester.generate_failure_report()

    print("🔍 ROBUSTNESS TESTING COMPLETE")
    print(f"Success rate: {report['summary']['success_rate']:.1%}")
    print("Most common failures:", report['failure_analysis']['by_type'])
    print("Failure stages:", report['failure_analysis']['by_stage'])

    return results, report


def get_available_blocks() -> List[str]:
    """Get list of available blocks that can be tested"""
    try:
        from pli_profiling import DEPENDENCY_GRAPH
        return list(DEPENDENCY_GRAPH.keys())
    except ImportError:
        return [
            'setup_spatial_spaces',
            'setup_mappings',
            '_setup_truths_and_measurement',
            'setup_prior_measure',
            'create_problems',
            'compute_model_posterior',
            'compute_property_posterior'
        ]


def test_block(
    block_name: str,
    parameters: Dict[str, Any],
    timeout_seconds: float = 60
) -> TestResult:
    """
    Convenience function to test a single block with given parameters.

    Parameters:
    -----------
    block_name : str
        Name of the block to test (use get_available_blocks() to see options)
    parameters : Dict[str, Any]
        Parameters dictionary for the test
    timeout_seconds : float
        Timeout for the test

    Returns:
    --------
    TestResult
        Result of the test

    Example:
    --------
    params = {
        'N': 20, 'N_d': 40, 'N_p': 10,
        'endpoints': (0, 1), 'basis_type': 'sine',
        # ... other required parameters
    }

    result = test_block('compute_model_posterior', params)
    print(f"Success: {result.success}")
    """

    available_blocks = get_available_blocks()
    if block_name not in available_blocks:
        raise ValueError(f"Unknown block '{block_name}'. Available blocks: {available_blocks}")

    tester = RobustTester(timeout_seconds=timeout_seconds)
    return tester.run_single_test(parameters, target_block=block_name)


def example_custom_testing():
    """Example of how to run custom configuration testing"""

    # Base parameters that work
    base_params = {
        'endpoints': (0, 1),
        'basis_type': 'sine',
        'integration_method_G': 'trapz',
        'integration_method_T': 'trapz',
        'n_points_G': 1000,
        'n_points_T': 1000,
        'm_bar_callable': lambda x: np.sin(2 * np.pi * x),
        'm_0_callable': lambda x: np.zeros_like(x),
        'solver': 'LUSolver()'
    }

    # Create custom configurations to test
    custom_configs = []

    # Test different dimensions
    for N, N_d, N_p in [(10, 20, 5), (25, 50, 15), (50, 100, 25)]:
        config = base_params.copy()
        config.update({'N': N, 'N_d': N_d, 'N_p': N_p})
        custom_configs.append(config)

    # Test different noise levels
    for noise in [0.01, 0.1, 0.5]:
        config = base_params.copy()
        config.update({
            'true_data_noise': noise,
            'assumed_data_noise': noise,
            'N': 20, 'N_d': 40, 'N_p': 10
        })
        custom_configs.append(config)

    # Test different priors
    for alpha in [0.01, 0.1, 1.0]:
        config = base_params.copy()
        config.update({
            'alpha': alpha,
            'K': 20,
            'N': 20, 'N_d': 40, 'N_p': 10
        })
        custom_configs.append(config)

    print(f"Testing {len(custom_configs)} custom configurations")

    # Run the tests and save results
    results_df = run_custom_robustness_test(
        configurations=custom_configs,
        save_to_file='custom_test_results.csv',
        timeout_seconds=60,
        verbose=True
    )

    # Basic analysis
    print("\n📊 RESULTS SUMMARY:")
    print(f"Total tests: {len(results_df)}")
    print(f"Successful: {results_df['success'].sum()}")
    print(f"Failed: {(~results_df['success']).sum()}")
    print(f"Success rate: {results_df['success'].mean():.1%}")

    if 'failure_type' in results_df.columns:
        print("\nFailure types:")
        print(results_df['failure_type'].value_counts())

    print("\nExecution time stats:")
    print(results_df['execution_time'].describe())

    return results_df


if __name__ == "__main__":
    # You can run either the comprehensive test or custom test

    # Original comprehensive test
    # results, report = run_robustness_testing()

    # New custom configuration test
    custom_results = example_custom_testing()
