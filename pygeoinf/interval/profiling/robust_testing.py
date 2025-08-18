"""
Enhanced PLI Profiler - Phase 1: Robust Testing Framework
Goal: Find scenarios where PLI breaks and understand failure modes
"""

import numpy as np
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time


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

    def run_comprehensive_test_suite(self, base_parameters: Dict[str, Any]) -> List[TestResult]:
        """Run all test categories"""

        print("🧪 Starting Comprehensive PLI Test Suite...")

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
            result = self.run_single_test(config)
            self.results.append(result)

            if not result.success:
                print(f"❌ FAILED: {result.failure_type} at {result.failure_stage}")
                print(f"   Error: {result.error_message}")
            else:
                print(f"✅ SUCCESS: {result.execution_time:.2f}s")

        return self.results

    def run_single_test(self, parameters: Dict[str, Any]) -> TestResult:
        """Run a single test with comprehensive error handling"""

        start_time = time.time()
        warnings = []

        try:
            # Import here to avoid circular imports
            from pli_profiling import run_with_dependencies

            # Add timeout handling (simplified - real implementation would use signal)
            results = run_with_dependencies('compute_property_posterior', parameters)

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

            return TestResult(
                parameters=parameters,
                success=False,
                execution_time=execution_time,
                memory_peak=None,
                results=None,
                failure_type=failure_type,
                failure_stage=failure_stage,
                error_message=str(e),
                stacktrace=traceback.format_exc(),
                warnings=warnings
            )

    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception"""
        error_msg = str(exception).lower()

        if isinstance(exception, (np.linalg.LinAlgError, ValueError)) and 'singular' in error_msg:
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


# Example usage function
def run_robustness_testing():
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
    results = tester.run_comprehensive_test_suite(base_params)

    # Generate report
    report = tester.generate_failure_report()

    print("🔍 ROBUSTNESS TESTING COMPLETE")
    print(f"Success rate: {report['summary']['success_rate']:.1%}")
    print("Most common failures:", report['failure_analysis']['by_type'])
    print("Failure stages:", report['failure_analysis']['by_stage'])

    return results, report


if __name__ == "__main__":
    results, report = run_robustness_testing()
