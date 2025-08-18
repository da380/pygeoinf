"""
Enhanced PLI Profiler - Phase 2: Detailed Performance Analysis
Goal: Understand computational costs and bottlenecks
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import cProfile
import pstats
import io
from contextlib import contextmanager
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics for each stage"""
    stage_name: str
    execution_time: float
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_percent: float

    # Detailed algorithmic costs
    matrix_operations: Dict[str, float] = field(default_factory=dict)
    integration_costs: Dict[str, float] = field(default_factory=dict)
    eigenvalue_costs: Dict[str, float] = field(default_factory=dict)

    # Memory breakdown
    matrix_memory_mb: float = 0.0
    basis_memory_mb: float = 0.0
    artifact_memory_mb: float = 0.0


class DetailedProfiler:
    """Instrument the PLI pipeline with detailed performance monitoring"""

    def __init__(self):
        self.stage_metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager for profiling individual stages"""

        # Start monitoring
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_cpu = self.process.cpu_percent()

        # Enable detailed profiling
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            yield
        finally:
            # Stop profiling
            profiler.disable()

            # Collect metrics
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            end_cpu = self.process.cpu_percent()

            # Analyze profile
            detailed_costs = self._analyze_profile(profiler)

            # Create metrics object
            metrics = PerformanceMetrics(
                stage_name=stage_name,
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - self.baseline_memory,
                memory_peak_mb=max(start_memory, end_memory) - self.baseline_memory,
                cpu_percent=(start_cpu + end_cpu) / 2,
                **detailed_costs
            )

            self.stage_metrics.append(metrics)

    def _analyze_profile(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """Extract detailed costs from cProfile data"""

        # Capture profiler output
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        profile_output = s.getvalue()

        # Analyze for specific operation types
        matrix_ops = self._extract_matrix_operations(profile_output)
        integration_ops = self._extract_integration_operations(profile_output)
        eigenvalue_ops = self._extract_eigenvalue_operations(profile_output)

        return {
            'matrix_operations': matrix_ops,
            'integration_costs': integration_ops,
            'eigenvalue_costs': eigenvalue_ops
        }

    def _extract_matrix_operations(self, profile_output: str) -> Dict[str, float]:
        """Extract matrix operation costs from profile"""
        ops = {}

        # Look for common matrix operations
        matrix_patterns = [
            ('matrix_multiply', ['dot', 'matmul', '@']),
            ('matrix_solve', ['solve', 'inv', 'linalg']),
            ('cholesky', ['cholesky']),
            ('eigenvalue', ['eig', 'eigvals', 'eigvecs'])
        ]

        for op_name, patterns in matrix_patterns:
            total_time = 0.0
            for line in profile_output.split('\n'):
                for pattern in patterns:
                    if pattern in line.lower():
                        # Extract time (simplified parsing)
                        parts = line.split()
                        if len(parts) > 3:
                            try:
                                total_time += float(parts[3])
                            except (ValueError, IndexError):
                                pass
            ops[op_name] = total_time

        return ops

    def _extract_integration_operations(self, profile_output: str) -> Dict[str, float]:
        """Extract integration costs"""
        ops = {}

        integration_patterns = [
            ('simpson', ['simpson']),
            ('trapezoid', ['trapz']),
            ('function_eval', ['evaluate', 'call'])
        ]

        for op_name, patterns in integration_patterns:
            total_time = 0.0
            for line in profile_output.split('\n'):
                for pattern in patterns:
                    if pattern in line.lower():
                        parts = line.split()
                        if len(parts) > 3:
                            try:
                                total_time += float(parts[3])
                            except (ValueError, IndexError):
                                pass
            ops[op_name] = total_time

        return ops

    def _extract_eigenvalue_operations(self, profile_output: str) -> Dict[str, float]:
        """Extract eigenvalue computation costs"""
        # Similar pattern to above but for eigenvalue-specific operations
        return {'kl_expansion': 0.0, 'spectrum_computation': 0.0}

    def estimate_memory_usage(self, artifacts: Dict[str, Any]) -> Dict[str, float]:
        """Estimate memory usage of different components"""
        memory_breakdown = {
            'matrices': 0.0,
            'basis_functions': 0.0,
            'operators': 0.0,
            'measures': 0.0
        }

        for name, artifact in artifacts.items():
            if hasattr(artifact, 'shape'):  # Numpy arrays
                size_mb = artifact.nbytes / 1024 / 1024
                if 'matrix' in name.lower() or 'cov' in name.lower():
                    memory_breakdown['matrices'] += size_mb
                elif 'basis' in name.lower():
                    memory_breakdown['basis_functions'] += size_mb

        return memory_breakdown

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""

        total_time = sum(m.execution_time for m in self.stage_metrics)
        total_memory = sum(m.memory_peak_mb for m in self.stage_metrics)

        # Time breakdown by stage
        time_breakdown = {
            m.stage_name: m.execution_time for m in self.stage_metrics
        }

        # Memory breakdown by stage
        memory_breakdown = {
            m.stage_name: m.memory_peak_mb for m in self.stage_metrics
        }

        # Bottleneck identification
        bottlenecks = self._identify_bottlenecks()

        # Scaling analysis
        scaling_analysis = self._analyze_scaling()

        return {
            'summary': {
                'total_execution_time': total_time,
                'total_memory_mb': total_memory,
                'most_expensive_stage': max(self.stage_metrics,
                                          key=lambda x: x.execution_time).stage_name
            },
            'time_breakdown': time_breakdown,
            'memory_breakdown': memory_breakdown,
            'bottlenecks': bottlenecks,
            'scaling_analysis': scaling_analysis,
            'optimization_recommendations': self._get_optimization_recommendations()
        }

    def _identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks"""
        total_time = sum(m.execution_time for m in self.stage_metrics)

        bottlenecks = {}

        for metric in self.stage_metrics:
            percentage = (metric.execution_time / total_time) * 100
            if percentage > 20:  # Stage takes >20% of total time
                bottlenecks[metric.stage_name] = {
                    'time_percentage': percentage,
                    'primary_cost': self._get_primary_cost(metric),
                    'recommendation': self._get_stage_recommendation(metric)
                }

        return bottlenecks

    def _get_primary_cost(self, metric: PerformanceMetrics) -> str:
        """Identify the primary cost driver for a stage"""
        costs = {
            'matrix_ops': sum(metric.matrix_operations.values()),
            'integration': sum(metric.integration_costs.values()),
            'eigenvalues': sum(metric.eigenvalue_costs.values())
        }

        return max(costs, key=costs.get) if costs else "unknown"

    def _get_stage_recommendation(self, metric: PerformanceMetrics) -> str:
        """Get optimization recommendation for a stage"""
        primary_cost = self._get_primary_cost(metric)

        recommendations = {
            'matrix_ops': "Consider using sparse matrices or iterative solvers",
            'integration': "Reduce integration points or use adaptive methods",
            'eigenvalues': "Reduce KL expansion terms or use approximate methods"
        }

        return recommendations.get(primary_cost, "Profile further for specific bottlenecks")

    def _analyze_scaling(self) -> Dict[str, str]:
        """Analyze algorithmic scaling properties"""
        # This would need multiple runs with different sizes
        # For now, provide theoretical scaling

        return {
            'gram_matrix_computation': 'O(N² × integration_points)',
            'matrix_solve': 'O(N³) for dense, O(N^1.5) for sparse',
            'kl_expansion': 'O(K × N) where K is number of modes',
            'integration': 'O(N × integration_points)',
            'recommendation': 'N and integration_points are main scaling factors'
        }

    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze patterns in metrics
        avg_matrix_time = np.mean([
            sum(m.matrix_operations.values()) for m in self.stage_metrics
        ])
        avg_integration_time = np.mean([
            sum(m.integration_costs.values()) for m in self.stage_metrics
        ])

        if avg_matrix_time > avg_integration_time:
            recommendations.append(
                "Matrix operations dominate - consider iterative solvers"
            )
        else:
            recommendations.append(
                "Integration dominates - optimize quadrature or basis functions"
            )

        # Memory recommendations
        total_memory = sum(m.memory_peak_mb for m in self.stage_metrics)
        if total_memory > 1000:  # >1GB
            recommendations.append(
                "High memory usage - consider chunking or out-of-core algorithms"
            )

        return recommendations

    def plot_performance_breakdown(self, save_path: Optional[str] = None):
        """Generate performance visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Time breakdown pie chart
        stage_names = [m.stage_name for m in self.stage_metrics]
        stage_times = [m.execution_time for m in self.stage_metrics]

        ax1.pie(stage_times, labels=stage_names, autopct='%1.1f%%')
        ax1.set_title('Execution Time by Stage')

        # Memory usage bar chart
        stage_memory = [m.memory_peak_mb for m in self.stage_metrics]
        ax2.bar(range(len(stage_names)), stage_memory)
        ax2.set_xticks(range(len(stage_names)))
        ax2.set_xticklabels(stage_names, rotation=45)
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Peak Memory by Stage')

        # Time vs Memory scatter
        ax3.scatter(stage_times, stage_memory)
        for i, name in enumerate(stage_names):
            ax3.annotate(name, (stage_times[i], stage_memory[i]))
        ax3.set_xlabel('Execution Time (s)')
        ax3.set_ylabel('Peak Memory (MB)')
        ax3.set_title('Time vs Memory Trade-off')

        # Operation type breakdown (example for first stage)
        if self.stage_metrics:
            first_stage = self.stage_metrics[0]
            op_types = list(first_stage.matrix_operations.keys())
            op_times = list(first_stage.matrix_operations.values())

            if op_times:
                ax4.bar(op_types, op_times)
                ax4.set_title('Operation Breakdown (First Stage)')
                ax4.set_ylabel('Time (s)')
                plt.setp(ax4.get_xticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


# Integration with existing dependency system
def profile_pli_pipeline(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run PLI pipeline with detailed profiling"""

    profiler = DetailedProfiler()

    # Import the dependency system
    from pli_profiling import DEPENDENCY_GRAPH, run_with_dependencies

    # Profile each major stage
    stages_to_profile = [
        'setup_spatial_spaces',
        'setup_mappings',
        '_setup_truths_and_measurement',
        'setup_prior_measure',
        'compute_model_posterior',
        'compute_property_posterior'
    ]

    results = {}

    for stage in stages_to_profile:
        with profiler.profile_stage(stage):
            stage_result = run_with_dependencies(stage, parameters)
            results[stage] = stage_result

    # Generate comprehensive report
    performance_report = profiler.generate_performance_report()

    # Create visualizations
    profiler.plot_performance_breakdown()

    return {
        'stage_results': results,
        'performance_analysis': performance_report,
        'raw_metrics': profiler.stage_metrics
    }


if __name__ == "__main__":
    # Example usage
    test_params = {
        'N': 50, 'N_d': 100, 'N_p': 20,
        'endpoints': (0, 1), 'basis_type': 'sine',
        'integration_method_G': 'trapz', 'integration_method_T': 'trapz',
        'n_points_G': 1000, 'n_points_T': 1000,
        'm_bar_callable': lambda x: np.sin(2 * np.pi * x),
        'm_0_callable': lambda x: np.zeros_like(x),
        'alpha': 0.1, 'K': 50,
        'true_data_noise': 0.1, 'assumed_data_noise': 0.1,
        'solver': 'LUSolver()'  # Would need actual instance
    }

    print("🔍 Running Detailed Performance Analysis...")
    analysis = profile_pli_pipeline(test_params)

    print("📊 Performance Summary:")
    summary = analysis['performance_analysis']['summary']
    print(f"Total time: {summary['total_execution_time']:.2f}s")
    print(f"Total memory: {summary['total_memory_mb']:.1f}MB")
    print(f"Bottleneck: {summary['most_expensive_stage']}")
