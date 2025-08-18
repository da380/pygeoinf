"""
Enhanced PLI Profiler - Phase 3: Parameter Optimization Framework
Goal: Find optimal parameters balancing precision and computational cost
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import itertools
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pandas as pd


@dataclass
class OptimizationObjective:
    """Define optimization objectives and constraints"""
    # Performance targets
    max_execution_time: float = np.inf
    max_memory_mb: float = np.inf

    # Precision targets
    min_posterior_accuracy: float = 0.0
    min_data_fit_quality: float = 0.0

    # Weights for multi-objective optimization
    time_weight: float = 1.0
    memory_weight: float = 1.0
    accuracy_weight: float = 2.0

    # Constraint penalties
    constraint_penalty: float = 1000.0


@dataclass
class ParameterSpace:
    """Define the parameter space for optimization"""
    # Dimension parameters
    N_range: Tuple[int, int] = (10, 200)
    N_d_range: Tuple[int, int] = (20, 500)
    N_p_range: Tuple[int, int] = (5, 100)

    # Integration parameters
    n_points_G_range: Tuple[int, int] = (100, 2000)
    n_points_T_range: Tuple[int, int] = (100, 2000)
    integration_methods: List[str] = None

    # Prior parameters
    alpha_range: Tuple[float, float] = (0.01, 10.0)
    K_range: Tuple[int, int] = (5, 500)

    # Noise parameters
    noise_range: Tuple[float, float] = (0.001, 1.0)

    # Basis types
    basis_types: List[str] = None

    def __post_init__(self):
        if self.integration_methods is None:
            self.integration_methods = ['trapz', 'simpson']
        if self.basis_types is None:
            self.basis_types = ['sine', 'fourier']


class ParameterOptimizer:
    """Multi-objective parameter optimization for PLI"""

    def __init__(self, objective: OptimizationObjective, param_space: ParameterSpace):
        self.objective = objective
        self.param_space = param_space
        self.evaluation_history: List[Dict[str, Any]] = []
        self.pareto_frontier: List[Dict[str, Any]] = []

    def optimize_parameters(self,
                          method: str = 'bayesian',
                          n_evaluations: int = 100,
                          initial_samples: int = 20) -> Dict[str, Any]:
        """Run parameter optimization using specified method"""

        print(f"🎯 Starting parameter optimization using {method}...")
        print(f"Target: ≤{self.objective.max_execution_time}s, "
              f"≤{self.objective.max_memory_mb}MB")

        if method == 'grid_search':
            return self._grid_search_optimization()
        elif method == 'random_search':
            return self._random_search_optimization(n_evaluations)
        elif method == 'bayesian':
            return self._bayesian_optimization(n_evaluations, initial_samples)
        elif method == 'evolutionary':
            return self._evolutionary_optimization(n_evaluations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _grid_search_optimization(self) -> Dict[str, Any]:
        """Systematic grid search (expensive but thorough)"""

        # Define grid points for key parameters
        N_values = np.linspace(self.param_space.N_range[0],
                              self.param_space.N_range[1], 5, dtype=int)
        alpha_values = np.logspace(np.log10(self.param_space.alpha_range[0]),
                                  np.log10(self.param_space.alpha_range[1]), 5)
        K_values = np.linspace(self.param_space.K_range[0],
                              self.param_space.K_range[1], 4, dtype=int)

        best_config = None
        best_score = np.inf

        total_configs = len(N_values) * len(alpha_values) * len(K_values)
        print(f"Evaluating {total_configs} grid configurations...")

        for i, (N, alpha, K) in enumerate(itertools.product(N_values, alpha_values, K_values)):
            print(f"Grid search progress: {i+1}/{total_configs}")

            # Create full parameter configuration
            config = self._create_config_from_key_params(N, alpha, K)

            # Evaluate configuration
            result = self._evaluate_configuration(config)

            if result['objective_score'] < best_score:
                best_score = result['objective_score']
                best_config = config

            self.evaluation_history.append({
                'config': config,
                'result': result,
                'method': 'grid_search'
            })

        return self._create_optimization_result(best_config, best_score, 'grid_search')

    def _random_search_optimization(self, n_evaluations: int) -> Dict[str, Any]:
        """Random search optimization"""

        best_config = None
        best_score = np.inf

        print(f"Evaluating {n_evaluations} random configurations...")

        for i in range(n_evaluations):
            if i % 10 == 0:
                print(f"Random search progress: {i+1}/{n_evaluations}")

            # Sample random configuration
            config = self._sample_random_configuration()

            # Evaluate
            result = self._evaluate_configuration(config)

            if result['objective_score'] < best_score:
                best_score = result['objective_score']
                best_config = config

            self.evaluation_history.append({
                'config': config,
                'result': result,
                'method': 'random_search'
            })

        return self._create_optimization_result(best_config, best_score, 'random_search')

    def _bayesian_optimization(self, n_evaluations: int, initial_samples: int) -> Dict[str, Any]:
        """Bayesian optimization using Gaussian Process"""

        # Initial random sampling
        print(f"Initial sampling: {initial_samples} configurations...")
        for i in range(initial_samples):
            config = self._sample_random_configuration()
            result = self._evaluate_configuration(config)
            self.evaluation_history.append({
                'config': config,
                'result': result,
                'method': 'bayesian_initial'
            })

        # Prepare data for GP
        X_train = np.array([self._config_to_vector(h['config'])
                           for h in self.evaluation_history])
        y_train = np.array([h['result']['objective_score']
                           for h in self.evaluation_history])

        # Train Gaussian Process
        gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )

        best_config = self.evaluation_history[np.argmin(y_train)]['config']
        best_score = np.min(y_train)

        # Bayesian optimization loop
        print(f"Bayesian optimization: {n_evaluations - initial_samples} iterations...")

        for i in range(n_evaluations - initial_samples):
            print(f"BO iteration: {i+1}/{n_evaluations - initial_samples}")

            # Fit GP
            gp.fit(X_train, y_train)

            # Find next point using acquisition function (Expected Improvement)
            next_config = self._find_next_point_ei(gp, X_train)

            # Evaluate
            result = self._evaluate_configuration(next_config)

            if result['objective_score'] < best_score:
                best_score = result['objective_score']
                best_config = next_config

            # Update training data
            X_train = np.vstack([X_train, self._config_to_vector(next_config)])
            y_train = np.append(y_train, result['objective_score'])

            self.evaluation_history.append({
                'config': next_config,
                'result': result,
                'method': 'bayesian_optimization'
            })

        return self._create_optimization_result(best_config, best_score, 'bayesian')

    def _evolutionary_optimization(self, n_evaluations: int) -> Dict[str, Any]:
        """Evolutionary optimization using differential evolution"""

        # Define bounds for continuous parameters
        bounds = [
            self.param_space.N_range,
            self.param_space.alpha_range,
            self.param_space.K_range,
            self.param_space.n_points_G_range,
            self.param_space.n_points_T_range
        ]

        def objective_func(x):
            # Convert vector to configuration
            config = self._vector_to_config(x)
            result = self._evaluate_configuration(config)
            return result['objective_score']

        # Run differential evolution
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=n_evaluations // 15,  # Population size ~15
            popsize=15,
            seed=42
        )

        best_config = self._vector_to_config(result.x)
        best_score = result.fun

        return self._create_optimization_result(best_config, best_score, 'evolutionary')

    def _evaluate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a parameter configuration"""

        try:
            # Import and run PLI pipeline
            from pli_profiling import run_with_dependencies
            from performance_analysis import profile_pli_pipeline

            # Run with performance monitoring
            analysis = profile_pli_pipeline(config)

            # Extract metrics
            performance = analysis['performance_analysis']['summary']
            execution_time = performance['total_execution_time']
            memory_mb = performance['total_memory_mb']

            # Calculate precision metrics (simplified for demo)
            # In practice, you'd compare against ground truth
            precision_score = self._calculate_precision_score(analysis)

            # Calculate multi-objective score
            objective_score = self._calculate_objective_score(
                execution_time, memory_mb, precision_score
            )

            return {
                'execution_time': execution_time,
                'memory_mb': memory_mb,
                'precision_score': precision_score,
                'objective_score': objective_score,
                'success': True,
                'detailed_analysis': analysis
            }

        except Exception as e:
            # Penalize failed configurations heavily
            return {
                'execution_time': np.inf,
                'memory_mb': np.inf,
                'precision_score': 0.0,
                'objective_score': self.objective.constraint_penalty,
                'success': False,
                'error': str(e)
            }

    def _calculate_precision_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate precision/accuracy score from analysis results"""
        # This would implement actual precision metrics
        # For now, return a placeholder score

        # Could include:
        # - Posterior mean accuracy vs ground truth
        # - Uncertainty quantification quality
        # - Data fit quality (chi-squared test)
        # - Cross-validation scores

        return 0.85  # Placeholder

    def _calculate_objective_score(self,
                                  execution_time: float,
                                  memory_mb: float,
                                  precision_score: float) -> float:
        """Calculate multi-objective optimization score"""

        # Normalize metrics
        time_norm = execution_time / self.objective.max_execution_time
        memory_norm = memory_mb / self.objective.max_memory_mb
        precision_norm = 1.0 - precision_score  # Lower is better

        # Apply constraint penalties
        penalty = 0.0
        if execution_time > self.objective.max_execution_time:
            penalty += self.objective.constraint_penalty * time_norm
        if memory_mb > self.objective.max_memory_mb:
            penalty += self.objective.constraint_penalty * memory_norm
        if precision_score < self.objective.min_posterior_accuracy:
            penalty += self.objective.constraint_penalty * precision_norm

        # Weighted combination
        score = (self.objective.time_weight * time_norm +
                self.objective.memory_weight * memory_norm +
                self.objective.accuracy_weight * precision_norm +
                penalty)

        return score

    def _sample_random_configuration(self) -> Dict[str, Any]:
        """Sample a random configuration from parameter space"""

        # Sample discrete parameters
        N = np.random.randint(self.param_space.N_range[0], self.param_space.N_range[1])
        N_d = np.random.randint(self.param_space.N_d_range[0], self.param_space.N_d_range[1])
        N_p = np.random.randint(self.param_space.N_p_range[0], self.param_space.N_p_range[1])

        # Ensure N_d >= N_p for well-posed problems
        N_d = max(N_d, N_p + 5)

        # Sample continuous parameters
        alpha = np.random.uniform(self.param_space.alpha_range[0],
                                 self.param_space.alpha_range[1])
        K = np.random.randint(self.param_space.K_range[0], self.param_space.K_range[1])

        # Integration parameters
        n_points_G = np.random.randint(self.param_space.n_points_G_range[0],
                                      self.param_space.n_points_G_range[1])
        n_points_T = np.random.randint(self.param_space.n_points_T_range[0],
                                      self.param_space.n_points_T_range[1])

        # Categorical parameters
        integration_method = np.random.choice(self.param_space.integration_methods)
        basis_type = np.random.choice(self.param_space.basis_types)

        return self._create_full_config(N, N_d, N_p, alpha, K,
                                       n_points_G, n_points_T,
                                       integration_method, basis_type)

    def _create_full_config(self, N, N_d, N_p, alpha, K,
                           n_points_G, n_points_T,
                           integration_method, basis_type) -> Dict[str, Any]:
        """Create a complete configuration dictionary"""

        return {
            'N': int(N), 'N_d': int(N_d), 'N_p': int(N_p),
            'endpoints': (0, 1),
            'basis_type': basis_type,
            'integration_method_G': integration_method,
            'integration_method_T': integration_method,
            'n_points_G': int(n_points_G),
            'n_points_T': int(n_points_T),
            'alpha': float(alpha),
            'K': int(K),
            'm_bar_callable': lambda x: np.sin(2 * np.pi * x),
            'm_0_callable': lambda x: np.zeros_like(x),
            'true_data_noise': 0.1,
            'assumed_data_noise': 0.1,
            'solver': 'LUSolver()'  # Would need actual instance
        }

    def _config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert configuration to vector for GP"""
        return np.array([
            config['N'],
            config['alpha'],
            config['K'],
            config['n_points_G'],
            config['n_points_T'],
            1.0 if config['integration_method_G'] == 'simpson' else 0.0,
            1.0 if config['basis_type'] == 'fourier' else 0.0
        ])

    def _vector_to_config(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert vector back to configuration"""
        integration_method = 'simpson' if x[5] > 0.5 else 'trapz'
        basis_type = 'fourier' if x[6] > 0.5 else 'sine'

        return self._create_full_config(
            int(x[0]), int(x[0] * 2), int(x[0] // 2),  # N, N_d, N_p
            x[1], int(x[2]), int(x[3]), int(x[4]),     # alpha, K, n_points
            integration_method, basis_type
        )

    def _find_next_point_ei(self, gp, X_train):
        """Find next point using Expected Improvement acquisition function"""
        # Simplified implementation - would use proper acquisition optimization
        # For now, just sample some candidates and pick best EI

        candidates = [self._sample_random_configuration() for _ in range(100)]
        candidate_vectors = [self._config_to_vector(c) for c in candidates]

        best_ei = -np.inf
        best_candidate = candidates[0]

        current_best = np.min([h['result']['objective_score']
                              for h in self.evaluation_history])

        for candidate, x_vec in zip(candidates, candidate_vectors):
            mu, sigma = gp.predict([x_vec], return_std=True)

            # Expected Improvement
            improvement = current_best - mu[0]
            if sigma[0] > 0:
                z = improvement / sigma[0]
                ei = improvement * self._norm_cdf(z) + sigma[0] * self._norm_pdf(z)
            else:
                ei = 0.0

            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        return best_candidate

    def _norm_cdf(self, x):
        """Standard normal CDF"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

    def _norm_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _create_config_from_key_params(self, N, alpha, K):
        """Create config from key parameters with reasonable defaults"""
        return self._create_full_config(
            N, N*2, N//2, alpha, K, 1000, 1000, 'trapz', 'sine'
        )

    def _create_optimization_result(self, best_config, best_score, method):
        """Create standardized optimization result"""
        return {
            'best_configuration': best_config,
            'best_score': best_score,
            'optimization_method': method,
            'total_evaluations': len(self.evaluation_history),
            'evaluation_history': self.evaluation_history,
            'pareto_frontier': self._compute_pareto_frontier()
        }

    def _compute_pareto_frontier(self):
        """Compute Pareto frontier for multi-objective optimization"""
        # Simplified Pareto frontier computation
        # In practice, would consider all objectives simultaneously

        successful_evals = [h for h in self.evaluation_history
                           if h['result']['success']]

        if not successful_evals:
            return []

        pareto_points = []

        for eval1 in successful_evals:
            is_dominated = False

            for eval2 in successful_evals:
                if eval1 == eval2:
                    continue

                # Check if eval1 is dominated by eval2
                r1, r2 = eval1['result'], eval2['result']

                if (r2['execution_time'] <= r1['execution_time'] and
                    r2['memory_mb'] <= r1['memory_mb'] and
                    r2['precision_score'] >= r1['precision_score'] and
                    (r2['execution_time'] < r1['execution_time'] or
                     r2['memory_mb'] < r1['memory_mb'] or
                     r2['precision_score'] > r1['precision_score'])):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_points.append(eval1)

        return pareto_points

    def plot_optimization_progress(self, save_path: Optional[str] = None):
        """Plot optimization progress and results"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Optimization progress
        scores = [h['result']['objective_score'] for h in self.evaluation_history]
        best_scores = np.minimum.accumulate(scores)

        ax1.plot(scores, 'b-', alpha=0.6, label='Evaluation scores')
        ax1.plot(best_scores, 'r-', linewidth=2, label='Best score so far')
        ax1.set_xlabel('Evaluation')
        ax1.set_ylabel('Objective Score')
        ax1.set_title('Optimization Progress')
        ax1.legend()
        ax1.set_yscale('log')

        # Time vs Memory trade-off
        successful = [h for h in self.evaluation_history if h['result']['success']]
        times = [h['result']['execution_time'] for h in successful]
        memories = [h['result']['memory_mb'] for h in successful]

        ax2.scatter(times, memories, alpha=0.6)
        ax2.set_xlabel('Execution Time (s)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Time vs Memory Trade-off')

        # Parameter importance (N vs objective score)
        N_values = [h['config']['N'] for h in successful]
        scores_successful = [h['result']['objective_score'] for h in successful]

        ax3.scatter(N_values, scores_successful, alpha=0.6)
        ax3.set_xlabel('N (Model Space Dimension)')
        ax3.set_ylabel('Objective Score')
        ax3.set_title('Model Dimension Impact')

        # Pareto frontier
        pareto = self._compute_pareto_frontier()
        if pareto:
            pareto_times = [p['result']['execution_time'] for p in pareto]
            pareto_memories = [p['result']['memory_mb'] for p in pareto]

            ax4.scatter(times, memories, alpha=0.3, label='All evaluations')
            ax4.scatter(pareto_times, pareto_memories, color='red', s=50,
                       label='Pareto frontier')
            ax4.set_xlabel('Execution Time (s)')
            ax4.set_ylabel('Memory Usage (MB)')
            ax4.set_title('Pareto Frontier')
            ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


# Example usage
def run_parameter_optimization_example():
    """Example of comprehensive parameter optimization"""

    # Define optimization objective
    objective = OptimizationObjective(
        max_execution_time=60.0,    # 1 minute max
        max_memory_mb=2000.0,       # 2GB max
        min_posterior_accuracy=0.8,  # 80% accuracy minimum
        time_weight=1.0,
        memory_weight=0.5,
        accuracy_weight=2.0
    )

    # Define parameter space
    param_space = ParameterSpace(
        N_range=(10, 100),
        N_d_range=(20, 200),
        N_p_range=(5, 50),
        alpha_range=(0.01, 1.0),
        K_range=(10, 100)
    )

    # Run optimization
    optimizer = ParameterOptimizer(objective, param_space)

    # Try different methods
    methods = ['random_search', 'bayesian']  # 'grid_search' too expensive for demo

    results = {}
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running {method} optimization...")
        print(f"{'='*50}")

        result = optimizer.optimize_parameters(method=method, n_evaluations=30)
        results[method] = result

        print(f"\n{method} Results:")
        print(f"Best score: {result['best_score']:.3f}")
        print(f"Best config: N={result['best_configuration']['N']}, "
              f"α={result['best_configuration']['alpha']:.3f}")

    # Generate comparison plots
    optimizer.plot_optimization_progress()

    return results


if __name__ == "__main__":
    results = run_parameter_optimization_example()
