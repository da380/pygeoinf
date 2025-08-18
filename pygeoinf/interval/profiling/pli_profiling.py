from pygeoinf.interval.function_providers import (
    NormalModesProvider,
    BumpFunctionProvider,
)
from pygeoinf.interval.interval_domain import IntervalDomain
from pygeoinf.interval.l2_space import L2Space
from pygeoinf.hilbert_space import EuclideanSpace
from pygeoinf.interval.operators import SOLAOperator
from pygeoinf.interval.functions import Function
from pygeoinf.gaussian_measure import GaussianMeasure
import numpy as np
from pygeoinf.interval.boundary_conditions import BoundaryConditions
from pygeoinf.interval.operators import LaplacianInverseOperator
from pygeoinf.forward_problem import LinearForwardProblem
from pygeoinf.linear_bayesian import LinearBayesianInference


from typing import Tuple, Dict, Any, List
import time

######################################
# FUNDAMENTAL BLOCKS
######################################


def setup_spatial_spaces(
    *,
    N: int,
    N_d: int,
    N_p: int,
    endpoints: Tuple[int, int],
    basis_type: str
):
    # Create a function domain and spaces

    function_domain = IntervalDomain(endpoints[0], endpoints[1])
    M = L2Space(N, function_domain, basis_type=basis_type)  # model space
    D = EuclideanSpace(N_d)  # data space
    P = EuclideanSpace(N_p)  # property space

    return M, D, P


def setup_mappings(
    M, D, P, /, *,
    integration_method_G,
    integration_method_T,
    n_points_G,
    n_points_T,
    N_p
):
    # Create forward and property mappings
    normal_modes_provider = NormalModesProvider(
        M,
        gaussian_width_percent_range=(5, 10),
        freq_range=(5, 10),
        random_state=39
    )
    G = SOLAOperator(
        M,
        D,
        normal_modes_provider,
        integration_method=integration_method_G,
        n_points=n_points_G
    )
    width = 0.1  # width of the bump target functions
    function_domain = M.function_domain
    # centers of the bumps
    centers = np.linspace(
        function_domain.a + width / 2,
        function_domain.b - width / 2,
        N_p
    )
    target_provider = BumpFunctionProvider(
        M,
        centers=centers,
        default_width=width,
        default_k=0.0
    )
    T = SOLAOperator(
        M,
        P,
        target_provider,
        integration_method=integration_method_T,
        n_points=n_points_T
    )
    return G, T


def _setup_truths_and_measurement(
        M, D, G, /, *,
        m_bar: Function,
        true_data_noise: float,
        assumed_data_noise: float,
):
    # Generate synthetic observations
    d_bar = G(m_bar)  # Clean observations
    noise_level = true_data_noise * np.max(d_bar)
    np.random.seed(42)  # For reproducibility
    d_tilde = d_bar + np.random.normal(
        0, noise_level, d_bar.shape
    )  # Noisy observations

    # Define data noise covariance
    noise_variance = (
        assumed_data_noise * np.max(d_tilde)
    ) ** 2  # 10% of peak signal
    C_D_matrix = noise_variance * np.eye(D.dim)

    # Create data ERROR measure (zero mean) for LinearForwardProblem
    gaussian_D_error = GaussianMeasure.from_covariance_matrix(
        D, C_D_matrix, expectation=np.zeros(D.dim)
    )

    # Create data measure (with observed data mean) for visualization
    gaussian_D = GaussianMeasure.from_covariance_matrix(
        D, C_D_matrix, expectation=d_tilde
    )

    return gaussian_D, gaussian_D_error, d_bar, d_tilde


def setup_prior_measure(
        M, /, *,
        alpha: float, K: int,
        m_0: Function
):
    # Define prior measure parameters
    bc_dirichlet = BoundaryConditions(bc_type='dirichlet', left=0, right=0)
    # Correlation length parameter (smaller α → longer correlations)
    C_0 = LaplacianInverseOperator(M, bc_dirichlet, alpha=alpha)

    # Create Gaussian measure on model space
    M.create_gaussian_measure(
        method='kl',
        kl_expansion=K,
        covariance=C_0,
        expectation=m_0
    )

    return M.gaussian_measure


def create_problems(
        G, gaussian_D_error,
        prior_M, T
):
    # Bayesian update computation
    forward_problem = LinearForwardProblem(
        G, data_error_measure=gaussian_D_error
    )
    bayesian_inference = LinearBayesianInference(
        forward_problem, prior_M, T
    )

    return forward_problem, bayesian_inference


def compute_model_posterior(
        M, d_tilde,
        bayesian_inference,
        /, *, solver
):
    # Compute posterior using the built-in solver
    t0 = time.time()
    posterior_model = bayesian_inference.model_posterior_measure(
        d_tilde, solver
    )
    t1 = time.time()
    posterior_model_computation_time = t1 - t0

    # Extract mean and covariance for compatibility
    m_tilde = posterior_model.expectation
    t0 = time.time()
    C_M_matrix = posterior_model.covariance.matrix(dense=True)
    t1 = time.time()
    C_M_computation_time = t1 - t0
    # Create new Gaussian measure with sampling capability using
    # dense covariance
    GaussianMeasure.from_covariance_matrix(
        M, C_M_matrix, expectation=m_tilde
    )

    return {
        "posterior_model_computation_time": posterior_model_computation_time,
        "C_M_computation_time": C_M_computation_time
    }


def compute_property_posterior(
        P, bayesian_inference, d_tilde, /, *, solver
):
    # Compute property posterior using built-in methods

    # Use the LinearBayesianInference class to compute property posterior
    # directly
    t0 = time.time()
    property_posterior = bayesian_inference.property_posterior_measure(
        d_tilde, solver
    )
    t1 = time.time()
    property_posterior_computation_time = t1 - t0

    # Extract property mean and covariance
    p_tilde = property_posterior.expectation
    t0 = time.time()
    cov_P_matrix = property_posterior.covariance.matrix(dense=True)
    t1 = time.time()
    cov_P_computation_time = t1 - t0

    # Basic diagnostics for the covariance matrix
    cov_diagnostics = {}

    try:
        if cov_P_matrix is not None:
            cov_diagnostics['cov_P_shape'] = getattr(
                cov_P_matrix, 'shape', None
            )

            # Compute symmetric eigenvalues (use eigvalsh for Hermitian)
            try:
                eigs = np.linalg.eigvalsh(cov_P_matrix)
            except Exception:
                # Fall back to a general eigenvalue routine if needed
                eigs = np.linalg.eigvals(cov_P_matrix)
                eigs = np.real(eigs)

            min_eig = float(np.min(eigs))
            max_eig = float(np.max(eigs))
            num_negative = int(np.sum(eigs < -1e-12))

            # Condition number heuristic (use absolute min eigenvalue)
            eps = 1e-16
            denom = max(float(abs(min_eig)), eps)
            condition_number = float(max_eig) / denom

            cov_diagnostics['cov_P_min_eigenvalue'] = min_eig
            cov_diagnostics['cov_P_max_eigenvalue'] = max_eig
            cov_diagnostics['cov_P_num_negative_eigenvalues'] = num_negative
            cov_diagnostics['cov_P_condition_number'] = condition_number
            cov_diagnostics['cov_P_is_positive_definite'] = (
                bool(min_eig > 1e-12)
            )
            cov_diagnostics['cov_P_is_invertible'] = (
                bool(abs(min_eig) > 1e-12)
            )

            # Suggest a minimal regularization amount if needed
            suggestions = []
            if min_eig <= 1e-12:
                # Suggest adding diagonal jitter to shift min_eig positive
                jitter = max(1e-8, float(-min_eig) + 1e-8)
                suggestions.append(
                    f"Add diagonal regularization ~ {jitter:.2e} to cov_P"
                )
            if condition_number > 1e12:
                suggestions.append(
                    "High condition number: consider stronger regularization"
                )

            if suggestions:
                cov_diagnostics['cov_P_suggested_regularization'] = suggestions

    except Exception as e:
        # If diagnostics fail, record the exception message
        cov_diagnostics['cov_P_suggested_regularization'] = [
            f"Diagnostics error: {str(e)}"
        ]

    # Keep original behavior (create GaussianMeasure) but record diagnostics
    try:
        GaussianMeasure.from_covariance_matrix(
            P, cov_P_matrix, expectation=p_tilde
        )
    except Exception as e:
        # Record that measure construction failed but continue returning
        # diagnostics so callers can inspect what went wrong.
        cov_diagnostics['cov_P_measure_failed'] = True
        cov_diagnostics['cov_P_measure_error'] = str(e)

    # Return timings plus diagnostics for downstream analysis
    out = {
        "property_posterior_computation_time": (
            property_posterior_computation_time
        ),
        "cov_P_computation_time": cov_P_computation_time,
    }

    # Merge diagnostics into output explicitly to avoid type-checker issues
    for k, v in cov_diagnostics.items():
        out[k] = v

    return out


######################################
# DEPENDENCY RESOLUTION SYSTEM
######################################

# Define the dependency graph
DEPENDENCY_GRAPH = {
    'setup_spatial_spaces': [],
    'setup_mappings': ['setup_spatial_spaces'],
    '_setup_truths_and_measurement': [
        'setup_spatial_spaces',
        'setup_mappings',
    ],
    'setup_prior_measure': ['setup_spatial_spaces'],
    'create_problems': [
        'setup_mappings',
        '_setup_truths_and_measurement',
        'setup_prior_measure',
    ],
    'compute_model_posterior': ['setup_spatial_spaces', 'create_problems'],
    'compute_property_posterior': ['setup_spatial_spaces', 'create_problems'],
}

# Function registry
FUNCTION_REGISTRY = {
    'setup_spatial_spaces': setup_spatial_spaces,
    'setup_mappings': setup_mappings,
    '_setup_truths_and_measurement': _setup_truths_and_measurement,
    'setup_prior_measure': setup_prior_measure,
    'create_problems': create_problems,
    'compute_model_posterior': compute_model_posterior,
    'compute_property_posterior': compute_property_posterior,
}

# Define parameter requirements for each function
PARAMETER_REQUIREMENTS = {
    'setup_spatial_spaces': ['N', 'N_d', 'N_p', 'endpoints', 'basis_type'],
    'setup_mappings': ['integration_method_G', 'integration_method_T',
                       'n_points_G', 'n_points_T', 'N_p'],
    '_setup_truths_and_measurement': ['m_bar_callable', 'true_data_noise',
                                      'assumed_data_noise'],
    'setup_prior_measure': ['alpha', 'K', 'm_0_callable'],
    'create_problems': [],  # uses only artifacts from dependencies
    'compute_model_posterior': ['solver'],
    'compute_property_posterior': ['solver'],
}

# Define artifact outputs for each function (in order)
ARTIFACT_OUTPUTS = {
    'setup_spatial_spaces': ['M', 'D', 'P'],
    'setup_mappings': ['G', 'T'],
    '_setup_truths_and_measurement': [
        'gaussian_D',
        'gaussian_D_error',
        'd_bar',
        'd_tilde',
    ],
    'setup_prior_measure': ['prior_M'],
    'create_problems': ['forward_problem', 'bayesian_inference'],
    'compute_model_posterior': [],  # returns only results dict
    'compute_property_posterior': [],  # returns only results dict
}


def topological_sort(dependencies: Dict[str, List[str]]) -> List[str]:
    """Topologically sort dependencies to determine execution order."""
    visited = set()
    temp_visited = set()
    result = []

    def visit(node):
        if node in temp_visited:
            raise ValueError(f"Circular dependency detected involving {node}")
        if node in visited:
            return

        temp_visited.add(node)
        for dep in dependencies.get(node, []):
            visit(dep)
        temp_visited.remove(node)
        visited.add(node)
        result.append(node)

    for node in dependencies:
        visit(node)

    return result


def extract_parameters(
    all_params: Dict[str, Any], func_name: str
) -> Dict[str, Any]:
    """Extract parameters needed for a specific function."""
    required = PARAMETER_REQUIREMENTS[func_name]
    return {k: all_params[k] for k in required if k in all_params}


def run_with_dependencies(
    target_function: str, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a target function and all its dependencies automatically.

    Args:
        target_function: Name of the function to run
        parameters: Dict containing all parameters needed for target and
            dependencies.
                   For Function objects, you can pass callables that will be
                   converted
                   to Function objects once the appropriate space is available.

    Returns:
        Dict containing all results from functions that return results

    Example:
        results = run_with_dependencies('compute_property_posterior', {
            'N': 30, 'N_d': 100, 'N_p': 10,
            'endpoints': (0, 1), 'basis_type': 'sine',
            'integration_method_G': 'trapz', 'integration_method_T': 'trapz',
            'n_points_G': 1000, 'n_points_T': 1000,
            'm_bar_callable': lambda x: np.sin(x), 'true_data_noise': 0.1,
            'assumed_data_noise': 0.1, 'alpha': 0.1, 'K': 500,
            'm_0_callable': lambda x: x, 'solver': CholeskySolver()
        })
    """
    if target_function not in DEPENDENCY_GRAPH:
        raise ValueError(f"Unknown function: {target_function}")

    # Find all dependencies for the target
    all_deps = set()
    to_visit = [target_function]

    while to_visit:
        current = to_visit.pop()
        if current not in all_deps:
            all_deps.add(current)
            # extend stack with current's dependencies
            to_visit.extend(DEPENDENCY_GRAPH[current])

    # Sort dependencies topologically
    subgraph = {
        k: v
        for k, v in DEPENDENCY_GRAPH.items()
        if k in all_deps
    }
    execution_order = topological_sort(subgraph)

    # Storage for artifacts between function calls
    artifacts = {}
    all_results = {}

    # Execute functions in dependency order
    for func_name in execution_order:
        func = FUNCTION_REGISTRY[func_name]
        func_params = extract_parameters(parameters, func_name)

        # Prepare positional arguments (artifacts) for this function
        pos_args = []
        if func_name == 'setup_mappings':
            pos_args = [
                artifacts['M'],
                artifacts['D'],
                artifacts['P'],
            ]
        elif func_name == '_setup_truths_and_measurement':
            pos_args = [
                artifacts['M'],
                artifacts['D'],
                artifacts['G'],
            ]
        elif func_name == 'setup_prior_measure':
            pos_args = [artifacts['M']]
        elif func_name == 'create_problems':
            pos_args = [
                artifacts['G'],
                artifacts['gaussian_D_error'],
                artifacts['prior_M'],
                artifacts['T'],
            ]
        elif func_name == 'compute_model_posterior':
            pos_args = [
                artifacts['M'],
                artifacts['d_tilde'],
                artifacts['bayesian_inference'],
            ]
        elif func_name == 'compute_property_posterior':
            pos_args = [
                artifacts['P'],
                artifacts['bayesian_inference'],
                artifacts['d_tilde'],
            ]

        # Execute the function
        print(f"Executing {func_name}...")

        # Handle Function object creation after spaces are available
        if func_name == '_setup_truths_and_measurement':
            # Convert m_bar_callable to Function if provided
            if 'm_bar_callable' in parameters:
                func_params['m_bar'] = Function(
                    artifacts['M'],
                    evaluate_callable=parameters['m_bar_callable'],
                )
                # Remove the callable parameter since we converted it
                func_params.pop('m_bar_callable', None)
        elif func_name == 'setup_prior_measure':
            # Convert m_0_callable to Function if provided
            if 'm_0_callable' in parameters:
                func_params['m_0'] = Function(
                    artifacts['M'],
                    evaluate_callable=parameters['m_0_callable'],
                )
                # Remove the callable parameter since we converted it
                func_params.pop('m_0_callable', None)

        result = func(*pos_args, **func_params)

        # Store artifacts for next functions
        output_names = ARTIFACT_OUTPUTS[func_name]
        if output_names:
            if isinstance(result, tuple):
                for i, name in enumerate(output_names):
                    artifacts[name] = result[i]
            else:
                artifacts[output_names[0]] = result

        # Store results if function returns a results dict
        if isinstance(result, dict) and func_name in (
            'compute_model_posterior', 'compute_property_posterior'
        ):
            all_results[func_name] = result

    return all_results


# Convenience functions for common targets
def run_model_posterior(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run compute_model_posterior and all its dependencies."""
    return run_with_dependencies('compute_model_posterior', parameters)


def run_property_posterior(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run compute_property_posterior and all its dependencies."""
    return run_with_dependencies('compute_property_posterior', parameters)


def run_full_analysis(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run both model and property posterior computations."""
    results = {}
    results.update(run_model_posterior(parameters))
    # Property computation reuses artifacts from model computation
    prop_results = run_with_dependencies(
        'compute_property_posterior', parameters
    )
    results.update(prop_results)
    return results
