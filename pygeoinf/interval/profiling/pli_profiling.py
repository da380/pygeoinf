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


class CovarianceMatrixError(Exception):
    """Exception raised when covariance matrix has numerical issues"""

    def __init__(self, message: str, diagnostics: Dict[str, Any] = None):
        super().__init__(message)
        self.diagnostics = diagnostics if diagnostics is not None else {}


######################################
# COVARIANCE MATRIX DIAGNOSTICS
######################################

def _compute_eigenvalue_diagnostics(cov_matrix) -> Dict[str, Any]:
    """Compute eigenvalue-based diagnostics for a covariance matrix."""
    diagnostics = {}

    try:
        if cov_matrix is not None:
            diagnostics['cov_P_shape'] = getattr(cov_matrix, 'shape', None)

            # Compute symmetric eigenvalues (use eigvalsh for Hermitian)
            try:
                eigs = np.linalg.eigvalsh(cov_matrix)
            except Exception:
                # Fall back to a general eigenvalue routine if needed
                eigs = np.linalg.eigvals(cov_matrix)
                eigs = np.real(eigs)

            min_eig = float(np.min(eigs))
            max_eig = float(np.max(eigs))
            num_negative = int(np.sum(eigs < -1e-12))

            # Condition number heuristic (use absolute min eigenvalue)
            eps = 1e-16
            denom = max(float(abs(min_eig)), eps)
            condition_number = float(max_eig) / denom

            diagnostics.update({
                'cov_P_min_eigenvalue': min_eig,
                'cov_P_max_eigenvalue': max_eig,
                'cov_P_num_negative_eigenvalues': num_negative,
                'cov_P_condition_number': condition_number,
                'cov_P_is_positive_definite': bool(min_eig > 1e-12),
                'cov_P_is_invertible': bool(abs(min_eig) > 1e-12)
            })

    except Exception as e:
        diagnostics['cov_P_eigenvalue_error'] = str(e)

    return diagnostics


def _test_measure_construction(P, cov_matrix, expectation) -> Dict[str, Any]:
    """Test if Gaussian measure can be constructed from covariance matrix."""
    result = {
        'cov_P_measure_failed': False,
        'cov_P_measure_error': None
    }

    try:
        GaussianMeasure.from_covariance_matrix(P, cov_matrix, expectation=expectation)
    except Exception as e:
        result['cov_P_measure_failed'] = True
        result['cov_P_measure_error'] = str(e)

    return result


def _detect_critical_failures(diagnostics: Dict[str, Any]) -> List[str]:
    """Detect critical failures that should raise exceptions."""
    critical_failures = []

    # Check if covariance matrix is not positive definite
    if not diagnostics.get('cov_P_is_positive_definite', True):
        min_eig = diagnostics.get('cov_P_min_eigenvalue', 'unknown')
        critical_failures.append(
            f"Covariance matrix not positive definite (min eigenvalue: {min_eig})"
        )

    # Check if measure construction failed
    if diagnostics.get('cov_P_measure_failed', False):
        measure_error = diagnostics.get('cov_P_measure_error', 'unknown')
        critical_failures.append(
            f"Gaussian measure construction failed: {measure_error}"
        )

    # Check for negative eigenvalues
    num_neg = diagnostics.get('cov_P_num_negative_eigenvalues', 0)
    if num_neg > 0:
        critical_failures.append(f"{num_neg} negative eigenvalues detected")

    return critical_failures


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
    """Compute property posterior using built-in methods with diagnostics."""

    # Use the LinearBayesianInference class to compute property posterior
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

    # Run comprehensive covariance matrix diagnostics
    cov_diagnostics = _compute_eigenvalue_diagnostics(cov_P_matrix)

    # Test measure construction
    measure_results = _test_measure_construction(P, cov_P_matrix, p_tilde)
    cov_diagnostics.update(measure_results)

    # Prepare output with timings and diagnostics
    out = {
        "property_posterior_computation_time": property_posterior_computation_time,
        "cov_P_computation_time": cov_P_computation_time,
    }
    out.update(cov_diagnostics)

    # Check for critical failures and raise exception if needed
    critical_failures = _detect_critical_failures(cov_diagnostics)
    if critical_failures:
        error_message = "; ".join(critical_failures)
        raise CovarianceMatrixError(error_message, diagnostics=out)

    return out


def compute_model_posterior_detailed(
        M, d_tilde, bayesian_inference, /, *, solver
):
    """Compute model posterior with detailed intermediate diagnostics."""

    # Step 1: Compute normal operator (covariance of prior predictive distribution)
    t0 = time.time()
    normal_operator = bayesian_inference.normal_operator
    t1 = time.time()
    normal_operator_time = t1 - t0

    # Step 2: Get normal operator matrix and diagnose it
    t0 = time.time()
    normal_op_matrix = normal_operator.matrix(dense=True)
    t1 = time.time()
    normal_op_matrix_time = t1 - t0

    normal_op_diagnostics = _compute_eigenvalue_diagnostics(normal_op_matrix)
    # Rename keys to be specific to normal operator
    normal_op_diagnostics = {f"normal_op_{k.replace('cov_P_', '')}": v
                           for k, v in normal_op_diagnostics.items()}

    # Step 3: Solve normal equations (this creates the inverse normal operator)
    t0 = time.time()
    try:
        if hasattr(solver, '__call__'):
            # Direct solver case (e.g., LUSolver)
            inverse_normal_operator = solver(normal_operator)
        else:
            # Iterative solver case
            inverse_normal_operator = solver(normal_operator)
    except Exception as e:
        # Capture solver failure
        return {
            "normal_operator_time": normal_operator_time,
            "normal_op_matrix_time": normal_op_matrix_time,
            "solver_failed": True,
            "solver_error": str(e),
            **normal_op_diagnostics
        }
    t1 = time.time()
    solver_time = t1 - t0

    # Step 4: Compute posterior mean
    t0 = time.time()
    data_space = bayesian_inference.data_space
    model_space = bayesian_inference.model_space
    forward_operator = bayesian_inference.forward_problem.forward_operator
    prior_model_covariance = bayesian_inference.model_prior_measure.covariance

    # Calculate shifted data: d - A*mu_u - mu_e
    shifted_data = data_space.subtract(
        d_tilde, forward_operator(bayesian_inference.model_prior_measure.expectation)
    )
    if bayesian_inference.forward_problem.data_error_measure_set:
        shifted_data = data_space.subtract(
            shifted_data, bayesian_inference.forward_problem.data_error_measure.expectation
        )

    # Posterior mean update: mu_post = mu_u + C_u*A^T*C_d^-1*(shifted_data)
    mean_update = (
        prior_model_covariance @ forward_operator.adjoint @ inverse_normal_operator
    )(shifted_data)
    expectation = model_space.add(bayesian_inference.model_prior_measure.expectation, mean_update)
    t1 = time.time()
    posterior_mean_time = t1 - t0

    # Step 5: Compute posterior covariance
    t0 = time.time()
    # C_post = C_u - C_u*A^T*C_d^-1*A*C_u
    covariance = prior_model_covariance - (
        prior_model_covariance
        @ forward_operator.adjoint
        @ inverse_normal_operator
        @ forward_operator
        @ prior_model_covariance
    )
    t1 = time.time()
    posterior_cov_time = t1 - t0

    # Step 6: Convert covariance to dense matrix and diagnose
    t0 = time.time()
    C_M_matrix = covariance.matrix(dense=True)
    t1 = time.time()
    C_M_matrix_time = t1 - t0

    model_cov_diagnostics = _compute_eigenvalue_diagnostics(C_M_matrix)
    # Rename keys to be specific to model posterior covariance
    model_cov_diagnostics = {f"model_post_{k.replace('cov_P_', '')}": v
                           for k, v in model_cov_diagnostics.items()}

    # Step 7: Test measure construction
    measure_results = _test_measure_construction(M, C_M_matrix, expectation)
    model_measure_results = {f"model_{k}": v for k, v in measure_results.items()}

    # Prepare comprehensive output
    out = {
        "normal_operator_time": normal_operator_time,
        "normal_op_matrix_time": normal_op_matrix_time,
        "solver_time": solver_time,
        "posterior_mean_time": posterior_mean_time,
        "posterior_cov_time": posterior_cov_time,
        "C_M_matrix_time": C_M_matrix_time,
        "solver_failed": False,
        **normal_op_diagnostics,
        **model_cov_diagnostics,
        **model_measure_results
    }

    # Check for critical failures in model posterior
    critical_failures = []
    if not model_cov_diagnostics.get('model_post_is_positive_definite', True):
        min_eig = model_cov_diagnostics.get('model_post_min_eigenvalue', 'unknown')
        critical_failures.append(f"Model posterior covariance not PD (min eigenvalue: {min_eig})")

    if model_measure_results.get('model_cov_P_measure_failed', False):
        critical_failures.append(f"Model measure construction failed: {model_measure_results.get('model_cov_P_measure_error', 'unknown')}")

    if critical_failures:
        error_message = "; ".join(critical_failures)
        raise CovarianceMatrixError(error_message, diagnostics=out)

    return out


def compute_property_posterior_detailed(
        P, M, bayesian_inference, d_tilde, /, *, solver
):
    """Compute property posterior with detailed step-by-step diagnostics."""

    # Step 1: First compute model posterior with full diagnostics
    try:
        model_results = compute_model_posterior_detailed(M, d_tilde, bayesian_inference, solver=solver)
    except CovarianceMatrixError as e:
        # Model posterior failed, but we want to capture diagnostics
        return {
            "model_posterior_failed": True,
            "model_posterior_error": str(e),
            **e.diagnostics
        }

    # Step 2: Get the actual model posterior measure for property computation
    t0 = time.time()
    model_posterior = bayesian_inference.model_posterior_measure(d_tilde, solver)
    t1 = time.time()
    model_posterior_measure_time = t1 - t0

    # Step 3: Apply property mapping to get property posterior
    t0 = time.time()
    property_posterior = model_posterior.affine_mapping(operator=bayesian_inference.property_operator)
    t1 = time.time()
    property_mapping_time = t1 - t0

    # Step 4: Extract property posterior components
    p_tilde = property_posterior.expectation
    t0 = time.time()
    cov_P_matrix = property_posterior.covariance.matrix(dense=True)
    t1 = time.time()
    cov_P_matrix_time = t1 - t0

    # Step 5: Diagnose property posterior covariance
    property_cov_diagnostics = _compute_eigenvalue_diagnostics(cov_P_matrix)
    # Rename keys to be specific to property posterior
    property_cov_diagnostics = {f"property_post_{k.replace('cov_P_', '')}": v
                              for k, v in property_cov_diagnostics.items()}

    # Step 6: Test property measure construction
    property_measure_results = _test_measure_construction(P, cov_P_matrix, p_tilde)
    property_measure_results = {f"property_{k}": v for k, v in property_measure_results.items()}

    # Step 7: Check property operator properties
    t0 = time.time()
    try:
        # Get model posterior covariance for rank analysis
        model_cov = model_posterior.covariance
        model_cov_matrix = model_cov.matrix(dense=True)

        # Get property operator matrix if possible
        prop_op_matrix = bayesian_inference.property_operator.matrix(dense=True)

        # Compute property covariance via explicit formula: P @ C_model @ P^T
        explicit_prop_cov = prop_op_matrix @ model_cov_matrix @ prop_op_matrix.T

        # Compare with the affine mapping result
        diff_norm = np.linalg.norm(explicit_prop_cov - cov_P_matrix, ord='fro')

        # Diagnose property operator
        prop_op_svd = np.linalg.svd(prop_op_matrix, compute_uv=False)
        prop_op_rank = np.sum(prop_op_svd > 1e-12 * prop_op_svd[0])
        prop_op_cond = prop_op_svd[0] / max(prop_op_svd[-1], 1e-16)

        property_operator_diagnostics = {
            "property_operator_shape": prop_op_matrix.shape,
            "property_operator_rank": int(prop_op_rank),
            "property_operator_condition": float(prop_op_cond),
            "property_operator_min_singular": float(prop_op_svd[-1]),
            "property_operator_max_singular": float(prop_op_svd[0]),
            "explicit_vs_affine_diff_norm": float(diff_norm),
            "explicit_computation_successful": True
        }

        # Additional diagnostics on the explicit computation
        explicit_diagnostics = _compute_eigenvalue_diagnostics(explicit_prop_cov)
        explicit_diagnostics = {f"explicit_{k.replace('cov_P_', '')}": v
                              for k, v in explicit_diagnostics.items()}

    except Exception as e:
        property_operator_diagnostics = {
            "explicit_computation_successful": False,
            "explicit_computation_error": str(e)
        }
        explicit_diagnostics = {}

    t1 = time.time()
    property_operator_analysis_time = t1 - t0

    # Combine all results
    out = {
        # Timing information
        "model_posterior_measure_time": model_posterior_measure_time,
        "property_mapping_time": property_mapping_time,
        "cov_P_matrix_time": cov_P_matrix_time,
        "property_operator_analysis_time": property_operator_analysis_time,

        # Include all model posterior diagnostics
        **model_results,

        # Property-specific diagnostics
        **property_cov_diagnostics,
        **property_measure_results,
        **property_operator_diagnostics,
        **explicit_diagnostics
    }

    # Check for critical failures in property posterior
    critical_failures = []
    if not property_cov_diagnostics.get('property_post_is_positive_definite', True):
        min_eig = property_cov_diagnostics.get('property_post_min_eigenvalue', 'unknown')
        critical_failures.append(f"Property posterior covariance not PD (min eigenvalue: {min_eig})")

    if property_measure_results.get('property_cov_P_measure_failed', False):
        critical_failures.append(f"Property measure construction failed: {property_measure_results.get('property_cov_P_measure_error', 'unknown')}")

    if critical_failures:
        error_message = "; ".join(critical_failures)
        raise CovarianceMatrixError(error_message, diagnostics=out)

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
    'compute_model_posterior_detailed': ['setup_spatial_spaces', 'create_problems'],
    'compute_property_posterior_detailed': ['setup_spatial_spaces', 'create_problems'],
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
    'compute_model_posterior_detailed': compute_model_posterior_detailed,
    'compute_property_posterior_detailed': compute_property_posterior_detailed,
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
    'compute_model_posterior_detailed': ['solver'],
    'compute_property_posterior_detailed': ['solver'],
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
    'compute_model_posterior_detailed': [],  # returns only results dict
    'compute_property_posterior_detailed': [],  # returns only results dict
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
        elif func_name == 'compute_model_posterior_detailed':
            pos_args = [
                artifacts['M'],
                artifacts['d_tilde'],
                artifacts['bayesian_inference'],
            ]
        elif func_name == 'compute_property_posterior_detailed':
            pos_args = [
                artifacts['P'],
                artifacts['M'],
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
            'compute_model_posterior', 'compute_property_posterior',
            'compute_model_posterior_detailed',
            'compute_property_posterior_detailed'
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
