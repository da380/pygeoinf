#!/usr/bin/env python3
"""
Generate a visual dependency graph for pygeoinf classes.
Shows the dependency tree from most fundamental to most derived classes.
"""

try:
    import graphviz
except ImportError:
    print("Installing graphviz...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'graphviz'])
    import graphviz

def create_dependency_graph():
    """Create a visual dependency graph of pygeoinf classes including interval folder."""

    # Create a new directed graph
    dot = graphviz.Digraph(comment='PyGeoInf Class Dependencies (Complete)')
    dot.attr(rankdir='TB', size='16,20')
    dot.attr('node', shape='box', style='rounded,filled')

    # Define node styles by level
    fundamental_style = {'fillcolor': '#e8f4fd', 'color': '#1f77b4'}
    core_style = {'fillcolor': '#fff2cc', 'color': '#ff7f0e'}
    composition_style = {'fillcolor': '#e8f5e8', 'color': '#2ca02c'}
    problem_style = {'fillcolor': '#ffe8e8', 'color': '#d62728'}
    algorithm_style = {'fillcolor': '#f0e8ff', 'color': '#9467bd'}
    specialized_style = {'fillcolor': '#ffe0b3', 'color': '#8c564b'}
    interval_style = {'fillcolor': '#e6f3ff', 'color': '#0066cc'}  # Light blue for interval

    # FUNDAMENTAL LEVEL
    dot.node('random_matrix', 'random_matrix.py\n(algorithmic functions)', **fundamental_style)
    dot.node('HilbertSpace', 'HilbertSpace\n(ABC)', **fundamental_style)

    # CORE LEVEL
    dot.node('LinearForm', 'LinearForm', **core_style)
    dot.node('Operator', 'Operator', **core_style)
    dot.node('LinearOperator', 'LinearOperator', **core_style)
    dot.node('EuclideanSpace', 'EuclideanSpace', **core_style)
    dot.node('DiagonalLinearOperator', 'DiagonalLinearOperator', **core_style)

    # COMPOSITION LEVEL
    dot.node('HilbertSpaceDirectSum', 'HilbertSpaceDirectSum', **composition_style)
    dot.node('BlockLinearOperator', 'BlockLinearOperator\nfamily', **composition_style)
    dot.node('GaussianMeasure', 'GaussianMeasure', **composition_style)
    dot.node('LinearSolver', 'LinearSolver\nfamily', **composition_style)

    # PROBLEM FORMULATION LEVEL
    dot.node('ForwardProblem', 'ForwardProblem', **problem_style)
    dot.node('Inversion', 'Inversion\n(ABC)', **problem_style)

    # ALGORITHM LEVEL
    dot.node('LinearBayesianInversion', 'LinearBayesianInversion', **algorithm_style)
    dot.node('LinearLeastSquaresInversion', 'LinearLeastSquaresInversion', **algorithm_style)
    dot.node('LinearBayesianInference', 'LinearBayesianInference', **algorithm_style)

    # SPECIALIZED IMPLEMENTATIONS (original)
    dot.node('AbstractInvariantLebesgueSpace', 'AbstractInvariant\nLebesgueSpace', **specialized_style)
    dot.node('CircleSpaces', 'Circle.Lebesgue\nCircle.Sobolev\nCircleHelper', **specialized_style)

    # INTERVAL IMPLEMENTATIONS (new)
    # Core interval classes
    dot.node('IntervalDomain', 'IntervalDomain', **interval_style)
    dot.node('Function', 'Function\n(interval)', **interval_style)
    dot.node('BoundaryConditions', 'BoundaryConditions', **interval_style)

    # Interval Hilbert spaces
    dot.node('L2Space', 'L2Space\n(interval)', **interval_style)
    dot.node('SobolevSpace', 'Sobolev\n(interval)', **interval_style)

    # Interval operators
    dot.node('IntervalOperators', 'LaplacianOperator\nGradientOperator\nLaplacianInverse', **interval_style)

    # Function providers and utilities
    dot.node('FunctionProviders', 'FunctionProviders\n(Fourier, Splines,\nBumps, etc.)', **interval_style)
    dot.node('BasisProviders', 'BasisProviders\nSpectrumProviders', **interval_style)
    dot.node('FEMSolvers', 'GeneralFEMSolver', **interval_style)

    # Define dependencies (edges)
    dependencies = [
        # Core depends on fundamental
        ('HilbertSpace', 'LinearForm'),
        ('HilbertSpace', 'Operator'),
        ('HilbertSpace', 'EuclideanSpace'),
        ('Operator', 'LinearOperator'),
        ('random_matrix', 'LinearOperator'),
        ('LinearOperator', 'DiagonalLinearOperator'),

        # Composition depends on core
        ('HilbertSpace', 'HilbertSpaceDirectSum'),
        ('LinearOperator', 'BlockLinearOperator'),
        ('HilbertSpaceDirectSum', 'BlockLinearOperator'),
        ('HilbertSpace', 'GaussianMeasure'),
        ('LinearOperator', 'GaussianMeasure'),
        ('LinearOperator', 'LinearSolver'),

        # Problem formulation depends on composition
        ('LinearOperator', 'ForwardProblem'),
        ('GaussianMeasure', 'ForwardProblem'),
        ('ForwardProblem', 'Inversion'),

        # Algorithms depend on problem formulation
        ('Inversion', 'LinearBayesianInversion'),
        ('ForwardProblem', 'LinearBayesianInversion'),
        ('GaussianMeasure', 'LinearBayesianInversion'),
        ('Inversion', 'LinearLeastSquaresInversion'),
        ('ForwardProblem', 'LinearLeastSquaresInversion'),
        ('LinearSolver', 'LinearLeastSquaresInversion'),
        ('LinearBayesianInversion', 'LinearBayesianInference'),

        # Specialized implementations (geometric)
        ('HilbertSpace', 'AbstractInvariantLebesgueSpace'),
        ('LinearOperator', 'AbstractInvariantLebesgueSpace'),
        ('GaussianMeasure', 'AbstractInvariantLebesgueSpace'),
        ('AbstractInvariantLebesgueSpace', 'CircleSpaces'),

        # INTERVAL DEPENDENCIES
        # Basic interval classes depend on fundamentals
        ('HilbertSpace', 'L2Space'),
        ('L2Space', 'SobolevSpace'),
        ('HilbertSpace', 'IntervalDomain'),
        ('IntervalDomain', 'Function'),
        ('L2Space', 'Function'),

        # Providers depend on interval basics
        ('Function', 'FunctionProviders'),
        ('L2Space', 'BasisProviders'),

        # Operators depend on interval spaces
        ('LinearOperator', 'IntervalOperators'),
        ('L2Space', 'IntervalOperators'),
        ('BoundaryConditions', 'IntervalOperators'),

        # FEM depends on operators and boundary conditions
        ('IntervalOperators', 'FEMSolvers'),
        ('BoundaryConditions', 'FEMSolvers'),
        ('L2Space', 'FEMSolvers'),

        # Interval classes integrate with main framework
        ('L2Space', 'ForwardProblem'),
        ('IntervalOperators', 'ForwardProblem'),
        ('SobolevSpace', 'GaussianMeasure'),
        ('IntervalOperators', 'GaussianMeasure'),
    ]

    # Add edges
    for source, target in dependencies:
        dot.edge(source, target)

    # Add level clustering using subgraphs
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='FUNDAMENTAL LEVEL', style='dashed', color='blue')
        c.node('random_matrix')
        c.node('HilbertSpace')

    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='CORE LEVEL', style='dashed', color='orange')
        c.node('LinearForm')
        c.node('Operator')
        c.node('LinearOperator')
        c.node('EuclideanSpace')
        c.node('DiagonalLinearOperator')

    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='COMPOSITION LEVEL', style='dashed', color='green')
        c.node('HilbertSpaceDirectSum')
        c.node('BlockLinearOperator')
        c.node('GaussianMeasure')
        c.node('LinearSolver')

    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='PROBLEM FORMULATION LEVEL', style='dashed', color='red')
        c.node('ForwardProblem')
        c.node('Inversion')

    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='ALGORITHM LEVEL', style='dashed', color='purple')
        c.node('LinearBayesianInversion')
        c.node('LinearLeastSquaresInversion')
        c.node('LinearBayesianInference')

    with dot.subgraph(name='cluster_5') as c:
        c.attr(label='GEOMETRIC SPACES', style='dashed', color='brown')
        c.node('AbstractInvariantLebesgueSpace')
        c.node('CircleSpaces')

    with dot.subgraph(name='cluster_6') as c:
        c.attr(label='INTERVAL IMPLEMENTATIONS', style='dashed', color='#0066cc')
        c.node('IntervalDomain')
        c.node('Function')
        c.node('BoundaryConditions')
        c.node('L2Space')
        c.node('SobolevSpace')
        c.node('IntervalOperators')
        c.node('FunctionProviders')
        c.node('BasisProviders')
        c.node('FEMSolvers')

    return dot

    # Define dependencies (edges)
    dependencies = [
        # Core depends on fundamental
        ('HilbertSpace', 'LinearForm'),
        ('HilbertSpace', 'Operator'),
        ('HilbertSpace', 'EuclideanSpace'),
        ('Operator', 'LinearOperator'),
        ('random_matrix', 'LinearOperator'),
        ('LinearOperator', 'DiagonalLinearOperator'),

        # Composition depends on core
        ('HilbertSpace', 'HilbertSpaceDirectSum'),
        ('LinearOperator', 'BlockLinearOperator'),
        ('HilbertSpaceDirectSum', 'BlockLinearOperator'),
        ('HilbertSpace', 'GaussianMeasure'),
        ('LinearOperator', 'GaussianMeasure'),
        ('LinearOperator', 'LinearSolver'),

        # Problem formulation depends on composition
        ('LinearOperator', 'ForwardProblem'),
        ('GaussianMeasure', 'ForwardProblem'),
        ('ForwardProblem', 'Inversion'),

        # Algorithms depend on problem formulation
        ('Inversion', 'LinearBayesianInversion'),
        ('ForwardProblem', 'LinearBayesianInversion'),
        ('GaussianMeasure', 'LinearBayesianInversion'),
        ('Inversion', 'LinearLeastSquaresInversion'),
        ('ForwardProblem', 'LinearLeastSquaresInversion'),
        ('LinearSolver', 'LinearLeastSquaresInversion'),
        ('LinearBayesianInversion', 'LinearBayesianInference'),

        # Specialized implementations
        ('HilbertSpace', 'AbstractInvariantLebesgueSpace'),
        ('LinearOperator', 'AbstractInvariantLebesgueSpace'),
        ('GaussianMeasure', 'AbstractInvariantLebesgueSpace'),
        ('AbstractInvariantLebesgueSpace', 'CircleSpaces'),
    ]

    # Add edges
    for source, target in dependencies:
        dot.edge(source, target)

    # Add level clustering using subgraphs
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='FUNDAMENTAL LEVEL', style='dashed', color='blue')
        c.node('random_matrix')
        c.node('HilbertSpace')

    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='CORE LEVEL', style='dashed', color='orange')
        c.node('LinearForm')
        c.node('Operator')
        c.node('LinearOperator')
        c.node('EuclideanSpace')
        c.node('DiagonalLinearOperator')

    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='COMPOSITION LEVEL', style='dashed', color='green')
        c.node('HilbertSpaceDirectSum')
        c.node('BlockLinearOperator')
        c.node('GaussianMeasure')
        c.node('LinearSolver')

    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='PROBLEM FORMULATION LEVEL', style='dashed', color='red')
        c.node('ForwardProblem')
        c.node('Inversion')

    with dot.subgraph(name='cluster_4') as c:
        c.attr(label='ALGORITHM LEVEL', style='dashed', color='purple')
        c.node('LinearBayesianInversion')
        c.node('LinearLeastSquaresInversion')
        c.node('LinearBayesianInference')

    with dot.subgraph(name='cluster_5') as c:
        c.attr(label='SPECIALIZED IMPLEMENTATIONS', style='dashed', color='brown')
        c.node('AbstractInvariantLebesgueSpace')
        c.node('CircleSpaces')

    return dot

def create_simplified_graph():
    """Create a simplified version showing just key relationships."""

    dot = graphviz.Digraph(comment='PyGeoInf Key Dependencies (Simplified)')
    dot.attr(rankdir='TB', size='12,14')
    dot.attr('node', shape='ellipse', style='filled')

    # Core abstractions
    dot.node('HilbertSpace', 'HilbertSpace\n(foundation)', fillcolor='lightblue')
    dot.node('LinearOperator', 'LinearOperator\n(workhorse)', fillcolor='lightgreen')
    dot.node('GaussianMeasure', 'GaussianMeasure\n(probability)', fillcolor='lightyellow')

    # Problem setup
    dot.node('ForwardProblem', 'ForwardProblem\n(physics)', fillcolor='lightcoral')

    # Solution methods
    dot.node('Bayesian', 'Bayesian Methods', fillcolor='lightpink')
    dot.node('Optimization', 'Optimization Methods', fillcolor='lightgray')

    # Geometric extensions
    dot.node('Geometry', 'Geometric Spaces\n(Circle, Sphere)', fillcolor='wheat')

    # Interval implementations
    dot.node('Interval', 'Interval Spaces\n(1D Functions, FEM)', fillcolor='lightcyan')

    # Key dependencies
    dot.edge('HilbertSpace', 'LinearOperator')
    dot.edge('HilbertSpace', 'GaussianMeasure')
    dot.edge('LinearOperator', 'GaussianMeasure')
    dot.edge('LinearOperator', 'ForwardProblem')
    dot.edge('GaussianMeasure', 'ForwardProblem')
    dot.edge('ForwardProblem', 'Bayesian')
    dot.edge('ForwardProblem', 'Optimization')
    dot.edge('HilbertSpace', 'Geometry')
    dot.edge('LinearOperator', 'Geometry')
    dot.edge('GaussianMeasure', 'Geometry')

    # Interval dependencies
    dot.edge('HilbertSpace', 'Interval')
    dot.edge('LinearOperator', 'Interval')
    dot.edge('GaussianMeasure', 'Interval')
    dot.edge('Interval', 'ForwardProblem')

    return dot

if __name__ == '__main__':
    print("Generating pygeoinf dependency graphs (including interval folder)...")

    # Create detailed graph
    detailed_graph = create_dependency_graph()
    detailed_graph.render('pygeoinf_dependencies_detailed', format='png', cleanup=True)
    detailed_graph.render('pygeoinf_dependencies_detailed', format='svg', cleanup=True)

    # Create simplified graph
    simple_graph = create_simplified_graph()
    simple_graph.render('pygeoinf_dependencies_simple', format='png', cleanup=True)
    simple_graph.render('pygeoinf_dependencies_simple', format='svg', cleanup=True)

    print("✓ Generated dependency graphs:")
    print("  - pygeoinf_dependencies_detailed.png/svg (complete class hierarchy + interval)")
    print("  - pygeoinf_dependencies_simple.png/svg (key relationships + interval)")
    print("\nTo view: open the .png files or use a browser for .svg files")
    print("\nNEW: Interval folder classes are now included!")
    print("  - L2Space, SobolevSpace: Hilbert spaces for 1D functions")
    print("  - IntervalDomain, Function: geometric domain and function representation")
    print("  - FunctionProviders: factories for common function families")
    print("  - IntervalOperators: Laplacian, gradient, FEM solvers")
