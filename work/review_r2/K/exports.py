import ast, pathlib, re, importlib
import pygeoinf2 as gi
root = pathlib.Path("pygeoinf2")
# 1. names examples use via `gi.X` / `from pygeoinf2 import X` / `from pygeoinf2.sub import X`
used_top=set(); used_sub={}
for f in sorted((root/"examples").glob("*.py")):
    src=f.read_text(); tree=ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pygeoinf2"):
            for a in node.names:
                if node.module=="pygeoinf2": used_top.add(a.name)
                else: used_sub.setdefault(node.module,set()).add(a.name)
    for m in re.finditer(r"\bgi\.([A-Za-z_][A-Za-z0-9_]*)", src): used_top.add(m.group(1))
missing_top=[n for n in sorted(used_top) if not hasattr(gi,n)]
print("examples top-level names:", len(used_top), "missing:", missing_top)
bad=[]
for mod,names in sorted(used_sub.items()):
    m=importlib.import_module(mod)
    for n in sorted(names):
        if not hasattr(m,n): bad.append(f"{mod}.{n}")
print("examples deep imports:", sum(len(v) for v in used_sub.values()), "missing:", bad)
print("deep-import modules used by examples:", sorted(used_sub))
# 2. __all__ consistency
print("len(gi.__all__)=", len(gi.__all__), "unresolvable:", [n for n in gi.__all__ if not hasattr(gi,n)])
# 3. D-5: subpackages
for s in ["inference","numerics","plotting","geometry","probability","symmetric_space","algebra","backends","testing","compat"]:
    print(f"gi.{s}:", hasattr(gi,s))
# 4. K Must-2 list
for n in ["LinearForwardProblem","ForwardProblem","LinearGaussianInversion","MinimumNorm","LeastSquares","CGSolver","CholeskySolver","LUSolver","MinResSolver","plot","plot_corner","ProgressCallback","GMRESSolver","FlexibleCGSolver"]:
    print(f"  gi.{n}: {hasattr(gi,n)}")
# 5. K Must-3 export gaps
from pygeoinf2 import numerics, inference, algebra
from pygeoinf2.numerics import solvers, randomised
import pygeoinf2.backends.mfem as mf
print("solvers.__all__ has GMRES/FlexibleCG:", "GMRESSolver" in solvers.__all__, "FlexibleCGSolver" in solvers.__all__)
print("numerics has:", {n: hasattr(numerics,n) for n in ["GMRESSolver","FlexibleCGSolver","deflated_diagonal","ProximalBundleMethod","LevelBundleMethod","monotone_root","weighted_chi2_bound","resolve_solver","best_available_qp_solver"]})
print("randomised.__all__ has deflated_diagonal:", "deflated_diagonal" in randomised.__all__)
print("mfem.__all__:", mf.__all__, "missing of 8:", [n for n in ["MfemSpace","essential_dofs_of","operator_from_bilinear_form","operator_from_linear_forms","solver_from_bilinear_form","functional_from_linear_form","white_noise_load","matern_measure"] if n not in mf.__all__])
print("inference has FactoredNormalOperator:", hasattr(inference,"FactoredNormalOperator"), "algebra HilbertModule:", hasattr(algebra,"HilbertModule"), "require_module:", hasattr(algebra,"require_module"))
print("gi.MassWeightedSpace:", hasattr(gi,"MassWeightedSpace"), "algebra:", hasattr(algebra,"MassWeightedSpace"))
# 6. CURRENT_STATE names
import pygeoinf2.testing as T
print("testing:", {n:hasattr(T,n) for n in ["check_space","check_coordinates","check_operator","check_traits","check_white_noise"]})
from pygeoinf2 import symmetric_space as ss
print("geometry submodules:", {n: hasattr(ss,n) for n in ["sphere","circle","line","torus","plane","box"]})
import pygeoinf2.symmetric_space.sphere as sph
print("sphere.Lebesgue is class:", isinstance(sph.Lebesgue,type), "Sobolev:", isinstance(sph.Sobolev,type))
print("Sphere has to_colatitude_radians/to_latitude_degrees:", hasattr(sph.Sphere,"to_colatitude_radians"), hasattr(sph.Sphere,"to_latitude_degrees"))
print("grid_values/from_grid_values:", hasattr(sph.Sphere,"grid_values"), hasattr(sph.Sphere,"from_grid_values"))
print("LinearOperator.from_matrix:", hasattr(gi.LinearOperator,"from_matrix"), "from_derivative_matrix:", hasattr(gi.LinearOperator,"from_derivative_matrix"), "from_component_matrix:", hasattr(gi.LinearOperator,"from_component_matrix"), "from_formal_adjoint:", hasattr(gi.LinearOperator,"from_formal_adjoint"))
print("GaussianMeasure.from_standard_deviations:", hasattr(gi.GaussianMeasure,"from_standard_deviations"), "from_product:", hasattr(gi.GaussianMeasure,"from_product"), "condition:", hasattr(gi.GaussianMeasure,"condition"))
print("samples sig:", __import__("inspect").signature(gi.ProbabilityMeasure.samples))
print("matrix sig:", __import__("inspect").signature(gi.LinearOperator.matrix))
print("Sphere.__init__ sig:", __import__("inspect").signature(sph.Sphere.__init__))
print("estimator solve:", [n for n in dir(gi.LinearGaussianInversion) if "solve" in n or "result" in n])
print("check_operator sig:", __import__("inspect").signature(T.check_operator))
print("AffineSubspace.condition:", hasattr(gi.AffineSubspace,"condition"), hasattr(gi.AffineSubspace,"condition_gaussian_measure"), "LinearSubspace:", hasattr(gi.LinearSubspace,"condition"))
print("plot sig:", __import__("inspect").signature(gi.plot)); 
import pygeoinf2.plotting.sphere as ps; print("sphere plot sig:", __import__("inspect").signature(ps.plot)); print("plot_points sig:", __import__("inspect").signature(ps.plot_points))
print("plot_corner sig:", __import__("inspect").signature(gi.plot_corner))
print("plot_densities sig:", __import__("inspect").signature(gi.plot_densities))
print("sphere space methods:", [n for n in ["from_coefficient_operator","coefficient_operator","to_coefficients","from_coefficients","power_spectrum","random_domain_points","invariant_covariance_function","covariance_function","norm_scaled_sobolev_measure","sobolev_measure","heat_measure","invariant_measure","correlated_measure","correlated_measure_from_correlations","point_evaluation_operator","dirac","l2_products_operator","multiplication_operator","order_inclusion_operator","with_order","lift_formal_adjoint","stations"] if hasattr(sph.Sphere,n)])
print("sphere missing:", [n for n in ["from_coefficient_operator","random_domain_points","invariant_covariance_function","covariance_function","norm_scaled_sobolev_measure","lift_formal_adjoint"] if not hasattr(sph.Sphere,n)])
import pygeoinf2.symmetric_space.base as sb
print("base module names:", [n for n in dir(sb) if "lift" in n or "formal" in n or "Mass" in n])
