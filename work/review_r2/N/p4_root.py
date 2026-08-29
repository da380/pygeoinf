from common import *
from pygeoinf2.numerics.root_find import monotone_root, Evaluation
from scipy.optimize import brentq
# a chi^2-like decreasing function of the damping
def chi(t): return 1000.0 / (1.0 + t)**1.3 + 50.0
target = 120.0
for initial in (1.0, 1e-3, 1e3):
    calls = Counter()
    def ev(t, prev):
        calls["n"] += 1
        return Evaluation(chi(t), None, 0)
    r = monotone_root(ev, target, initial=initial, rtol=1e-6)
    # prototype: same bracketing, then brentq in log t
    calls2 = Counter()
    def f(u): calls2["n"] += 1; return chi(np.exp(u)) - target
    lo, hi = np.log(r.bracket[0]), np.log(r.bracket[1])
    # count bracketing probes of v2: expansions until straddle from initial
    t = initial; nb = 1
    while chi(t) > target: t *= 10; nb += 1
    lo_b, hi_b = t/10, t
    if nb == 1:
        while chi(t) < target: t /= 10; nb += 1
        lo_b, hi_b = t, t*10
    u = brentq(f, np.log(lo_b), np.log(hi_b), xtol=2e-6)
    print(f"initial={initial}: v2 evaluations {r.evaluations} (root {r.argument:.6g}, conv {r.converged}); "
          f"bracket probes {nb} + brentq {calls2['n']} = {nb+calls2['n']} (root {np.exp(u):.6g})")
