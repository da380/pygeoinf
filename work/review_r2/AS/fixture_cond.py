from common import *
for d in (3, 10, 30, 60, 100, 150, 200, 400):
    s = make_dense_metric_space(d)
    print(d, f"cond(G) = {np.linalg.cond(s._gram):.3g}")
