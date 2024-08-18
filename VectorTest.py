import LinearInference.Vector as Vec
import numpy as np

n = 5
X = Vec.Space(n, lambda x : x, lambda x : x)

Xp = Vec.DualSpace(X)

x = X.Random()

cp = np.zeros(Xp.Dimension)
xp = Xp.FromComponents(cp)

print(Xp.ToComponents(xp))

