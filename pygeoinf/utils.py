import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_colourline(x, y, c):
    fig = plt.figure(1, figsize=(5, 5))
    col = cm.jet((c - np.min(c)) / (np.max(c) - np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=col[i])
        im = ax.scatter(x, y, c=c, s=0, cmap=cm.jet)
    fig.colorbar(im)
    plt.show()
