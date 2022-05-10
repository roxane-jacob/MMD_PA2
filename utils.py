import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def two_dim_visualization(data, labels, algorithm, path):
    """
    Create a 2D scatter plot from the given input data with coloring corresponding to the labelling.

    Parameters
    ----------
    data : ndarray of shape (n samples, 2 features)
        Input data.
    labels : list (int)
        List of labels, corresponding to the respective labeling of the data points.
    path : str
        String referring to the path, where the plot shall be stored.
    """

    """
    x1, x2 = data[:, 0], data[:, 1]

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'label': labels})
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(18, 10))

    for name, group in groups:
        ax.plot(group.x1, group.x2, marker='o', linestyle='', ms=5, alpha=0.6, markeredgecolor=None)
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
    """

    x1, x2 = data[:, 0], data[:, 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(x1, x2, c=labels, cmap='prism', alpha=1.)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.savefig(path, dpi=150, format='png')
    plt.show()
