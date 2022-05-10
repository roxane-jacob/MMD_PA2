import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    with np.load(path) as f:
        X_train, y_train = f['train'], f['train_labels']
        X_test, y_test = f['test'], f['test_labels']
        X = np.concatenate((X_train.T, X_test.T))
        y = np.concatenate((y_train.flatten().astype(int), y_test.flatten().astype(int)))
        return X, y


def two_dim_visualization(data, labels, path):
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

    x1, x2 = data[:, 0], data[:, 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(x1, x2, c=labels, cmap='prism', alpha=1.)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.savefig(path, dpi=150, format='png')
    plt.show()
