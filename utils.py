import pandas as pd
import matplotlib.pyplot as plt


def two_dim_visualization(features, labels, algorithm, filename):

    x1, x2 = features[:, 0], features[:, 1]

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'label': labels})
    groups = df.groupby('label')

    fig, ax = plt.subplots(figsize=(18, 10))

    for name, group in groups:
        ax.plot(group.x1, group.x2, marker='o', linestyle='', ms=5, alpha=0.6, markeredgecolor=None)
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    ax.set_title("{}".format(algorithm))
    plt.savefig(f'output/{filename}.png')
    plt.show()