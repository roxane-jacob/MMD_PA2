from runner_toydata import runner_toydata
from runner_mnist import runner_mnist


if __name__ == '__main__':

    # set paths
    toy_tiny = 'data/toydata_tiny.csv'
    toy_large = 'data/toydata_large.csv'
    mnist = 'data/mnist.npz'

    # run procedures on all three datasets
    sgd_progress_linear_tiny, sgd_progress_rff_tiny = runner_toydata(toy_tiny, tiny=True)
    sgd_progress_linear_large, sgd_progress_rff_large = runner_toydata(toy_large, tiny=False)
    runner_mnist(mnist)
