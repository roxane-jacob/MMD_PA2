from runner_toydata_tiny import runner_toydata_tiny
from runner_toydata_large import runner_toydata_large
from runner_mnist import runner_mnist


if __name__ == '__main__':

    # set paths
    toy_tiny = 'data/toydata_tiny.csv'
    toy_large = 'data/toydata_large.csv'
    mnist = 'data/mnist.npz'

    # run procedures on all three datasets
    runner_toydata_tiny(toy_tiny)
    runner_toydata_large(toy_large)
    runner_mnist(mnist)
