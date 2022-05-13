from runner_toydata import runner_toydata
from runner_mnist import runner_mnist
from utils import plot_parallel_runtimes, plot_parallel_accuracies


if __name__ == '__main__':

    # set paths
    toy_tiny = 'data/toydata_tiny.csv'
    toy_large = 'data/toydata_large.csv'
    mnist = 'data/mnist.npz'

    # run procedures on all three datasets
    number_of_machines_tiny, parallel_runtimes_tiny, parallel_accuracies_tiny = runner_toydata(toy_tiny, tiny=True)
    number_of_machines_large, parallel_runtimes_large, parallel_accuracies_large = runner_toydata(toy_large, tiny=False)
    number_of_machines_mnist, parallel_runtimes_mnist, parallel_accuracies_mnist = runner_mnist(mnist)

    # plot runtimes and accuracies from parallel screening
    plot_parallel_runtimes(number_of_machines_tiny, number_of_machines_large, number_of_machines_mnist,
                           parallel_runtimes_tiny, parallel_runtimes_large, parallel_runtimes_mnist,
                           'output/parallel_runtimes.png')
    plot_parallel_accuracies(number_of_machines_tiny, number_of_machines_large, number_of_machines_mnist,
                             parallel_accuracies_tiny, parallel_accuracies_large, parallel_accuracies_mnist,
                             'output/parallel_accuracies.png')
