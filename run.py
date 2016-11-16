import time
import FORTE.utils as utils
from FORTE.algorithms import NuclearNormProjected, FactoredGradient, FactoredGradientSGD
import blackbox
import numpy as np
import matplotlib.pyplot as plt


def run(n, d, plot=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    n = n
    d = d
    m = 10 * n * d * np.log(n)  # number of labels

    # Generate centered data points
    Xtrue = np.random.randn(n, d)
    Xtrue = Xtrue - 1. / n * np.dot(np.ones((n, n)),  Xtrue)
    Mtrue = np.dot(Xtrue, Xtrue.transpose())

    trace_norm = np.trace(Mtrue)
    Strain = utils.triplets(Xtrue, m)
    Stest = utils.triplets(Xtrue, m)

    Mhat = NuclearNormProjected.computeEmbedding(n, d,
                                                 Strain,
                                                 max_iter_GD=10000,
                                                 epsilon=0.000001,
                                                 trace_norm=trace_norm)
    emp_loss_train = utils.empirical_lossM(Mhat, Strain)
    emp_loss_test = utils.empirical_lossM(Mhat, Stest)
    print ('Empirical Training loss = {},   '
           'Empirical Test loss = {},   '
           'Relative Error = {} ').format(emp_loss_train,
                                          emp_loss_test,
                                          (np.linalg.norm(Mtrue - Mhat, 'fro')**2 /
                                           np.linalg.norm(Mtrue, 'fro')**2))
    if plot:
        _, Xhat = utils.transform_MtoX(Mhat, 2)
        _, Xpro, _ = utils.procrustes(Xtrue, Xhat)
        plt.figure(1)

        plt.subplot(121)
        plt.plot(*zip(*Xtrue), marker='o', color='r', ls='')
        plt.subplot(122)
        plt.plot(*zip(*Xpro), marker='o', color='b', ls='')

        plt.show()


def run_FG(n, d, plot=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    n = n
    d = d
    m = 10 * n * d * np.log(n)  # number of labels

    # Generate centered data points
    Xtrue = np.random.randn(n, d)
    Xtrue = Xtrue - 1. / n * np.dot(np.ones((n, n)), Xtrue)

    Strain = utils.triplets(Xtrue, m)
    Stest = utils.triplets(Xtrue, m)

    Xhat = FactoredGradient.computeEmbedding(n, d, Strain,
                                             num_random_restarts=0,
                                             max_num_passes_SGD=16, max_iter_GD=50,
                                             max_norm=1, epsilon=0.01, verbose=False)

    emp_loss_train = utils.empirical_lossX(Xhat, Strain)
    emp_loss_test = utils.empirical_lossX(Xhat, Stest)
    print ('Empirical Training loss = {},   '
           'Empirical Test loss = {},').format(emp_loss_train,
                                          emp_loss_test)

    if plot:
        _, Xpro, _ = utils.procrustes(Xtrue, Xhat)
        plt.figure(1)
        plt.subplot(121)
        plt.plot(*zip(*Xtrue), marker='o', color='r', ls='')
        plt.subplot(122)
        plt.plot(*zip(*Xpro), marker='o', color='b', ls='')

        plt.show()

if __name__ == '__main__':
    blackbox.set_experiment('TimeTest')

    # blackbox.takeoff('n=30, d=2, m=1000 10 runs, NucNorm', force=True)
    # times = []
    # for i in range(10):
    #     ts = time.time()
    #     run(100, 2, plot=False)
    #     times.append(time.time() - ts)
    # print 'average execution time - NucNormProjected', sum(times) / len(times)
    # blackbox.land()

    blackbox.takeoff('n=30, d=2, m=1000 10 runs, NucNorm', force=True)
    times = []
    for i in range(10):
        ts = time.time()
        run_FG(100, 2, plot=False)
        times.append(time.time() - ts)
    print 'average execution time - FactoredGradient', sum(times) / len(times)
    blackbox.land()


# n = 30, d = 2, 1000 triplets
# blackbox no save verbose=True: 56.795
# blackbox no save verbose=False: 54.0928
# blackbox save verbose=True: 58.45
# blackbox save verbose=False: 56.52:
# no blackbox: 53.3
