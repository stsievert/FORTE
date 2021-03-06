import time
import FORTE.utils as utils
from FORTE.objectives import CrowdKernelLoss, LogisticLoss, L1Loss
from FORTE.algorithms import (RankdPGD,
                              NuclearNormPGD,
                              FactoredGradient,
                              FactoredGradientSGD,
                              FactoredGradientSGDHingeLoss,
                              CrowdKernel,
                              RankdPGDHingeLoss,
                              L1Embedding)
import blackbox
import numpy as np
import matplotlib.pyplot as plt

def logistic_noise(X, q):
  score = utils.triplet_scoreX(X,q)
  return 1./(1. + np.exp(-score))

mu = .001
def ck_noise(X, q):
  return CrowdKernelLoss.getCrowdKernelTripletProbability(X[q[0]], X[q[1]], X[q[2]], mu)

def all_triplets(X):
    S = []
    n = X.shape[0]; d = X.shape[1]
    for i in range(n):
        for j in range(n):
            for k in range(j):
                if i!=j and k!=i and k!=j:
                    score = utils.triplet_scoreX(X, [i,j,k])
                    if score < 0:
                        S.append([i, k, j])
                    else:
                        S.append([i, j, k])
    return S

def run_RankdPGD(n, d, plot=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    n = n
    d = d
    m = n*n*n
    # 20 * n * d * np.log(n)  # number of labels

    # Generate centered data points
    Xtrue = np.random.randn(n, d)
    #Xtrue = Xtrue - 1. / n * np.dot(np.ones((n, n)),  Xtrue)
    Mtrue = np.dot(Xtrue, Xtrue.transpose())
    print "Bayes Loss", LogisticLoss.getLossX(Xtrue, all_triplets(Xtrue))
   
    # Strain = utils.triplets(Xtrue, m, logistic_noise)
    Strain = utils.triplets(Xtrue, m)
    print "Xtrue Empirical Loss", utils.empirical_lossX(Xtrue, Strain)
    print "Strain Log Loss", LogisticLoss.getLossX(Xtrue, Strain)

    # Stest = utils.triplets(Xtrue, m, logistic_noise)
    Stest = utils.triplets(Xtrue, m)
    Mhat = RankdPGD.computeEmbedding(n, d,
                                     Strain,
                                     max_iter_GD=100,
                                     num_random_restarts=0,
                                     epsilon=0.00001,
                                     verbose=True)
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

def run_RankdPGDHingeLoss(n, d, plot=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    n = n
    d = d
    m = n*n*n
    # noise_func = logistic_noise   # change it to logistic noise
    noise_func = None
    # 20 * n * d * np.log(n)  # number of labels

    # Generate centered data points
    Xtrue = np.random.rand(n, d)
    Xtrue = Xtrue - 1. / n * np.dot(np.ones((n, n)),  Xtrue)
    Mtrue = np.dot(Xtrue, Xtrue.transpose())

    Strain = utils.triplets(Xtrue, m, noise_func=noise_func)
    Stest = utils.triplets(Xtrue, m, noise_func=noise_func)
    Mhat = RankdPGDHingeLoss.computeEmbedding(n, d,
                                              Strain,
                                              max_iter_GD=2000,
                                              num_random_restarts=0,
                                              epsilon=1e-10,
                                              verbose=True)
    emp_loss_train = utils.empirical_lossM(Mhat, Strain)
    emp_loss_test = utils.empirical_lossM(Mhat, Stest)
    print('performance:')
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


def run_NuclearNormPGD(n, d, plot=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    print('Here')
    n = n
    d = d
    m = n*n
    Xtrue = np.random.rand(n, d)/np.sqrt(d)
    # Xtrue = Xtrue - 1. / n * np.dot(np.ones((n, n)),  Xtrue)
    Mtrue = np.dot(Xtrue, Xtrue.transpose())
    max_norm = np.max([np.linalg.norm(Xtrue[i])
                       for i in range(Xtrue.shape[0])])
    Strain = utils.triplets(Xtrue, m, noise_func=None)
    Stest = utils.triplets(Xtrue, m, noise_func=None)

    ts = time.time()
    b = 0
    total = 0
    for i in range(n):
        for j in range(i):
            for k in range(n):
                if i!=j and i!=k and j!=k:
                    p = 1/(1+np.exp(-(Mtrue[k,k] -2*Mtrue[i,k] + 2*Mtrue[i,j] - Mtrue[j,j])))
                    b += -p*np.log(p)
                    total += 1 
    b = b/total
    print time.time() - ts
    print('Bayes loss: ', b)
    Mhat = NuclearNormPGD.computeEmbedding(n, d,
                                           Strain,
                                           max_iter_GD=200,
                                           trace_norm=4*np.trace(Mtrue),
                                           epsilon=1e-12,
                                           verbose=True)
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
    m = n*n*n#10 * n * d * np.log(n)  # number of labels

    # Generate centered data points
    Xtrue = np.random.randn(n, d)
    Xtrue = Xtrue - 1. / n * np.dot(np.ones((n, n)), Xtrue)
    max_norm = np.max([np.linalg.norm(Xtrue[i])
                       for i in range(Xtrue.shape[0])])

    Strain = utils.triplets(Xtrue, m, noise_func=logistic_noise)
    Stest = utils.triplets(Xtrue, m, noise_func=logistic_noise)

    Xhat = FactoredGradientSGDHingeLoss.computeEmbedding(n, d, Strain,
                                                num_random_restarts=0,
                                                max_num_passes_SGD=16,
                                                max_iter_GD=100,
                                                max_norm=max_norm,
                                                epsilon=5e-3, verbose=True)

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


def run_CK(n, d, plot=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    n = n
    d = d
    m = int(10 * n * d * np.log(n))  # number of labels
    # Generate centered data points
    Xtrue = np.random.randn(n, d)
    print "CK Loss - Bayes", CrowdKernelLoss.getLoss(Xtrue, all_triplets(Xtrue))
    # Strain = utils.triplets(Xtrue, m, ck_noise)
    Strain = utils.triplets(Xtrue, m, None)
    print "Empirical Loss on Strain", utils.empirical_lossX(Xtrue, Strain)
    print "CK Loss on Strain", CrowdKernelLoss.getLoss(Xtrue, Strain)

    # Stest = utils.triplets(Xtrue, m, ck_noise)
    Stest = utils.triplets(Xtrue, m, None)
    Xhat = CrowdKernel.computeEmbedding(n, d, 
                                        Strain,
                                        mu=mu,
                                        num_random_restarts=0,
                                        max_num_passes_SGD=16,
                                        max_iter_GD=50,
                                        max_norm=1., 
                                        epsilon=0.0001, verbose=True)

    emp_loss_train = utils.empirical_lossX(Xhat, Strain)
    emp_loss_test = utils.empirical_lossX(Xhat, Stest)
    print ('Empirical Training loss = {},   '
           'CK Loss on all Triplets = {},').format(emp_loss_train, 
                                                   CrowdKernelLoss.getLoss(Xhat, all_triplets(Xtrue)))

    if plot:
        _, Xpro, _ = utils.procrustes(Xtrue, Xhat)
        plt.figure(1)
        plt.subplot(121)
        plt.plot(*zip(*Xtrue), marker='o', color='r', ls='')
        plt.subplot(122)
        plt.plot(*zip(*Xpro), marker='o', color='b', ls='')

        plt.show()

def run_L1Embedding(n, d, plot=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    n = n
    d = d
    m = n*n*n
    # 20 * n * d * np.log(n)  # number of labels

    # Generate centered data points
    Xtrue = np.random.randn(n, d)
    #Xtrue = Xtrue - 1. / n * np.dot(np.ones((n, n)),  Xtrue)
    Mtrue = np.dot(Xtrue, Xtrue.transpose())
    print "Bayes Loss", L1Loss.l1Loss(Xtrue, all_triplets(Xtrue))
   
    # Strain = utils.triplets(Xtrue, m, logistic_noise)
    Strain = utils.triplets(Xtrue, m)
    print "Xtrue Empirical Loss", utils.empirical_lossX(Xtrue, Strain)
    print "Strain L1 Loss", L1Loss.l1Loss(Xtrue, Strain)

    # Stest = utils.triplets(Xtrue, m, logistic_noise)
    Stest = utils.triplets(Xtrue, m)
    Xhat = L1Embedding.computeEmbedding(n, d,
                                         Strain,
                                         max_iter_GD=500,
                                         num_random_restarts=0,
                                         epsilon=0.00001,
                                         verbose=True)
    emp_loss_train = utils.empirical_lossX(Xhat, Strain)
    emp_loss_test = utils.empirical_lossX(Xhat, Stest)
    _, Xpro, _ = utils.procrustes(Xtrue, Xhat)
    print ('Empirical Training loss = {},   '
           'Empirical Test loss = {},   '
           'Relative Error = {} ').format(emp_loss_train,
                                          emp_loss_test,
                                          (np.linalg.norm(Xtrue - Xpro, 'fro')**2 /
                                           np.linalg.norm(Xtrue, 'fro')**2))
    if plot:
        plt.figure(1)
        plt.subplot(121)
        plt.plot(*zip(*Xtrue), marker='o', color='r', ls='')
        plt.subplot(122)
        plt.plot(*zip(*Xpro), marker='o', color='b', ls='')
        plt.show()

if __name__ == '__main__':
    np.random.seed(421)
    # blackbox.set_experiment('TimeTest')
    # blackbox.takeoff(('n=30, d=2, max_iters=200'
    #                   'epsilon=.00001, 5 runs, NucNorm'), force=True)
    # times = []
    # for i in range(1):
    #     ts = time.time()
    #     run_RankdPGD(20, 2, plot=True)
    #     times.append(time.time() - ts)
    # print 'average execution time - RankdPGD', sum(times) / len(times)
    # blackbox.land()


    # blackbox.takeoff(('n=30, d=2, max_iters=200'
    #                   'epsilon=.00001, 5 runs, NucNorm'), force=True)
    # times = []
    # for i in range(1):
    #     ts = time.time()
    #     run_CK(60, 3, plot=False)
    #     times.append(time.time() - ts)
    # print 'average execution time - RankdCK', sum(times) / len(times)
    blackbox.land()

    # blackbox.takeoff('n=30, d=2, m=1000 10 runs, NucNorm', force=True)
    # times = []
    # print('Running')
    # for i in range(1):
    #     ts = time.time()
    #     run_NuclearNormPGD(200, 10, plot=False)
    #     times.append(time.time() - ts)
    # print 'average execution time - NuclearNormPGD', sum(times) / len(times)
    # blackbox.land()

    blackbox.takeoff('n=30, d=2, m=1000 10 runs, FG', force=True)
    times = []
    for i in range(1):
        ts = time.time()
        run_FG(20, 2, plot=False)
        times.append(time.time() - ts)
    print 'average execution time - FactoredGradient', sum(times) / len(times)
    blackbox.land()
    
    # blackbox.takeoff('n=30, d=2, m=1000 10 runs, CK', force=True)
    # times = []
    # for i in range(1):
    #     ts = time.time()
    #     run_CK(20, 2, plot=True)
    #     times.append(time.time() - ts)
    # print 'average execution time - CK', sum(times) / len(times)
    # blackbox.land()

    # blackbox.takeoff('n=30, d=2, m=1000 10 runs, L1', force=True)
    # times = []
    # for i in range(1):
    #     ts = time.time()
    #     run_L1Embedding(20, 2, plot=False)
    #     times.append(time.time() - ts)
    # print 'average execution time - FactoredGradient', sum(times) / len(times)
    # blackbox.land()

# n = 30, d = 2, 1000 triplets
# blackbox no save verbose=True: 56.795
# blackbox no save verbose=False: 54.0928
# blackbox save verbose=True: 58.45
# blackbox save verbose=False: 56.52:
# no blackbox: 53.3
