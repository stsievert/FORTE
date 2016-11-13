import FORTE.utils as utils
from FORTE.algorithms import STEConvexNucNormProjected
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
    m = 1000  # number of labels        

    # Generate centered data points
    Xtrue = np.random.randn(n,d);
    Xtrue = Xtrue -1./n*np.dot(np.ones((n,n)), Xtrue)
    Mtrue = np.dot(Xtrue, Xtrue.transpose())

    trace_norm = np.trace(Mtrue)
    Strain = utils.triplets(Xtrue, m)
    Stest = utils.triplets(Xtrue, m)
    
    Mhat = STEConvexNucNormProjected.computeEmbedding(n,d,
                                                      Strain,
                                                      max_iter_GD=2000,
                                                      epsilon=0.000001,
                                                      trace_norm=trace_norm)
    emp_loss_train = utils.empirical_lossM(Mhat, Strain)
    emp_loss_test  = utils.empirical_lossM(Mhat, Stest)
    print ('Empirical Training loss = {},   '
           'Empirical Test loss = {},   '
           'Relative Error = {} ').format(emp_loss_train,
                                          emp_loss_test,
                                          ( np.linalg.norm(Mtrue-Mhat,'fro')**2/
                                            np.linalg.norm(Mtrue,'fro')**2))
    if plot:
        _, Xhat = utils.transform_MtoX(Mhat,2)
        _, Xpro, _ = utils.procrustes(Xtrue, Xhat)
        plt.figure(1)

        plt.subplot(121)
        plt.plot(*zip(*Xtrue), marker='o', color='r', ls='')
        plt.subplot(122)
        plt.plot(*zip(*Xpro), marker='o', color='b', ls='')

        plt.show()

if __name__=='__main__':
    run(20, 2, plot=True)
