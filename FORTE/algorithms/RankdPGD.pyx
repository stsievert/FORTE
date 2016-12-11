import cython
import time, math, sys
sys.path.append('../..')

import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
import blackbox

import FORTE.objectives.LogisticLoss as LL
import FORTE.utils as utils


norm = np.linalg.norm
floor = math.floor
ceil = math.ceil

@blackbox.record
def computeEmbedding(int n, int d, S,
                     num_random_restarts=0,
                     max_iter_GD=50,
                     max_norm=1,
                     epsilon=0.01,
                     verbose=False):
    """
    Computes STE MLE of a Gram matrix M using the triplets of S.
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n].
        (int) num_random_restarts : number of random restarts (nonconvex
        optimization, may converge to local minima). E.g., 9 random restarts
        means take the best of 10 runs of the optimization routine.
        (int) max_iter_GD: maximum number of GD iteration (default equals 500)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, smaller means more accurate (default = 0.01)
        (boolean) verbose : outputs some progress (default equals False)

    Outputs:
        (numpy.ndarray) M : output Gram matrix
    """
    cdef np.ndarray M, M_old
    cdef np.ndarray J = np.eye(n)-np.ones((n,n))*1./n
    emp_loss_old = float('inf')
    for restart in range(num_random_restarts+1):
        # Initialization of centered Gram Matrix
        M = np.random.randn(n,n)
        M = (M+M.transpose())/2

        M = np.dot(np.dot(J,M),J)
        ts = time.time()
        M_new  = computeEmbeddingWithGD(M, S, d,
                                        max_iters=max_iter_GD,
                                        max_norm=max_norm,
                                        epsilon=epsilon,
                                        verbose=verbose)
        te_gd = time.time()-ts
        emp_loss_new = utils.empirical_lossM(M_new, S)
        log_loss = LL.getLossM(M_new, S)
        if emp_loss_new < emp_loss_old:
            M_old = M_new
            emp_loss_old = emp_loss_new
        
            blackbox.logdict({'restart':restart,
                              'emp_loss':emp_loss_new,
                              'log_loss':LL.getLossM(M_new, S),
                              'duration':te_gd})
            blackbox.save(verbose=verbose)
    return M_old

def computeEmbeddingWithGD(np.ndarray M, S,
                           int d,
                           max_iters=50,
                           max_norm=1.,
                           epsilon=0.01,
                           c1=.000001,
                           rho=0.5,
                           verbose=False):
    """
    Performs gradient descent with geometric amarijo line search (with parameter c1)
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Implements linesearch from Fukushima and Mine, 
    "A generalized proximal point algorithm for certain non-convex minimization problems" 
    
    Inputs:
        (numpy.ndarray) M : input Gram matrix
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) d: embedding dimension d
        (int) max_iters : maximum number of iterations of SGD (default equals 40*len(S))
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, (default = 0.001)
                          exits if new iteration does not differ in fro norm by more than epsilo 
        (float) c1 : Amarijo stopping condition parameter (default equals 0.0001)
        (float) rho : Backtracking line search parameter (default equals 0.5)
        (boolean) verbose : output iteration progress or not (default equals False)

    Outputs:
        (numpy.ndarray) M : output Gram matrix
    Usage:
        M = computeEmbeddingWithGD(M,S, 2)
    """
    cdef int n = M.shape[0]
    cdef double alpha = 1
    cdef int t, inner_t;
    
    cdef np.ndarray G, M_k, d_k, normG, normM;
    cdef double emp_loss, log_loss, rel_max_grad 

    while t < max_iters:
        t+=1
        # get gradient and stopping-time statistics
        G = LL.getFullGradientM(M, S)
        normG = norm(G, axis=0)
        normM = norm(M, axis=0)
        rel_max_grad = np.sqrt( max(normG)/ sum(normM)/n)
        # perform backtracking line search
        log_loss = LL.getLossM(M,S)
        norm_grad_sq = sum(normG)

        alpha = 1.3*alpha
        M_k = projected(M-alpha*G, d)
        log_loss_k = LL.getLossM( M_k , S)
        d_k = M_k - M
        Delta = norm(d_k, 'fro')        

        if Delta<epsilon or rel_max_grad<epsilon:
            blackbox.logdict({'iter':t,
                              'emp_loss':emp_loss,
                              'log_loss':log_loss,
                              'Delta':Delta,
                              'G_norm':norm_grad_sq,
                              'inner_t':inner_t})
            blackbox.save(verbose=verbose)
            break

        beta = rho
        inner_t = 0
        while log_loss_k > log_loss - c1*alpha*norm_grad_sq and inner_t < 10:
            beta = beta*beta
            log_loss = LL.getLossM(M + beta*d_k ,S)
            inner_t += 1
        if inner_t > 0:
            alpha = max(.1, alpha*rho)
        M = projected(M+beta*d_k, d)

        blackbox.logdict({'iter':t,
                          'emp_loss':utils.empirical_lossM(M, S),
                          'log_loss':log_loss,
                          'Delta':Delta,
                          'G_norm':norm_grad_sq,
                          'inner_t':inner_t})
        blackbox.save(verbose=verbose)    
    return M

def projected(M, int d):
    '''
    Project onto rank d psd matrices
    '''
    cdef int n = M.shape[0]
    cdef np.ndarray D,V
    D, V = np.linalg.eigh(M)
    cdef np.ndarray perm = D.argsort()
    cdef double bound = max(D[perm][-d], 0)
    for i in range(n):
        if D[i] < bound:
            D[i] = 0
    M = np.dot(np.dot(V,np.diag(D)),V.transpose());
    return M
