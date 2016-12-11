import sys
sys.path.append('../..')

import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
import blackbox
from libc.math cimport exp as c_exp
from time import time

import FORTE.objectives.HingeLoss as HL
import FORTE.utils as utils

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

norm = np.linalg.norm

@blackbox.record
def computeEmbedding(int n, int d,S,num_random_restarts=0,
                    max_num_passes_SGD=16,max_iter_GD=50,
                    max_norm=1,epsilon=0.01,verbose=False):
    """
    Computes an embedding of n objects in d dimensions usin the triplets of S.
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.
    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n].
        (int) num_random_restarts : number of random restarts (nonconvex
        optimization, may converge to local minima). E.g., 9 random restarts
        means take the best of 10 runs of the optimization routine.
        (int) max_num_passes : maximum number of passes over data SGD makes before proceeding to GD (default equals 16)
        (int) max_iter_GD: maximum number of GD iteration (default equals 500)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, smaller means more accurate (default = 0.01)
        (boolean) verbose : outputs some progress (default equals False)
    Outputs:
        (numpy.ndarray) X : output embedding  
    """
    X = np.random.rand(n,d)
    X_old = None
    cdef double emp_loss_old = float('inf')
    for restart in range(num_random_restarts + 1):
        if verbose:
            print "Restart:{}/{}".format(restart, num_random_restarts)
        ts = time()
        X,_ = computeEmbeddingWithEpochSGD(n,d,S,
                                             max_num_passes=max_num_passes_SGD,
                                             max_norm=max_norm,
                                             epsilon=epsilon,
                                             verbose=verbose)
        te_sgd = time()-ts
        ts = time()
        X_new,emp_loss_new,hinge_loss_new, _ = computeEmbeddingWithFG(X,S,
                                                                    max_iters=max_iter_GD,
                                                                    max_norm=max_norm,
                                                                    epsilon=epsilon,
                                                                    verbose=verbose)
        te_gd = time()-ts        
        if emp_loss_new < emp_loss_old:
            X_old = X_new
            emp_loss_old = emp_loss_new
        if verbose:
            print ("restart %d:   emp_loss = %f,   "
                   "hinge_loss = %f,   duration=%f+%f") %(restart,
                                                        emp_loss_new,hinge_loss_new,
                                                        te_sgd,te_gd)
    return X_old

def computeEmbeddingWithEpochSGD(int n,int d,S,max_num_passes=0,max_norm=0,epsilon=0.001,a=1,verbose=False):
    """
    Performs epochSGD where step size is constant across each epoch, epochs are 
    doubling in size, and step sizes are getting cut in half after each epoch.
    This has the effect of having a step size decreasing like 1/T. a0 defines 
    the initial step size on the first epoch. 
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.
    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) max_num_passes : maximum number of passes over data (default equals 16)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition (default = 0.01)
        (float) a0 : inititial step size (default equals 0.1)
        (boolean) verbose : output iteration progress or not (default equals False)
    Outputs:
        (numpy.ndarray) X : output embedding
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and 
                        b is the avg row norm of the centered embedding matrix X. This is a 
                        means to determine how close the current solution is to the "best" solution.  
    Usage:
        X,gamma = computeEmbeddingWithEpochSGD(n,d,S)
    """
    cdef int m = len(S)
    cdef int i,j,k 
    cdef double score, outer_loss
    # norm of each object is equal to 1 in expectation
    cdef np.ndarray[DTYPE_t, ndim=2] X = np.random.randn(n,d)
    if max_num_passes==0:
        max_iters = 16*m
    else:
        max_iters = max_num_passes*m
    if max_norm == 0:
        max_norm = 10*d
    cdef int epoch_length = m
    cdef int t = 0
    cdef int t_e = 0
    cdef double rel_max_grad = float('inf')

    while t < max_iters:
        t += 1
        t_e += 1
        # check epoch conditions, udpate step size
        if t_e % epoch_length == 0:
            a = a*0.5
            epoch_length = 2*epoch_length
            t_e = 0
            if epsilon>0 or verbose:
                # get losses
                emp_loss = utils.empirical_lossX(X,S)
                hinge_loss = HL.getLossX(X,S)
                # get gradient and check stopping-time statistics
                G = HL.getFullGradientX(X,S)
                rel_max_grad, norm_grad_sq_0 = relative_grad(G,X)
                blackbox.logdict({'iter':t,
                                  'epoch':t_e,
                                  'emp_loss':emp_loss,
                                  'hinge_loss':hinge_loss,
                                  'G_norm':norm_grad_sq_0,
                                  'rel_max_grad':rel_max_grad,
                                  'alpha':a})
                blackbox.save(verbose=verbose) 
                if rel_max_grad < epsilon:
                    break
        # get random triplet unifomrly at random
        q = S[np.random.randint(m)]
        i,j,k = q
        score = np.dot(X[k],X[k]) -2*np.dot(X[i],X[k]) + 2*np.dot(X[i],X[j]) - np.dot(X[j],X[j])
        outer_loss = 1./(1.+c_exp(score))
        X[i] = X[i] + 2.*a*outer_loss*(X[j] - X[k])          # gradient update for X[i]
        X[j] = X[j] + 2.*a*outer_loss*(X[i] - X[j])          # gradient update for X[j]
        X[k] = X[k] + 2.*a*outer_loss*(X[k] - X[i])          # gradient update for X[k]

        # X[q,:] = X[q,:] - a*grad_partial[q,:]
        # project back onto ball such that norm(X[i])<=max_norm
        for i in q:
            norm_i = np.linalg.norm(X[i])
            if norm_i>max_norm:
                X[i] = X[i] * (max_norm / norm_i)
    return X,rel_max_grad

def computeEmbeddingWithFG(np.ndarray[DTYPE_t, ndim=2] X,S, max_iters=50,max_norm=1.,epsilon=0.01,c1=.00001,rho=0.5,verbose=False):
    """
    Performs Burer-Monteiro factored gradient descent to learn embedding with geometric amarijo 
    line search (with parameter c1). S is a list of triplets such that for each q in S, 
    q = [i,j,k] means that object k should be closer to i than j.
    Implements line search algorithm 3.1 of page 37 in Nocedal and Wright (2006) Numerical Optimization
    Inputs:
        (numpy.ndarray) X : input embedding
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) max_iters : maximum number of iterations of SGD (default equals 40*len(S))
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, exits if gamma<epsilon (default = 0.01)
        (float) c1 : Amarijo stopping condition parameter (default equals 0.0001)
        (float) rho : Backtracking line search parameter (default equals 0.5)
        (boolean) verbose : output iteration progress or not (default equals False)
    Outputs:
        (numpy.ndarray) X : output embedding
        (float) emp_loss : output 0/1 error
        (float) log_loss : output log loss
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and 
                        b is the avg row norm of the centered embedding matrix X. 
                        This is a means to determine how close the current solution is to the "best" solution.  
    Usage:
        X,gamma = computeEmbeddingWithFG(X,S)
    """
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef double alpha = 10.
    cdef double emp_loss, hinge_loss
    cdef int t = 0
    cdef int inner_t = 0
    cdef double hinge_loss_0 = HL.getLossX(X,S)
    cdef double rel_max_grad = float('inf')

    while t < max_iters:
        t = t + 1
        # get gradient and stopping-time statistics
        G = HL.getFullGradientX(X,S)
        rel_max_grad, norm_grad_sq_0 = relative_grad(G,X)
        if rel_max_grad < epsilon:
            break
        # perform backtracking line search
        hinge_loss_k = HL.getLossX(X-alpha*G,S)
        alpha = 1.1*alpha
        inner_t = 0
        while hinge_loss_k > hinge_loss_0 - c1*alpha*norm_grad_sq_0:
            alpha = alpha*rho
            hinge_loss_k = HL.getLossX(X-alpha*G,S)
            inner_t += 1
            if inner_t > 10:
                break
        X = X-alpha*G 
        hinge_loss_k = HL.getLossX(X,S)
        if inner_t == 0:
            alpha = 2*alpha
        hinge_loss_0 = hinge_loss_k     # save previous log_loss for next line search
        emp_loss = utils.empirical_lossX(X,S)           # not strictly necessary
        
        blackbox.logdict({'iter':t,
                  'emp_loss':emp_loss,
                  'hinge_loss':hinge_loss_0,
                  'G_norm':norm_grad_sq_0,
                  'rel_max_grad':rel_max_grad,
                  'alpha':alpha,
                  'inner_t':inner_t})
        blackbox.save(verbose=verbose)                    
    return X,emp_loss,hinge_loss_0,rel_max_grad


cdef inline relative_grad(np.ndarray[DTYPE_t, ndim=2]G, np.ndarray[DTYPE_t, ndim=2] X):
    ## Define variables
    cdef int n = X.shape[0]             # number of points
    cdef int d = X.shape[1]             # number of dimensions
    cdef np.ndarray mu = np.mean(X, 0)      
    cdef double max_grad_row_norm_sq = 0.
    cdef double grad_norm_sq = 0.          # incrementally compute frobenius norm squared
    cdef double row_norm_sq
    cdef double grad_row_norm_sq
    cdef double avg_row_norm_sq = 0.        # average norm of a row of X matrix

    for i in range(n):
        row_norm_sq = 0.
        grad_row_norm_sq = 0.
        for j in range(d):
            row_norm_sq = row_norm_sq + (X[i,j]-mu[j])*(X[i,j]-mu[j])
            grad_row_norm_sq = grad_row_norm_sq + G[i,j]*G[i,j]
        # update stats about G
        max_grad_row_norm_sq = max(max_grad_row_norm_sq,grad_row_norm_sq)
        grad_norm_sq = grad_norm_sq + grad_row_norm_sq      # update frobenius computation
        # update stat about X
        avg_row_norm_sq = avg_row_norm_sq + row_norm_sq/n
    return np.sqrt(max_grad_row_norm_sq / avg_row_norm_sq), grad_norm_sq




