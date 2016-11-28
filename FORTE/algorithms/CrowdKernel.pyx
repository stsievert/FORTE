import time, math, sys
sys.path.append('../..')

import numpy as np
cimport numpy as np
import blackbox
import FORTE.objectives.CrowdKernelLoss as ck 
cimport FORTE.objectives.CrowdKernelLoss as ck 

import FORTE.utils as utils

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

norm = np.linalg.norm
realmin = np.finfo(float).tiny
realmax = np.finfo(float).max

@blackbox.record
def computeEmbedding(int n, int d, S, mu=.01, num_random_restarts=0,max_num_passes_SGD=0,max_iter_GD=50,max_norm=0., epsilon=0.01, verbose=False):
    """
    Computes an embedding of n objects in d dimensions usin the triplets of S.
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n].
        (float) mu : regularization value 
        (int) num_random_restarts : number of random restarts (nonconvex optimization, may converge to local minima)
        (int) max_num_passes_SGD : maximum number of passes over data SGD makes before proceeding to GD (default equals 16)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, smaller means more accurate (default = 0.01)
        (boolean) verbose : outputs some progress (default equals False)

    Outputs:
        (numpy.ndarray) X : output embedding
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and b is the avg row norm of the centered embedding matrix X. This is a means to determine how close the current solution is to the "best" solution.  
    """

    if max_num_passes_SGD==0:
        max_num_passes_SGD = 32
    
    cdef np.ndarray[DTYPE_t, ndim=2] X_old, X, X_new
    X_old = np.random.rand(n,d)
    cdef double emp_loss_old = realmax
    cdef double ts, te_sgd, acc, emp_loss_new, log_loss_new, hinge_loss_new, acc_new, te_gd
    cdef int num_restarts = -1
    
    while num_restarts < num_random_restarts:
        num_restarts += 1
        
        print "Epoch SGD"
        ts = time.time()
        X,acc = computeEmbeddingWithEpochSGD(n,d,S,mu,max_num_passes_SGD=max_num_passes_SGD,epsilon=0.,verbose=verbose)
        te_sgd = time.time()-ts
        
        print "Gradient Descent"
        ts = time.time()
        X_new, emp_loss_new, log_loss_new, hinge_loss_new, acc_new = computeEmbeddingWithGD(X, S, mu, 
                                                                                            max_iters=max_iter_GD, 
                                                                                            max_norm=max_norm, 
                                                                                            epsilon=epsilon, 
                                                                                            verbose=verbose)
        emp_loss_new,hinge_loss_new,log_loss_new = ck._getLoss(X_new,S)
        te_gd = time.time()-ts

        if emp_loss_new<emp_loss_old:
            X_old = X_new
            emp_loss_old = emp_loss_new

        if verbose:
            print "restart %d:   emp_loss = %f,   hinge_loss = %f,   duration=%f+%f" %(num_restarts,emp_loss_new,hinge_loss_new,te_sgd,0)


    return X_old, emp_loss_old

def computeEmbeddingWithEpochSGD(int n, int d,S,double mu, max_num_passes_SGD=0,max_norm=0.,epsilon=0.01,a0=0.1,verbose=False):
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
        (float) mu : regularization value 
        (int) max_num_passes_SGD : maximum number of passes over data (default equals 16)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition (default = 0.01)
        (float) a0 : inititial step size (default equals 0.1)
        (boolean) verbose : output iteration progress or not (default equals False)

    Outputs:
        (numpy.ndarray) X : output embedding
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and b is the avg row norm of the centered embedding matrix X. This is a means to determine how close the current solution is to the "best" solution.  


    Usage:
        X,gamma = computeEmbeddingWithEpochSGD(n,d,S)
    """
    cdef int m = len(S)
    cdef int epoch_length, t, t_e, max_iters, i,j,k 
    cdef double a, emp_loss,hinge_loss,log_loss, rel_max_grad, rel_avg_grad, num, den, norm_i
    cdef np.ndarray[DTYPE_t, ndim=1] grad_i, grad_j, grad_k

    # norm of each object is equal to 1 in expectation
    cdef np.ndarray[DTYPE_t, ndim=2] X = np.random.randn(n,d)*.0001
    cdef np.ndarray[DTYPE_t, ndim=2] G

    if max_num_passes_SGD==0:
        max_iters = 16*m
    else:
        max_iters = max_num_passes_SGD*m

    if max_norm==0:
        max_norm = 10.*d


    epoch_length = m
    a = a0
    t = 0
    t_e = 0
    
    while t < max_iters:
        t = t + 1
        t_e = t_e + 1
        
        # check epoch conditions, udpate step size
        if t_e % epoch_length == 0:
            a = a*0.5
            epoch_length = 2*epoch_length

            if epsilon>0 or verbose:
                # get losses
                emp_loss,hinge_loss,log_loss = ck._getLoss(X,S)

                # get gradient and check stopping-time statistics
                G,log_loss,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq = ck.getGradient(X,S,mu)
                rel_max_grad = np.sqrt( max_grad_row_norm_sq / avg_row_norm_sq )
                rel_avg_grad = np.sqrt( avg_grad_row_norm_sq / avg_row_norm_sq )

                blackbox.logdict({'iter':t,
                                  'epoch':t_e,
                                  'emp_loss':emp_loss,
                                  'hinge_loss':hinge_loss,
                                  'log_loss':log_loss,
                                  'alpha':a})
                blackbox.save(verbose=verbose)
                if rel_max_grad < epsilon:
                    break
            t_e = 0

        # get random triplet uniformly at random
        q = S[np.random.randint(m)]

        # take gradient step
        i,j,k = q
        num = np.maximum(mu + norm(X[k]-X[i])*norm(X[k]-X[i]) , realmin)
        den = np.maximum(2.*mu + norm(X[k]-X[i])*norm(X[k]-X[i]) + norm(X[i]-X[j])*norm(X[i]-X[j]), realmin)
        grad_i = -1*(2./num * (X[i]-X[k])-2./den*(2.*X[i]-X[j]-X[k]))
        grad_j = -1*(2./den * (X[i]-X[j]))
        grad_k = -1*((2./num-2./den)*(X[k]-X[i]))
        
        X[i] = X[i] - a*grad_i 
        X[j] = X[j] - a*grad_j 
        X[k] = X[k] - a*grad_k 
        
        # project back onto ball such that norm(X[i])<=max_norm
        for i in q:
           norm_i = norm(X[i])
           if norm_i>max_norm:
               X[i] = X[i] * (max_norm / norm_i)

    return X,rel_max_grad

def computeEmbeddingWithGD(np.ndarray[DTYPE_t, ndim=2] X, S, double mu, max_iters=0, max_norm=0., epsilon=0.01, c1=0.0001, rho=.7, verbose=False):
    """
    Performs gradient descent with step size as implemented in stochastic triplet embedding code, namely ckl_x.m 
    See: http://homepage.tudelft.nl/19j49/ste/Stochastic_Triplet_Embedding_files/STE_Release.zip


    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Inputs:
        (numpy.ndarray) X : input embedding
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (float) mu : regularization parameter
        (int) max_iters : maximum number of iterations of SGD (default equals 40*len(S))
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, exits if gamma<epsilon (default = 0.01)
        (float) c1 : Amarijo stopping condition parameter (default equals 0.0001)
        (float) rho : Backtracking line search parameter (default equals 0.5)
        (boolean) verbose : output iteration progress or not (default equals False)

    Outputs:
        (numpy.ndarray) X : output embedding
        (float) emp_loss : output 0/1 error
        (float) hinge_loss : output hinge loss
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and b is the avg row norm of the centered embedding matrix X. This is a means to determine how close the current solution is to the "best" solution.  


    Usage:
        X,gamma = computeEmbeddingWithGD(X,S)
    """
    cdef int m = len(S)
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef int t, inner_t
    cdef double alpha,ts, rel_max_grad, rel_avg_grad, norm_i
    cdef double avg_grad_row_norm_sq, max_grad_row_norm_sq, avg_row_norm_sq
    cdef double log_loss

    cdef np.ndarray[DTYPE_t, ndim=2] G

    if max_iters==0:
        max_iters = 100

    if max_norm==0:
        max_norm = 10.*d

    alpha = .5*n
    t = 0
    cdef double norm_grad_sq_0 = realmax
    cdef double emp_loss_0 = realmax
    cdef double emp_loss_k = realmax
    cdef double hinge_loss_0 = realmax
    cdef double hinge_loss_k = realmax
    cdef double log_loss_0 = realmax
    cdef double log_loss_k = realmax

    G,log_loss,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq  = ck.getGradient(X, S, mu)
    rel_max_grad = np.sqrt( max_grad_row_norm_sq / avg_row_norm_sq )
    rel_avg_grad = np.sqrt( avg_grad_row_norm_sq / avg_row_norm_sq )
    
    while t < max_iters:
        t=t+1
        # get gradient and stopping-time statistics
        ts = time.time()
        if rel_max_grad < epsilon:
            blackbox.logdict({'iter':t,
                  'emp_loss':emp_loss_k,
                  'hinge_loss':hinge_loss_k,
                  'log_loss':log_loss_k,
                  'G_norm':norm_grad_sq_0,
                  'rel_avg_grad':rel_avg_grad,
                  'rel_max_grad':rel_max_grad,
                  'alpha':alpha})
            blackbox.save(verbose=verbose)
            if verbose:
                print "Exiting because of rel_max_grad=%s"%(rel_max_grad) 
            break
        
        # perform backtracking line search
        alpha = 2.*alpha
        ts = time.time()
        emp_loss_0, hinge_loss_0, log_loss_0 = ck._getLoss(X,S)
        norm_grad_sq_0 = avg_grad_row_norm_sq*n
        emp_loss_k, hinge_loss_k, log_loss_k = ck._getLoss(X-alpha*G, S)
        
        inner_t = 0
        while log_loss_k > log_loss_0 - c1*alpha*norm_grad_sq_0:
            alpha = alpha*rho
            emp_loss_k,hinge_loss_k,log_loss_k = ck._getLoss(X-alpha*G,S)
            inner_t += 1
        X  = X - alpha*G

        # project back onto ball such that norm(X[i])<=max_norm
        for i in range(n):
           norm_i = norm(X[i])
           if norm_i>max_norm:
               X[i] = X[i] * (max_norm / norm_i)
        
        # compute next gradient and stopping time statistics
        G,log_loss,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq  = ck.getGradient(X, S, mu)
        rel_max_grad = np.sqrt( max_grad_row_norm_sq / avg_row_norm_sq )
        rel_avg_grad = np.sqrt( avg_grad_row_norm_sq / avg_row_norm_sq )
        # save to blackbox
        blackbox.logdict({'iter':t,
                  'emp_loss':emp_loss_k,
                  'hinge_loss':hinge_loss_k,
                  'log_loss':log_loss_k,
                  'G_norm':norm_grad_sq_0,
                  'rel_avg_grad':rel_avg_grad,
                  'rel_max_grad':rel_max_grad,
                  'inner_t': inner_t,
                  'alpha':alpha})
        blackbox.save(verbose=verbose)
    return X,emp_loss_k,hinge_loss_k,log_loss_k,rel_max_grad

