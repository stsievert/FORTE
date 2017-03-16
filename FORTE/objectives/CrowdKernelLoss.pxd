import numpy as np 
cimport numpy as np 
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cimport cython

realmin = np.finfo(float).tiny
realmax = np.finfo(float).max

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _getTripletScore(np.ndarray[DTYPE_t, ndim=2] X, list q):
    """
    Given X,q=[i,j,k] returns score = ||x_i - x_k||^2 - ||x_i - x_j||^2
    If score > 0 then the triplet agrees with the embedding, otherwise it does not 

    Usage:
        score = getTripletScore(X,[3,4,5])
    """
    cdef int i = q[0]
    cdef int j = q[1]
    cdef int k = q[2]
    return np.dot(X[k],X[k]) -2*np.dot(X[i],X[k]) + 2*np.dot(X[i],X[j]) - np.dot(X[j],X[j])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline _getLoss(np.ndarray[DTYPE_t, ndim=2] X,list S):
    """
    Returns loss on X with respect to list of triplets S: 1/len(S) \sum_{q in S} loss(X,q).
    Intuitively, q=[i,j,k] "agrees" with X if ||x_j - x_k||^2 > ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_k - x_i||^2 - ||x_j - x_i||^2
    If loss is hinge_loss then loss(X,q) = max(0,1-s(X,q))
    If loss is emp_loss then loss(X,q) = 1 if s(X,q)<0, and 0 otherwise

    Usage:
        emp_loss, hinge_loss = getLoss(X,S)
    """
    cdef int m = len(S)
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef int i

    cdef double log_loss = 0. #log_loss in crowd kernel model
    cdef double loss_ijk

    for i in range(m):
        q = S[i]
        loss_ijk = _getTripletScore(X,q)
        log_loss = log_loss - c_log(_getCrowdKernelTripletProbability(X[q[0]],X[q[1]],X[q[2]],.01))


    log_loss = log_loss/m
    return log_loss

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _getCrowdKernelTripletProbability(np.ndarray[DTYPE_t, ndim=1]xi,
                                                    np.ndarray[DTYPE_t, ndim=1]xj,
                                                    np.ndarray[DTYPE_t, ndim=1]xk,
                                                    double mu):
    """
    Return the probability of triplet [i,j,k] where i is closer to j than k. If >0.5: agrees with this labelling

    Namely:
    pabc = (mu + || c - a||^2)/(2*mu + || b - a ||^2+|| c - a ||^2)
    
    Inputs:
        (numpy.ndarray) a : numpy array
        (numpy.ndarray) b : numpy array
        (numpy.ndarray) c : numpy array
        (float) mu : regularization parameter
    """
    cdef double Dik = np.linalg.norm(xk-xi)
    cdef double Dij = np.linalg.norm(xj-xi)
    return (mu+Dik*Dik)/(2.*mu+Dij*Dij+Dik*Dik)

@cython.boundscheck(False)
@cython.wraparound(False)     
cdef inline getGradient(np.ndarray[DTYPE_t, ndim=2]X, list S, double mu):
    """
    Returns gradient of the log loss of the crowd kernel probability distribution.
    Requires a parameter mu for regularization.
    
    Usage:
        G,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq = getGradient(X,S)
    """
    cdef int n = X.shape[0]
    cdef int d  = X.shape[1]
    cdef int m = len(S)
    cdef int i,j,k
    cdef double num, den
    cdef np.ndarray[DTYPE_t, ndim=1] muX
    cdef np.ndarray[DTYPE_t, ndim=2] M = np.dot(X,X.T)		# compute the Gram matrix

    cdef np.ndarray[DTYPE_t, ndim=2] G
    G = np.zeros((n,d))

    cdef double log_loss = 0.
    for q in S:
        i,j,k = q
        num = np.maximum(mu + M[k,k] - 2*M[k,i] + M[i,i], realmin)
        den = np.maximum(2.*mu + M[k,k] - 2.*M[i,k] + 2.*M[i,i] - 2.*M[i,j] + M[j,j], realmin)
        G[i] = G[i] + 1*(2./num * (X[i]-X[k])-2./den*(2.*X[i]-X[j]-X[k]))
        G[j] = G[j] + 1*(2./den * (X[i]-X[j]))
        G[k] = G[k] + 1*((2./num-2./den)*(X[k]-X[i]))
        log_loss = log_loss + c_log(den) - c_log(num)

    log_loss = log_loss/m
    # Remember, the loss function is the sum of log(1/p^k_ij), this leads to an extra minus sign
    G = -1./m * G
    # compute statistics about gradient used for stopping conditions
    muX = np.mean(X,0)
    cdef double avg_row_norm_sq = 0.
    cdef double avg_grad_row_norm_sq = 0.
    cdef double max_grad_row_norm_sq 
    max_grad_row_norm_sq = 0.
    cdef double norm_grad_sq_0 = 0.
    cdef double row_norm_sq, grad_row_norm_sq

    for i in range(n):
        row_norm_sq = 0.
        for j in range(d):
            row_norm_sq = row_norm_sq + (X[i,j]-muX[j])*(X[i,j]-muX[j])
            grad_row_norm_sq = grad_row_norm_sq + G[i,j]*G[i,j]  
        avg_row_norm_sq = avg_row_norm_sq + row_norm_sq/n
        avg_grad_row_norm_sq = avg_grad_row_norm_sq + grad_row_norm_sq/n
        max_grad_row_norm_sq = np.maximum(max_grad_row_norm_sq,grad_row_norm_sq)

    return G,log_loss,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq


