import numpy as np 
cimport numpy as np 
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _l1Loss(np.ndarray[DTYPE_t, ndim=2] X,list S):
    """
    Returns loss on X with respect to list of triplets S: 1/len(S) \sum_{q in S} loss(X,q).
    Intuitively, q=[i,j,k] "agrees" with X if ||x_i - x_k||_1 > ||x_i - x_j||_1.

    For q=[i,j,k], let s(X,q) = ||x_k - x_i||_1 - ||x_j - x_i||_1

    Usage:
        l1Loss = l1Loss(X,S)
    """
    cdef int m = len(S)
    cdef int i,j,k
    cdef double loss

    loss = 0.
    for t in range(m):
        i, j, k = S[t][0], S[t][1], S[t][2]
        loss = loss + c_log(1. + c_exp(np.linalg.norm(X[k] - X[i], ord=1) - np.linalg.norm(X[j] - X[i], ord=1)))
    return loss/m

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] _partialGradient(np.ndarray[DTYPE_t, ndim=2] X, list q):
    """
    Computes the partial gradient wrt triplet query q ordered [head, win, lose]

    inputs:
    ndarray X: the current embedding
    list q: [i,j,k] triplet in the format [head, win, lose]

    returns
    ndarray G: gradient matrix for triplet q
    """
    cdef int i = q[0]
    cdef int j = q[1]
    cdef int k = q[2]
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] G
    G = np.zeros((n,d))
    G[j] = 2*(X[j] > X[i]) - 1
    G[k] = -(2*(X[k] > X[i]) - 1)
    G[i] = -(G[j] + G[k])
    G = 1./(1. + c_exp(np.linalg.norm(X[k] - X[i], ord=1) - np.linalg.norm(X[j] - X[i], ord=1)))*G
    return G

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] _fullGradient(np.ndarray[DTYPE_t, ndim=2] X, list S):
    """
    Compute the full gradient matrix for a set of triplets S

    inputs:
    ndarray X: the current embedding
    list S: list of all triplets where each is in [head, win, lose] format

    return:
    ndarray G: the gradient matrix G for all triplets
    """
    cdef np.ndarray[DTYPE_t, ndim=2] G
    cdef int m = len(S)
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    G = np.zeros((n,d))
    for t in range(m):
        G = G + _partialGradient(X, S[t])
    return G/m






