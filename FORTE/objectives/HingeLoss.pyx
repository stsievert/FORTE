import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double[:,:] H = np.array([[0,1.,-1.],[ 1.,  -1.,  0.],[ -1.,  0.,  1.]], dtype=np.dtype('d'))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double getLossM(np.ndarray[DTYPE_t, ndim=2] M,S):
    """
    Returns loss on M with respect to list of triplets S: 1/len(S) \sum_{q in S} loss(M,q).
    Intuitively, q=[i,j,k] "agrees" with X if ||x_i - x_j||^2 < ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = M_kk-M_jj-2(M_ik-M_ij)

    Then the hinge loss, loss(M,q) = max(0, 1-s(M,q))
    Usage:
        loss = getLossM(M,S)
    """
    cdef double loss=0.;
    cdef int t;
    cdef int i,j,k
    cdef int m = len(S);
    for t in range(m):
        i,j,k = S[t]
        loss = loss + max(0., 1.-(M[k,k] -2*M[i,k] + 2*M[i,j] - M[j,j]))/m
    return loss

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline _getPartialGradientM(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] G, q):
    """
    Internal function for computing the gradient of the hinge loss with respect to a query. Useful for SGD and full gradient computations.
    Given a curent Gram matrix M, and a matrix G (used to aggregate partial gradients), updates G with the gradient of getLossM(M,q).
    Note, this does not return a partial gradient! You probably want getPartialGradientM.
    
    Let the pattern matrix be H = [[0,1.,-1.],[ 1.,  -1.,  0.],[ -1.,  0.,  1.]].
    If q = i,j,k, let score = M[j,j] -2*M[i,j] + 2*M[i,k] - M[k,k]. 
    The gradient with respect to index i,j = -H[i,j] if 1-score>0, 0 otherwise


    Usage:
    _getPartialGradient(M,G,q)
    """
    cdef int i,j,k
    cdef int n = M.shape[0]
    i,j,k=q 
    cdef int r
    cdef double nscore = 1.-(M[k,k] -2*M[i,k] + 2*M[i,j] - M[j,j])
    if 1-nscore > 0:
        for r in range(3):
            for s in range(3):
                G[q[r], q[s]] += -H[r,s]

@cython.boundscheck(False)
@cython.wraparound(False)
def getPartialGradientM(np.ndarray[DTYPE_t, ndim=2] M, q):
    '''
    Returns the partialGradient with respect to a query. See the documentation for _getPartialGradientM 
    '''
    cdef int n = M.shape[0]
    # compute Gradient and return
    cdef np.ndarray G = np.zeros((n,n), dtype=DTYPE)
    return _getPartialGradientM(M, G, q)
    
@cython.boundscheck(False)
@cython.wraparound(False)
def getFullGradientM(np.ndarray[DTYPE_t, ndim=2] M, S):
    """
    Returns normalized gradient of logistic loss with respect to M and S a set of triplets.
    In otherwords returns, 1/|S|\sum_{q in S} getPartialGradientM(M,q)

    See _getPartialGradientM to see how gradient is computed.
    TODO: Create an update version of this function to prevent reallocation of G each iteration?
    Usage:
        G = getFullGradient(X,S)
    """
    cdef int i,j
    cdef int n = M.shape[0]
    # compute Gradient
    cdef np.ndarray G = np.zeros((n,n), dtype=DTYPE)
    for i in range(len(S)):
        _getPartialGradientM(M,G,S[i])
    G = G/len(S)
    return G

# @cython.boundscheck(False)
# @cython.wraparound(False)
########### cdef this
cpdef double getLossX(np.ndarray[DTYPE_t, ndim=2] X,S):
    """
    Returns loss on M with respect to list of triplets S: 1/len(S) \sum_{q in S} loss(X,q).
    Intuitively, q=[i,j,k] "agrees" with X if ||x_i - x_j||^2 < ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_i - x_k||^2 - ||x_i - x_j||^2
    Then the logistic loss, loss(X,q) = log(1+exp(-s(X,q)))
    Usage:
        log_loss = getLossX(X,S)

    """
    cdef double loss, score
    cdef int t;
    cdef int i,j,k
    cdef int m = len(S)
    for t in range(m):
        i,j,k = S[t]
        score = np.dot(X[k],X[k]) -2*np.dot(X[i],X[k]) + 2*np.dot(X[i],X[j]) - np.dot(X[j],X[j])
        loss += max(0, 1-score)
    return loss/m

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline _getPartialGradientX(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] G, q):
    """
    Internal function for computing the gradient of the hinge loss with respect to a query. Useful for SGD and full gradient computations.
    Given a curent embedding X, and a matrix G (used to aggregate partial gradients), updates G with the gradient of getLossX(X,q).
    Note, this does not return a partial gradient! You probably want getPartialGradientX.
    
    If q = i,j,k, let score = M[j,j] -2*M[i,j] + 2*M[i,k] - M[k,k]. The gradient updates are
    G[i] = -2*(X[j] - X[k])
    G[j] = -2*(X[i] - X[j])
    G[k] = -2*(X[k] - X[i])

    Usage:
    _getPartialGradientX(X,G,q)
    """
    cdef int i,j,k
    i,j,k=q 
    # cdef int r
    # cdef double t,tt 
    cdef double score = np.dot(X[k],X[k]) -2*np.dot(X[i],X[k]) + 2*np.dot(X[i],X[j]) - np.dot(X[j],X[j])
    if 1-score>0:
        G[i] = G[i] - 2*(X[j] - X[k])
        G[j] = G[j] - 2*(X[i] - X[j])
        G[k] = G[k] - 2*(X[k]-X[i]) 

def getPartialGradientX(np.ndarray[DTYPE_t, ndim=2] X, q):
    """
    Function for computing the gradient of the logistic loss with respect to a query. See documentation for _getPartialGradientX
    
    Usage:
    G = getPartialGradientX(X,q)
    """
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    # compute Gradient and return
    cdef np.ndarray G = np.zeros((n,d), dtype=DTYPE)
    return _getPartialGradientX(X, G, q)
            
@cython.boundscheck(False)
@cython.wraparound(False)
def getFullGradientX(np.ndarray[DTYPE_t, ndim=2] X, S):
    """
    Returns normalized gradient of logistic loss wrt to X and S.

    For q=[i,j,k], let s(X,q) = ||x_i - x_k||^2 - ||x_i - x_j||^2
    If loss is logistic_loss then loss(X,q) = log(1+exp(s(X,q)))

    Usage:
        G = getFullGradient(X,S)
    """
    cdef int i,j
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    # compute Gradient
    cdef np.ndarray G = np.zeros((n,d), dtype=DTYPE)
    cdef int m = len(S)
    for i in range(m):
        _getPartialGradientX(X,G,S[i])
    G = G/m
    return G


