from __future__ import division
import numpy as np
cimport numpy as np
import utils
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
DTYPE = np.float64
from cython.parallel import prange
ctypedef np.float64_t DTYPE_t
cimport cython


def random_query(n):
    """
    Outputs a triplet [i,j,k] chosen uniformly at random from all possible triplets 
    Outputs:
        [(int) i, (int) j, (int) k] q : where k in [n], i in [n]-k, j in [n]-k-j        
    Usage:
        q = getRandomQuery(n)
    """
    cdef int i,j,k
    i = randint(n)
    j = randint(n)
    while (j==i):
        j = randint(n)
    k = randint(n)
    while (k==i) | (k==j):
        k = randint(n)
    q = [i, j, k]
    return q


def triplet_scoreM(M,q):
    """
    Given M,q=[i,j,k] returns score = M_kk - M_jj - 2(M_ik-M_ij)
    If score > 0 then the triplet agrees with the embedding, otherwise it does not 
    Usage:
        score = getTripletScore(M,[3,4,5])
    """
    cdef int i,j,k
    i,j,k = q
    return M[k,k] -2*M[i,k] + 2*M[i,j] - M[j,j]

def triplet_scoreX(X,q):
    """
    Given X,q=[i,j,k] returns score = ||x_i - x_k||^2 - ||x_i - x_j||^2
    If score < 0 then the triplet agrees with the embedding, otherwise it does not 
    Usage:
        score = getTripletScore(X,[3,4,5])
    """
    cdef int i,j,k
    i,j,k = q
    return np.dot(X[k],X[k]) -2*dot(X[i],X[k]) + 2*dot(X[i],X[j]) - dot(X[j],X[j])


def triplets(X, pulls, noise_func=None):
    """
    Generate a random set of #pulls triplets
    """
    S = []
    n,d = X.shape
    for i in range(0,pulls):
        # get random triplet
        q = getRandomQuery(n)
        score = getTripletScoreX(X,q)
        # align it so it agrees with Xtrue: "q[0] is more similar to q[1] than q[2]"
        if score > 0:
            q = [q[i] for i in [0,2,1]]
        # add some noise
        if not noise_func is None:
            if rand() > noise_func(X,q)
                q = [ q[i] for i in [0,2,1]]
        S.append(q)   
    return S


def empirical_loss(X, S):
    """
    Returns the empirical (0/1) loss of X on a set of triplets S. In other words, the proportion of triplets in S that are wrong. 
    Intuitively, q=[i,j,k] "agrees" with X if ||x_i - x_j||^2 < ||x_i - x_k||^2.

    Usage:
        emp_loss = empirical_loss(X, S)
    """
    cdef double loss
    cdef int t;
    cdef int i,j,k
    cdef int m = len(S)
    for t in range(m):
        i,j,k = S[t]
        if triplet_score(X,q) < 0:
            loss += 1 
    return loss/m

def transform_MtoX(M,d):
    '''
    Get a set of points X in R^d back from a Gram Matrix
    '''
    n,n = M.shape
    U,s,V = svd(M)
    
    for i in range(d, n):
        s[i] = 0
    s = diag(s)
    Mp = dot(dot(U.real,s),V.real.transpose())
    X = dot(U.real,sqrt(s).real)
    return Mp,X[:,0:d]

def transform_XtoM(M):
    '''
    Get a set of points X in R^d back from a Gram Matrix
    '''
    return dot(X,X.transpose())


def transform_DtoM(D):
    '''
    Transform a distance matrix to a Gram matrix.
    '''
    n,n = D.shape
    V = eye(n) - 1./n*ones((n,n))
    M = -1/2*dot(V,dot(D,V))
    return M
        
def transform_MtoD(M):
    '''
    Transform a Gram matrix to a distance matrix.
    '''
    n,n = M.shape
    D = zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = M[i,i]+M[j,j]-2*M[i,j]
    return D

def procrustes(X,Y):
    pass



