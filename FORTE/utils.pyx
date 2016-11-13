from __future__ import division
import numpy as np
cimport numpy as np
import utils
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log

#from libc.math cimport sqrt as c_sqrt
ctypedef np.npy_intp SIZE_t  
DTYPE = np.float64
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
    i = np.random.randint(n)
    j = np.random.randint(n)
    while (j==i):
        j = np.random.randint(n)
    k = np.random.randint(n)
    while (k==i) | (k==j):
        k = np.random.randint(n)
    q = [i, j, k]
    return q


cdef inline triplet_scoreM(np.ndarray[DTYPE_t, ndim=2] M,q):
    """
    Given M,q=[i,j,k] returns score = M_kk - M_jj - 2(M_ik-M_ij)
    If score > 0 then the triplet agrees with the embedding, otherwise it does not 
    Usage:
        score = getTripletScore(M,[3,4,5])
    """
    cdef int i,j,k
    i,j,k = q
    return M[k,k] -2*M[i,k] + 2*M[i,j] - M[j,j]

cdef inline triplet_scoreX(np.ndarray[DTYPE_t, ndim=2] X,q):
    """
    Given X,q=[i,j,k] returns score = ||x_i - x_k||^2 - ||x_i - x_j||^2
    If score < 0 then the triplet agrees with the embedding, otherwise it does not 
    Usage:
        score = getTripletScore(X,[3,4,5])
    """
    cdef int i,j,k
    i,j,k = q
    return np.dot(X[k],X[k]) -2*np.dot(X[i],X[k]) + 2*np.dot(X[i],X[j]) - np.dot(X[j],X[j])


def triplets(np.ndarray[DTYPE_t, ndim=2] X, int pulls, noise_func=None):
    """
    Generate a random set of #pulls triplets
    """
    S = []
    n = X.shape[0]; d = X.shape[1]
    for i in range(0,pulls):
        # get random triplet
        q = random_query(n)
        score = triplet_scoreX(X,q)
        # align it so it agrees with Xtrue: "q[0] is more similar to q[1] than q[2]"
        if score < 0:
            q = [q[i] for i in [0,2,1]]
        # add some noise
        if not noise_func is None:
            if np.random.rand() > noise_func(X,q):
                q = [ q[i] for i in [0,2,1]]
        S.append(q)   
    return S


cpdef inline empirical_lossM(np.ndarray[DTYPE_t, ndim=2] M, S):
    """
    Returns the empirical (0/1) loss of X on a set of triplets S. In other words, the proportion of triplets in S that are wrong. 
    Intuitively, q=[i,j,k] "agrees" with X if ||x_i - x_j||^2 < ||x_i - x_k||^2.

    Usage:
        emp_loss = empirical_loss(X, S)
    """
    cdef double loss = 0;
    cdef int t;
    cdef int m = len(S);
    for t in range(m):
        if triplet_scoreM(M,S[t]) < 0:
            loss += 1 
    return loss/m


cpdef inline empirical_lossX(np.ndarray[DTYPE_t, ndim=2] X, S):
    """
    Returns the empirical (0/1) loss of X on a set of triplets S. In other words, the proportion of triplets in S that are wrong. 
    Intuitively, q=[i,j,k] "agrees" with X if ||x_i - x_j||^2 < ||x_i - x_k||^2.

    Usage:
        emp_loss = empirical_loss(X, S)
    """
    cdef double loss = 0;
    cdef int t;
    cdef int m = len(S)
    for t in range(m):
        if triplet_scoreX(X,S[t]) < 0:
            loss += 1 
    return loss/m

def transform_MtoX(np.ndarray[DTYPE_t, ndim=2] M, int d):
    '''
    Get a set of points X in R^d back from a Gram Matrix
    '''
    n = M.shape[0]
    U,s,V = np.linalg.svd(M)
    
    for i in range(d, n):
        s[i] = 0
    s = np.diag(s)
    Mp = np.dot(np.dot(U.real,s),V.real.transpose())
    X = np.dot(U.real,np.sqrt(s).real)
    return Mp,X[:,0:d]

def transform_XtoM(np.ndarray[DTYPE_t, ndim=2] X):
    '''
    Get a set of points X in R^d back from a Gram Matrix
    '''
    return np.dot(X,X.transpose())


def transform_DtoM(np.ndarray[DTYPE_t, ndim=2] D):
    '''
    Transform a distance matrix to a Gram matrix.
    '''
    cdef double n = D.shape[0]
    V = np.eye(n) - np.ones((n,n))/n
    M = -1/2*np.dot(V,np.dot(D,V))
    return M
        
def transform_MtoD(np.ndarray[DTYPE_t, ndim=2] M):
    '''
    Transform a Gram matrix to a distance matrix.
    '''
    n = M.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = M[i,i]+M[j,j]-2*M[i,j]
    return D

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    A port of MATLAB's `procrustes` function to Numpy.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling 
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """
    n = X.shape[0]; m = X.shape[1]
    ny = Y.shape[0]; my = Y.shape[1]
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform
