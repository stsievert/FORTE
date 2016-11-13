import time, math, sys
sys.path.append('../..')

import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
#import blackbox

import FORTE.objectives.LogisticLoss as LL
import FORTE.utils as utils

norm = np.linalg.norm
floor = math.floor
ceil = math.ceil

#@blackbox.record
def computeEmbedding(int n, int d, S,
                     max_iter_GD=1000,
                     trace_norm=1,
                     epsilon=0.0001):
    """
    Computes STE MLE of a Gram matrix M using the triplets of S.
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n].
        (int) max_iter_GD: maximum number of GD iteration (default equals 500)
        (float) trace_norm : the maximum allowed trace norm of the gram matrix
        (float) epsilon : parameter that controls stopping condition, smaller means more accurate (default = 0.01)
        (boolean) verbose : outputs some progress (default equals False)
    Outputs:
        (numpy.ndarray) M : output Gram matrix
    """
    # Initialization of centered Gram Matrix
    cdef np.ndarray M = np.random.randn(n,n)
    M = (M+M.transpose())/2
    cdef np.ndarray J = np.eye(n)-np.ones((n,n))*1./n
    M = np.dot(np.dot(J,M),J)
    ts = time.time()
    
    M_new = computeEmbeddingWithGD(M,S,d,
                                   max_iters=max_iter_GD,
                                   trace_norm=trace_norm,
                                   epsilon=epsilon)
    te_gd = time.time()-ts        
    #blackbox.log()
    # print ("STEConvexNucNormProjected   emp_loss = %f,   "
    #        "log_loss = %f,   duration=%f") % (emp_loss_new,
    #                                           log_loss_new,
    #                                           te_gd)
    return M_new

def computeEmbeddingWithGD(np.ndarray M,S,int d,
                           max_iters=1000,
                           trace_norm=None,
                           epsilon=0.0001,
                           c1=.00001,
                           rho=0.5):
    """
    Performs gradient descent with geometric Amarijo line search (with parameter c1)
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object i should be closer to j than k.

    Inputs:
        (numpy.ndarray) M : input Gram matrix
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) max_iters : maximum number of iterations of SGD (default equals 40*len(S))
        (float) trace_norm : the maximum allowed trace norm of gram matrix (default equals inf)
        (float) epsilon : parameter that controls stopping condition, 
                          exits if new iteration does not differ in fro norm by more than epsilon (default = 0.0001)
        (float) c1 : Amarijo stopping condition parameter (default equals 0.0001)
        (float) rho : Backtracking line search parameter (default equals 0.5)
        (boolean) verbose : output iteration progress or not (default equals False)

    Outputs:
        (numpy.ndarray) M : output Gram matrix
    Usage:
        M = computeEmbeddingWithGD(Mtrue,S,d)
    """
    cdef int n  = M.shape[0] 
    cdef double alpha = 100
    cdef int t = 0

    cdef np.ndarray G, M_k, d_k, normG, normM;
    
    cdef double rel_max_grad, log_loss, log_loss_k, norm_grad_sq; 
    cdef double Delta, G_norm, max_fn_dev;
    
    cdef np.ndarray progress = np.random.rand(10)
    cdef int progress_idx = 0
    while t < max_iters:
        t+=1
        # get gradient and stopping-time statistics
        G = LL.getFullGradientM(M,S)
        normG = norm(G, axis=0)
        normM = norm(M, axis=0)
        rel_max_grad = np.sqrt( max(normG)/ sum(normM)/n)
        if rel_max_grad < epsilon:
            break
        # perform backtracking line search
        log_loss = LL.getLossM(M,S)
        norm_grad_sq = sum(normG)                             

        M_k = projected(M-alpha*G, trace_norm)
        log_loss_k = LL.getLossM( M_k , S)
        d_k = M_k - M

        Delta = norm(d_k, ord='fro')
        G_norm = np.sqrt(norm_grad_sq)
        progress[progress_idx] = log_loss
        progress_idx = (progress_idx + 1) % 10
        max_fn_dev = max(abs(progress-log_loss))
        # if Delta<epsilon or G_norm<epsilon or max_fn_dev<epsilon:
        #     #if verbose: print "Exiting: D_k=%f,  ||G||_F=%f,  Dfn=%f,  epsilon=%f,  alpha=%f" %(Delta,G_norm,max_fn_dev,epsilon,alpha)
        #     break

        # This linesearch comes from Fukushima and Mine, "A generalized proximal point algorithm for certain non-convex minimization problems"
        inner_t = 0
        while log_loss_k > log_loss - c1*alpha*norm_grad_sq and inner_t < 10:
            alpha = alpha*rho
            M_k = projected(M-alpha*G, trace_norm)
            log_loss_k = LL.getLossM( M_k ,S)
            inner_t += 1
        alpha = 1.2*alpha
        M = M_k
        emp_loss = utils.empirical_lossM(M,S)
        # print ("STEConvexNucNormProjected iter=%d,   emp_loss=%f,   log_loss=%f,   "
        #        "||d_k||=%f,   ||G||=%f,   alpha=%f,   i_t=%d") % (t,
        #                                                           emp_loss,
        #                                                           log_loss,
        #                                                           Delta,
        #                                                           G_norm,
        #                                                           alpha,inner_t)
    return M

def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    cdef int n = v.shape[0]  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    cdef np.ndarray u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    cdef np.ndarray w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    cdef int n = v.shape[0]  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def projected(M, R):
    '''
    Project onto psd nuclear norm ball of radius R
    '''
    cdef int n = M.shape[0]
    if R!=None:
        D, V = np.linalg.eigh(M)
        D = euclidean_proj_simplex(D, s=R)
        M = np.dot(np.dot(V,np.diag(D)),V.transpose());
    return M

def max_nonzero_sum(M):
    return max(abs(np.dot(M,np.ones((M.shape[0],1)))))
