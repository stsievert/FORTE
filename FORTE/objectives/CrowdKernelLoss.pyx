from __future__ import division
import numpy as np 
cimport numpy as np 
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
import time

DTYPE = np.float64
cimport cython

norm = np.linalg.norm
realmin = np.finfo(float).tiny
realmax = np.finfo(float).max

def getTripletScore(np.ndarray[DTYPE_t, ndim=2] X,q):
    return _getTripletScore(X,q)

def getLoss(np.ndarray[DTYPE_t, ndim=2] X,S):
    return _getLoss(X,S)

def getCrowdKernelTripletProbability(np.ndarray[DTYPE_t, ndim=1] b,
                                                    np.ndarray[DTYPE_t, ndim=1]c,
                                                    np.ndarray[DTYPE_t, ndim=1]a,
                                                    mu=0):
    return _getCrowdKernelTripletProbability(b,c,a,mu)

# def getEntropy(tau):
#     return _getEntropy(tau)

