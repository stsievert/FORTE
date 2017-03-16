import numpy as np 
cimport numpy as np 
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
cimport cython

def l1Loss(np.ndarray[DTYPE_t, ndim=2] X,list S):
    return _l1Loss(X,S)

def partialGradient(np.ndarray[DTYPE_t, ndim=2] X, list q):
    return _partialGradient(X, q)

def fullGradient(np.ndarray[DTYPE_t, ndim=2] X, list S):
    return _fullGradient(X, S)