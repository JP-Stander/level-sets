# cython: language_level=3

import numpy as np
cimport numpy as np
from scipy.spatial.distance import cdist
from ot import sinkhorn

# Function to perform the entire iteration loop
cpdef sinkhorn_sampler(np.ndarray[np.float64_t, ndim=2] X, int n, int num_iterations, np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b, double lambda_param):
    cdef int m = X.shape[0]
    cdef int d = X.shape[1]
    cdef np.ndarray[double, ndim=2] Y = X[np.random.choice(m, n, replace=False), :].copy()  # Select rows from X
    cdef np.ndarray[double, ndim=2] Y_new = np.zeros((n, d))
    cdef int m1 
    cdef int d1 
    for iteration in range(num_iterations):
        # Compute distance matrix
        C = cdist(X, Y)
        
        # Compute transport plan
        P = sinkhorn(a, b, C, reg=lambda_param)

        # Update Y based on transport plan
        for j in range(n):
            weighted_sum = np.zeros(d)
            weighted_total = 0.0
            for i in range(m):
                weighted_sum += P[i, j] * X[i, :]
                weighted_total += P[i, j]
            
            Y_new[j, :] = weighted_sum / weighted_total
        
        if np.sum(np.abs(Y - Y_new)) < 0.001:
            break
        
        Y = Y_new.copy()
    
    return Y
