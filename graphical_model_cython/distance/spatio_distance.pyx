# cython: boundscheck=False
# cython: wraparound=False
cimport numpy as np
import numpy as np
from libc.math cimport exp
from scipy.spatial.distance import cdist

cpdef double[:, :] calculate_distance(np.ndarray[double, ndim=2] spp2, str ls_spatial_dist, str ls_attr_dist, double alpha):
    cdef int i, j
    cdef int n = spp2.shape[0]
    cdef double m
    cdef double[:, :] distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            m = spatio_environ_dependence(
                spp2[i, :],
                spp2[j, :],
                ls_spatial_dist,
                ls_attr_dist,
                alpha
            )
            distance_matrix[i, j] = m
            distance_matrix[j, i] = m

    return distance_matrix

cdef double spatio_environ_dependence(double[:] point_a, double[:] point_b, str dist_type_a, str dist_type_b, double alpha):
    u_s = _dist(point_a[:2], point_b[:2], dist_type_a)
    u_e = _dist(point_a[2:], point_b[2:], dist_type_b)
    return exp(-(alpha * u_e + (1 - alpha) * u_s))

cdef double _dist(double[:] x, double[:] y, str dist_type):
    return cdist(np.array(x).reshape(1, -1), np.array(y).reshape(1, -1), metric=dist_type)[0][0]