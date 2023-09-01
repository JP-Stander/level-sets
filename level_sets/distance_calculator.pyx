import numpy as np
cimport numpy as np

def calculate_min_distance_index(np.ndarray[np.float64_t, ndim=1] data, np.ndarray[np.float64_t, ndim=2] reference_nodes):
    cdef int min_idx = 0
    cdef double min_distance = float('inf')
    cdef double temp_distance
    cdef int i, j
    
    for i in range(reference_nodes.shape[0]):
        temp_distance = 0
        for j in range(reference_nodes.shape[1]):
            temp_distance += (reference_nodes[i, j] - data[j]) ** 2
        if temp_distance < min_distance:
            min_distance = temp_distance
            min_idx = i

    return min_idx
