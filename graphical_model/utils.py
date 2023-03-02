# %%
import os
from PIL import Image
import numpy as np
import pandas as pd
from level_sets.utils import get_level_sets, spatio_environ_dependence
from level_sets.metrics import compactness, elongation


# %%
# Defining the image's level-sets as a spatial point pattern
# For now the location of the level-set is the mean location of all pixels in the set
# spp = [id, x, y, set-value, set-size]

#TODO: Get a better name for this function
def graphical_model(img, return_spp=False):
    level_sets = get_level_sets(img)

    uni_level_sets = pd.unique(level_sets.flatten())
    spp = np.zeros((uni_level_sets.shape[0], 7))
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        level_set = np.array(level_sets == ls)*ls
        set_value = img[subset[0]]
        set_size = len(subset)
        #TODO Get a smarter way to do this
        spp[i, 0] = ls
        spp[i, 1:3] = np.mean(subset, axis=0)
        spp[i, 3] = set_value
        spp[i, 4] = set_size
        spp[i, 5] = compactness(level_set)
        spp[i, 6] = elongation(level_set)

    distance_matrix = np.zeros((spp.shape[0], spp.shape[0]))
    for i in range(spp.shape[0]):
        for j in np.arange(i+1, spp.shape[0], 1):
            m = spatio_environ_dependence(spp[i, :], spp[j, :], "l2", "l1", 0.5)
            distance_matrix[i, j] = m
            distance_matrix[j, i] = m

    nodes = spp[:, 1:3]
    edges = distance_matrix
    
    if return_spp:
        return nodes, edges, spp
    return nodes, edges
