# graphical_model_cy.pyx
#cython: language_level=3

import numpy as np
cimport numpy as np
import pandas as pd
import sys
import os
from shapely.geometry import MultiPoint, mapping

current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_script_directory)
from level_sets.metrics import compactness, elongation
from level_sets.utils import get_level_sets, get_fuzzy_sets
from distance import spatio_distance

def graphical_model_cy(
    np.ndarray[np.float64_t, ndim=2] img,
    return_spp=False,
    set_type="level",
    fuzzy_cutoff=20,
    normalise_pixel_index=False,
    connectivity=4,
    alpha=0.5,
    normalise_gray=True,
    size_proportion=False,
    ls_spatial_dist="euclidean",
    ls_attr_dist="cityblock",
    centroid_method="mean"
):

    cdef:
        bint should_normalise
        np.ndarray[np.int64_t, ndim=2] level_sets
        np.ndarray[np.int64_t] uni_level_sets
        int X, Y
        list results
        int ls
        list subset
        np.ndarray[np.float64_t, ndim=2] level_set
        np.float64_t set_value, set_size, intensity
        object points, centroid
        object spp
        np.ndarray[np.float64_t, ndim=2] spp_np, nodes, edges

    should_normalise = normalise_gray and img.max() > 1
    
    level_sets = np.array(get_level_sets(img, connectivity=connectivity / 4), dtype=np.int64) if set_type == "level" else \
        np.array(get_fuzzy_sets(img, fuzzy_cutoff, connectivity) + 1, dtype=np.int64)

    uni_level_sets = pd.unique(level_sets.flatten())
    results = []
    X = img.shape[0]
    Y = img.shape[1]

    for ls in uni_level_sets:
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        level_set = ((level_sets == ls) * ls).astype(np.float64)
        set_value = img[subset[0]]
        set_size = len(subset) / (img.shape[0] * img.shape[1]) if size_proportion else len(subset)
        points = MultiPoint(subset)
        centroid = points.centroid if centroid_method == "mean" else points.representative_point()
        centroid = mapping(centroid)['coordinates']
        intensity = set_value / 255 if should_normalise else set_value

        results.append({
            "level-set": ls,
            "x-coor": centroid[0] / X if normalise_pixel_index else centroid[0],
            "y-coor": centroid[1] / X if normalise_pixel_index else centroid[1],
            "intensity": intensity,
            "size": set_size,
            "compactness": compactness(level_set),
            "elongation": elongation(level_set),
            "pixel_indices": subset
        })

    spp = pd.DataFrame(results)
    spp_np = spp.drop(labels=["pixel_indices"], axis=1).values.astype(float)
    nodes = np.array(spp.iloc[:, 1:3], dtype=np.float64)
    edges = np.asarray(
        spatio_distance.calculate_distance(
            spp_np,
            ls_spatial_dist,
            ls_attr_dist,
            alpha
        )
    )

    if return_spp:
        return nodes, edges, spp
    return nodes, edges
