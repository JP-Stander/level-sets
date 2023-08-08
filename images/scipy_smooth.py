import numpy as np
import pandas as pd
import statistics as stats
from skimage import measure

# def _smooth_level_set(img, n, nmax, connectivity, set_label, level_sets, N, M):
#     set_index = list(map(tuple, np.asarray(np.where(level_sets == set_label)).T.tolist()))
#     set_value = img[set_index[0]]
#     if n <= nmax:
#         neighbours = _find_neighbours(set_index, n, N, M, connectivity)
#         neighbour_values = [img[idx] for idx in neighbours]

#         median = stats.median(neighbour_values)
#         minimum = min(neighbour_values)
#         maximum = max(neighbour_values)
#         if minimum < median and median < maximum:
#             if not (minimum < set_value and set_value < maximum):
#                 return median
#             else:
#                 return _smooth_level_set(img, n+1, nmax, connectivity, set_label, level_sets, N, M)
#         else:
#             return median

#     else:
#         return set_value


def _smooth_level_set(img, n, nmax, connectivity, set_label, level_sets, N, M):
    set_index = list(map(tuple, np.asarray(np.where(level_sets == set_label)).T.tolist()))
    set_value = img[set_index[0]]
    for n in range(nmax):
        neighbours = _find_neighbours(set_index, n, N, M, connectivity)
        neighbour_values = [img[idx] for idx in neighbours]

        median = stats.median(neighbour_values)
        minimum = min(neighbour_values)
        maximum = max(neighbour_values)
        if minimum < median and median < maximum:
            if not (minimum < set_value and set_value < maximum):
                return median
        else:
            return median
    return set_value


def _find_neighbours(c, nmax, N, M, connectivity=4):  # noqa: C901
    """
    Function to get the neoghbour hood of a set of pixels in an image. This function
    makes use of 1-connectivity.

    Args:
        c: Set of pixels which neighbourhood needs to be determined
        nmax: Maximum number of neighbours of each element in each direction
        N: Height of the image
        M: Width of the image
    Returns:
        Smoothed image
    """
    w = []
    for i in range(nmax):
        for j in range(len(c)):
            if connectivity == 4:
                i1, i2 = c[j]
                if i2 - 1 >= 0:
                    w.append((i1, i2 - 1))
                if i2 + 1 < M:
                    w.append((i1, i2 + 1))
                if i1 - 1 >= 0:
                    w.append((i1 - 1, i2))
                if i1 + 1 < N:
                    w.append((i1 + 1, i2))
            elif connectivity == 8:
                i1, i2 = c[j]
                for y_chng in [-1, 0, 1]:
                    for x_chng in [-1, 0, 1]:
                        if i1 + y_chng >= 0 and i1 + y_chng < M:
                            if i2 + x_chng >= 0 and i2 + x_chng < N:
                                w.append((i1 + y_chng, i2 + x_chng))

        c = [a for a in pd.unique(c + w)]

    return c


# %%
img = np.random.randint(0, 255, (6, 6))
img[2, 2] = 80
img[2, 3] = 80
img[3, 2] = 80
level_sets = measure.label(img + 1, connectivity=1)
set_sizes = pd.value_counts(level_sets.flatten())
p = 3
req_set_sizes = set_sizes.iloc[(set_sizes == p).values].index
set_label = 15
N, M = img.shape

_smooth_level_set(img, 1, 3, 4, set_label, level_sets, N, M)
# %%


def levelset_median_smoother(
    image,  # Image to be smoothed
    pmax=3,  # Maximum level-set size to be smoothed
    nmax=2,  # Maximum order of neighbourhood
    pmin=1,  # Minimum order of neighbourhood
    connectivity=8  # Connectivity to be used for determining level-sets
):
    """
    Function to apply adaptive median smoother using level sets.

    Args:
        image: The greyscale images to be smooted
        pmax: Maximum size of level sets to be smoothed
        nmax: Maximum neighbourhood size
    Returns:
        Smoothed image
    """

    img = image.copy()

    # This function does not pick up a connected set with a value of 0
    level_sets = measure.label(img + 1, connectivity=1)
    set_sizes = pd.value_counts(level_sets.flatten())

    for p in np.arange(pmax, pmin - 1, -1):
        req_set_sizes = set_sizes.iloc[(set_sizes == p).values].index
        for set_label in req_set_sizes:
            N, M = img.shape
            smoothed_value = _smooth_level_set(img, 1, nmax, connectivity, set_label, level_sets, N, M)
            set_index = list(map(tuple, np.asarray(np.where(level_sets == set_label)).T.tolist()))
            set_value = img[set_index[0]]
            if smoothed_value != set_value:
                for idx in set_index:
                    img[idx] = smoothed_value

    return img
