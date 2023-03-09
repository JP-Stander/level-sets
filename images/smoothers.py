import numpy as np
import pandas as pd
from PIL import Image
import statistics as stats
from skimage import measure
from ..level_sets.utils import find_neighbours


def load_image(file):
    img = np.array(Image.open(file).convert('L'))
    return img


def levelset_median_smoother(image, pmax=3, nmax=2, pmin=1, method='median', connectivity=8,  # noqa: C901
                             keep_else=True, parallel=False):
    """
    Function to apply adaptive median smoother using level sets.

    Args:
        image: The greyscale images to be smooted
        pmax: Maximum size of level sets to be smoothed
        nmax: Maximum neighbourhood size
    Returns:
        Smoothed image
    """
    assert method in ['median', 'closest'], f"'{method}' is an invalid sampling method, Options are: ['median', 'closest']"

    img = image.copy()
    ref_img = img.copy()

    N, M = img.shape
    # This function does not pick up a connected set with a value of 0
    level_sets = measure.label(img + 1, connectivity=1)
    set_sizes = pd.value_counts(level_sets.flatten())

    # for p in np.arange(pmin,pmax+1):

    for p in np.arange(pmax, pmin - 1, -1):
        req_set_sizes = set_sizes.iloc[(set_sizes == p).values].index
        for set_label in req_set_sizes:
            n = 1
            while n < nmax:
                set_index = list(map(tuple, np.asarray(np.where(level_sets == set_label)).T.tolist()))
                set_value = img[set_index[0]]
                neighbours = find_neighbours(set_index, n, N, M, connectivity)
                if method == 'median':
                    neighbour_values = [ref_img[idx] for idx in neighbours]
                    median = stats.median(neighbour_values)
                    replace_value = stats.median(neighbour_values)
                elif method == 'closest':
                    neighbour_values = [ref_img[idx] for idx in neighbours if idx not in set_index]
                    closest = abs(neighbour_values - set_value)
                    median = stats.median(neighbour_values)
                    replace_value = neighbour_values[closest.argmin()]

                minimum = min(neighbour_values)
                maximum = max(neighbour_values)

                if minimum < median and median < maximum:
                    if not (minimum < set_value and set_value < maximum):
                        # for i in range(len(set_index)):
                        #     img[set_index[i]] = median
                        for idx in set_index:
                            img[idx] = replace_value

                    n = nmax
                else:
                    # NOT SURE IF THIS IS NEEDED
                    if keep_else:
                        for idx in set_index:
                            img[idx] = replace_value
                    n += 1
            if parallel is False:
                ref_img = img.copy()
    return img


def adaptive_median_filter(image, p=3):
    img = image.copy()
    N, M = img.shape
    for i in range(N):
        for j in range(M):
            k = 1
            ind = 0
            while k < p:
                set_value = img[i, j]
                neighbour_values = img[max(0, i - k):min(N, i + k), max(0, j - k):min(M, j + k)]
                minimum = min(neighbour_values.flatten())
                median = stats.median(neighbour_values.flatten())
                maximum = max(neighbour_values.flatten())
                if minimum < median and median < maximum:
                    if minimum < set_value and set_value < maximum:
                        img[i, j] = set_value
                        k = p
                        ind = 1
                    else:
                        img[i, j] = median
                        k = p
                        ind = 1

                #     if not (minimum < set_value and set_value < maximum):
                #         img[i, j] = median
                #     k = p
                #     ind = 1
                else:
                    # img[i, j] = median
                    k += 1
            if ind == 0:
                neighbour_values = img[max(0, i - k):min(N, i + k), max(0, j - k):min(N, j + k)]
                median = stats.median(neighbour_values.flatten())
                img[i, j] = median
    return img


def _padding(img, pad):

    padded_img = np.zeros((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad))
    padded_img[pad:-pad, pad:-pad] = img
    return padded_img


def adaptive_median_smoother(img, s=3, sMax=7):

    if len(img.shape) == 3:
        raise Exception("Single channel image only")

    H, W = img.shape
    a = sMax // 2
    padded_img = _padding(img, a)

    f_img = np.zeros(padded_img.shape)

    for i in range(a, H + a + 1):
        for j in range(a, W + a + 1):
            value = _lvl_a(padded_img, i, j, s, sMax)
            f_img[i, j] = value

    return f_img[a:-a, a:-a]


def _lvl_a(mat, x, y, s, sMax):

    window = mat[x - (s // 2):x + (s // 2) + 1, y - (s // 2):y + (s // 2) + 1]
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if A1 > 0 and A2 < 0:
        return _lvl_b(window)
    else:
        s += 2
        if s <= sMax:
            return _lvl_a(mat, x, y, s, sMax)
        else:
            return Zmed


def _lvl_b(window):

    h, w = window.shape
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    Zxy = window[h // 2, w // 2]
    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if B1 > 0 and B2 < 0:
        return Zxy
    else:
        return Zmed
