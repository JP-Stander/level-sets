import cv2
import numpy as np
from math import log
from scipy.ndimage import distance_transform_edt


def psnr(img, img_smoothed):
    # Peak signal to noise ratio

    if img.shape != img_smoothed.shape:
        raise Exception("Image should be the same size")
    n, m = img.shape
    max_t = 255 if img.flatten().max() > 1 else 1
    MSE = ((img - img_smoothed) ** 2).flatten().sum() / (n * m)
    PSNR = 20 * log(max_t, 10) - 10 * log(MSE, 10)
    return PSNR


def fom(img_original, img_smooth, lower=20, upper=50):
    # Pratt's Figure of Merit

    # Determine where edges are in the original and gold standard
    edges_smooth = cv2.Canny(np.uint8(img_smooth), lower, upper) > 0
    edges_original = cv2.Canny(np.uint8(img_original), lower, upper) > 0

    # Calculate the distance from each element to the closest edge element in the original image
    dist = distance_transform_edt(np.invert(edges_original))

    # Calculate the number of edge elements in the original and smoothed image
    N_smooth = (edges_smooth).sum()
    N_original = (edges_original).sum()

    # Initialize fom quantity
    fom = 0

    # Dimensions of image
    N, M = img_original.shape

    # Calculating the summation part of the FOM metric
    for i in range(N):
        for j in range(M):
            if edges_smooth[i, j]:
                fom += 1.0 / (1.0 + 1 / 9 * dist[i, j] ** 2)

    # Divide by maximum number of edges
    fom /= max(N_smooth, N_original)

    return fom
