import numpy as np
import pandas as pd
from skimage import measure
from PIL import Image
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import distance_metrics

# Used for testing
# from skimage import io, color, img_as_ubyte
# img = io.imread(r'C:\Users\QXZ1DJT\Google Drive\JP PhD\Data\DB\data\natural_images\car\car_0000.jpg')

# gray = color.rgb2gray(img)
# image = img_as_ubyte(gray)
# io.imshow(image)

# img = [[0,0,1,0,0],
#        [0,0,1,0,0],
#        [0,0,1,1,1],
#        [2,2,0,0,0],
#        [0,2,0,0,0]]

def load_image(file, new_size=None):
        
    img = Image.open(file)
    if new_size:
        new_size = tuple(int(el) for el in new_size)
        img = img.resize(new_size)

    return np.array(img)
    
def get_level_sets(img, connectivity=1):
    level_sets = measure.label(img, 
                               background=-1, 
                               connectivity=connectivity
                               )
    return level_sets

def cut_level_set(img):
    img = np.array(img)
    #Making sure the image is binarized
    img = (img != 0).astype(int)
    n, m = img.shape
    #Getting the first and last non zeros pixel over columns and rows
    min_r = max(np.where(img> 0)[0].min(), 0)
    max_r = min(np.where(img> 0)[0].max()+1, n)
    min_c = max(np.where(img> 0)[1].min(), 0)
    max_c = min(np.where(img> 0)[1].max()+1, m)
    
    #Croppping the image
    img = img[min_r:max_r, min_c:max_c]
    
    return img

def number_neighbours(c, nmax, N, M, connectivity):
    neig_num = 0
    neigs = np.zeros((N,M))
    og_c = c.copy()
    for i in range(nmax):
        neig_num += 1
        for j in range(len(c)):
            if connectivity==4:
                i1, i2 = c[j]
                if i2-1 >= 0:
                    if not neigs[i1, i2-1]:
                        neigs[i1, i2-1] = neig_num
                        c += [(i1, i2-1)]
                if i2+1 < M:
                    if not neigs[i1, i2+1]:
                        neigs[i1, i2+1] = neig_num
                        c += [(i1, i2+1)]
                if i1-1 >= 0:
                    if not neigs[i1-1, i2]:
                        neigs[i1-1, i2] = neig_num
                        c += [(i1-1, i2)]
                if i1+1 < N:
                    if not neigs[i1+1, i2]:
                        neigs[i1+1, i2] = neig_num
                        c += [(i1+1, i2)]
    
            elif connectivity==8:
                i1, i2 = c[j]
                for y_chng in [-1,0,1]:
                    for x_chng in [-1,0,1]:
                        if i1+y_chng >= 0 and i1+y_chng < M:
                            if i2+x_chng >= 0 and i2+x_chng < N:
                                if not neigs[i1+y_chng, i2+x_chng]:
                                    neigs[i1+y_chng, i2+x_chng] = neig_num
                                    c += [(i1+y_chng, i2+x_chng)]
    
    for loc in og_c:
        neigs[loc] = 0
        
    return neigs

def spatio_environ_dependence(point_a, point_b, dist_type_a='l2', dist_type_b='l1', alpha=0.5):
    assert dist_type_a in distance_metrics().keys(), f"{dist_type_a} is an invalid distance type for argument dist_type_a. The options are\n{list(distance_metrics().keys())}"
    assert dist_type_b in distance_metrics().keys(), f"{dist_type_b} is an invalid distance type for argument dist_type_b. The options are\n{list(distance_metrics().keys())}"

    u_s = _dist(point_a[1:3], point_b[1:3], dist_type_a)
    u_e = _dist(point_a[3:], point_b[3:], dist_type_b)
    
    m = np.exp(-(alpha*u_e + (1-alpha)*u_s))
    return m
    
def _dist(point_a, point_b, dist_type):
    point_a = np.array(point_a).reshape(1,- 1)
    point_b = np.array(point_b).reshape(1,- 1)
    distance = distance_metrics()[dist_type](point_a, point_b)
    return distance

# This function is required for the LULU median smoother function
def find_neighbours(c, nmax, N, M, connectivity=4):
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
            if connectivity==4:
                i1, i2 = c[j]
                if i2-1 >= 0:
                    w.append((i1, i2-1))
                if i2+1 < M:
                    w.append((i1, i2+1))
                if i1-1 >= 0:
                    w.append((i1-1, i2))
                if i1+1 < N:
                    w.append((i1+1, i2))
            elif connectivity==8:
                i1, i2 = c[j]
                for y_chng in [-1,0,1]:
                    for x_chng in [-1,0,1]:
                        if i1+y_chng >= 0 and i1+y_chng < M:
                            if i2+x_chng >= 0 and i2+x_chng < N:
                                w.append((i1+y_chng, i2+x_chng))
            
        
        c = [a for a in pd.unique(c+w)]
    return c
                


