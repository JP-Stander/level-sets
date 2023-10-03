import cv2
import numpy as np
from skimage import measure
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops, perimeter, label
# from skimage.feature import greycomatrix, greycoprops
from .utils import cut_level_set
# metrics for level-sets


def compactness(level_set):
    """_summary_

    Args:
        level_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    # This functions calculates the compactness of all the pulses in an image, it takes as
    # input the images with all the pulses and call the rest of the functions by its self
    # so it is not neccesary to call any other functions

    # Making sure the image is a numpy array
    level_set = np.array(level_set)
    level_set = cut_level_set(level_set)
    perimtr = perimeter(level_set)
    area = sum(sum(level_set))
    compactness = 2 * np.sqrt(np.pi * area) / perimtr

    return compactness

def elongation(level_set):
    # This functions calculates the elongatin of all the pulses in an image, it takes as
    # input the images with all the pulses and call the rest of the functions by its self
    # so it is not neccesary to call any other functions

    # Making sure the level-set is a numpy array
    level_set = np.array(level_set)

    level_set = cut_level_set(level_set)
    lenght_2 = get_major(level_set)
    area = sum(sum(level_set))
    elongation = area / lenght_2

    return elongation

def area(level_set, img_size):
    level_set = np.array(level_set)
    level_set = cut_level_set(level_set)
    area = sum(sum(level_set))
    return area/(img_size[0]*img_size[1])

def perimeter(level_set):

    perimeter = 0
    level_set = np.array(level_set)
    level_set = cut_level_set(level_set)
    level_set = (level_set != 0).astype(int)
    # Add a buffer of zeros so that the filter can be used on the edge of the pulse
    level_set = np.pad(level_set, pad_width=1, mode='constant', constant_values=0)
    r, c = level_set.shape
    # Filter to find 1-connectivity neighbhours
    filter = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    # Because matrix consists of 1s and 0s it can be used as True of False when using if
    for i in range(0, r):
        for j in range(0, c):
            if (level_set[i][j]):
                # The perimeter is 4 minus the number of 1-connectivity neighbhours
                perimeter += 4 - sum(sum((level_set[i - 1:i + 2, j - 1:j + 2] * filter)))

    return perimeter

def width_to_height(level_set):
    level_set = np.array(level_set)
    level_set = cut_level_set(level_set)
    height, width = level_set.shape

    return width/height

def max_distance(level_set, return_coordinates=False):
    level_set = np.array(level_set)
    level_set = cut_level_set(level_set)
    points = _pixels_to_points(level_set)
    max_dist = 0
    c1_max = None
    c2_max = None
    for c1 in points:
        for c2 in points:
            dist = sum((c1 - c2) ** 2)
            if dist > max_dist:
                max_dist = dist
                c1_max = c1
                c2_max = c2
    if return_coordinates:
        return max_dist, [c1_max, c2_max]

    return max_dist

def get_angle(image):
    # Get the indices of the non-zero elements (i.e., 1s)
    y, x = np.where(image == 1)

    # Compute the centroid
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    # Compute the second central moments
    mu_20 = np.sum((x - x_bar) ** 2)
    mu_02 = np.sum((y - y_bar) ** 2)
    mu_11 = np.sum((x - x_bar) * (y - y_bar))

    # Calculate the orientation
    theta = 0.5 * np.arctan2(2 * mu_11, mu_20 - mu_02)

    # Convert to degrees
    theta_deg = np.degrees(theta)
    
    return theta_deg/180

def major_axis(level_set):
    level_set = np.array(level_set)
    level_set = cut_level_set(level_set)

    length, major_axis_coordinates = max_distance(level_set, return_coordinates=True)

    incline = (major_axis_coordinates[0][1] - major_axis_coordinates[1][1])
    incline /= (major_axis_coordinates[0][0] - major_axis_coordinates[1][0])

    angle = np.arctan(incline) * 180 / np.pi

    return length, angle, major_axis_coordinates

def _get_enclosing_circle(level_set):
    level_set = np.array(level_set)
    level_set = cut_level_set(level_set).astype(int)

    points = _pixels_to_points(level_set)
    points = np.unique(points * 10, axis=0)
    points = np.array(points, dtype=np.int32)
    enclosing_circle = cv2.minEnclosingCircle(points)

    return (enclosing_circle[1] * 2 / 10) ** 2

def _pixels_to_points(pixels):
    '''Returns coordinates of corner of pixels of each border pixel'''
    pixels = (pixels != 0).astype(int)
    pixels = np.pad(pixels, pad_width=1, mode='constant', constant_values=0)
    locs = np.where(pixels == 1)
    locs = np.concatenate((pixels.shape[0] - locs[0].reshape(-1, 1), locs[1].reshape(-1, 1)), axis=1)

    filter = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    points = [[], []]
    for i in range(locs.shape[0]):
        neighbourhood = pixels[
            locs[i][0] - 2:locs[i][0] + 2,
            locs[i][1] - 1:locs[i][1] + 2
        ]
        if sum((filter * neighbourhood == 1).flatten()) == 4:
            continue
        points[0].append(locs[i, 0] + 0.5)
        points[0].append(locs[i, 0] - 0.5)
        points[0].append(locs[i, 0] + 0.5)
        points[0].append(locs[i, 0] - 0.5)

        points[1].append(locs[i, 1] + 0.5)
        points[1].append(locs[i, 1] + 0.5)
        points[1].append(locs[i, 1] - 0.5)
        points[1].append(locs[i, 1] - 0.5)

    return np.array(points).transpose()

def _get_bbox_area(region):
    minr, minc, maxr, maxc = region.bbox
    return (maxr-minr)*(maxc-minc)

def _get_aspect_ratio(region):
    minr, minc, maxr, maxc = region.bbox
    return (maxc - minc) / (maxr - minr)

def _get_extent(region):
    minr, minc, maxr, maxc = region.bbox
    return region.area / ((maxr - minr) * (maxc - minc))

def _get_convexity(region, area):
    convex_image = convex_hull_image(region.image)
    convex_area = np.sum(convex_image)
    return area/convex_area

def get_metrics(data, indecis=False, metric_names = ["all"], img_size=[0,0]):
    if indecis is True:
        pixels = [a for a in eval(data.get("pixel_indices"))] if ")," in data.get("pixel_indices") else [eval(data.get("pixel_indices"))]
        img_size = max(max(a[0] for a in pixels), max(a[1] for a in pixels))
        level_set = np.zeros((img_size+1, img_size+1))
        rows, cols = zip(*pixels)
        level_set[rows, cols] = 1
        level_set=level_set.astype(int)
    else:
        level_set = data.astype(int)
    regions = regionprops(level_set)
    region = regions[0] # Since there will always be only 1 region

    metric_functions = {
        "angle": lambda: get_angle(level_set),
        "area": lambda: area(level_set, img_size),
        "compactness": lambda: compactness(level_set),
        "elongation": lambda: elongation(level_set),
        "width_to_height": lambda: width_to_height(level_set),
        "bbox_area": lambda: _get_bbox_area(region),
        "aspect_ratio": lambda: _get_aspect_ratio(region),
        "extent": lambda: _get_extent(region),
        "orientation": lambda: region.orientation,
        "convexity": lambda: _get_convexity(region, area(level_set, img_size))
    }

    metric_names = list(metric_functions.keys()) if "all" in metric_names else metric_names
    metrics = {name: metric_functions[name]() for name in metric_names}

    return metrics


# class GLCM:
#     # Gray-level co-occurence matrices
#     def __init__(self, image, bins=None):
#         if not bins:
#             bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
#         inds = np.digitize(image, bins)
#
#         max_value = inds.max() + 1
#         matrix_coocurrence = greycomatrix(inds,
#                                           [1],
#                                           [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
#                                           levels=max_value,
#                                           normed=False,
#                                           symmetric=False
#                                           )
#         self.matrix_coocurrence = matrix_coocurrence
#     # GLCM properties
#
#     def contrast_feature(self):
#         contrast = greycoprops(self.matrix_coocurrence, 'contrast')
#         return contrast
#
#     def dissimilarity_feature(self):
#         dissimilarity = greycoprops(self.matrix_coocurrence, 'dissimilarity')
#         return dissimilarity
#
#     def homogeneity_feature(self):
#         homogeneity = greycoprops(self.matrix_coocurrence, 'homogeneity')
#         return homogeneity
#
#     def energy_feature(self):
#         energy = greycoprops(self.matrix_coocurrence, 'energy')
#         return energy
#
#     def correlation_feature(self):
#         correlation = greycoprops(self.matrix_coocurrence, 'correlation')
#         return correlation
#
#     def entropy_feature(self):
#         entropy = greycoprops(self.matrix_coocurrence, 'entropy')
#         return entropy


# To view points
# plt.scatter(locs[:,1],locs[:,0])

def _get_major_axis(pulse):
    points = _pixels_to_points(pulse)
    points = np.unique(points * 10, axis=0)
    points = np.array(points, dtype=np.int32)
    res = cv2.minEnclosingCircle(points)

    return (res[1] * 2 / 10) ** 2

def major_axis_length(pulses):
    pulses = np.array(pulses)
    level_sets = measure.label(pulses, connectivity=1)
    axes = {}
    for level_set in [a for a in np.unique(level_sets) if a > 0]:
        pulse = pulses * (pulses == level_set)
        points = _pixels_to_points(pulse)
        axes[level_set] = _get_major_axis(points)
    return axes

# def major_axis(pulses, ignore_0=True, return_length=True,
#                return_coordinates=False, return_angle=False):

#     pulses = np.array(pulses)
#     level_sets = measure.label(pulses,connectivity=1)

#     angles = {}
#     coordinates = {}
#     lengths = {}
#     for level_set in [a for a in np.unique(level_sets) if a > 0]:
#         pulse = pulses * (pulses == level_set)
#         points = _pixels_to_points(pulse)
#         dist = 0
#         for c1 in points:
#             for c2 in points:
#                 dist1 = np.sqrt(sum((c1 - c2) ** 2))
#                 if dist1 > dist:
#                     dist = dist1
#                     major_axis_coordinates = [c1, c2]
#         incline = (major_axis_coordinates[0][1] - major_axis_coordinates[1][1])
#         incline /= (major_axis_coordinates[0][0] - major_axis_coordinates[1][0])

#         angles[level_set] = np.arctan(incline) * 180/np.pi
#         coordinates[level_set] = major_axis_coordinates
#         lengths[level_set] = dist

#         output = {}
#         if return_length:
#             output['lenghts'] = lengths
#         if return_coordinates:
#             output['coordinates'] = coordinates
#         if return_angle:
#             output['angles'] = angles

#     return output


def get_major(img):
    locs = np.where(img == 1)
    locs = np.concatenate((locs[0].reshape(-1, 1), locs[1].reshape(-1, 1)), axis=1)

    pnts = [[], []]
    for i in range(locs.shape[0]):
        pnts[0].append(locs[i, 0] + 0.5)
        pnts[0].append(locs[i, 0] - 0.5)
        pnts[0].append(locs[i, 0] + 0.5)
        pnts[0].append(locs[i, 0] - 0.5)

        pnts[1].append(locs[i, 1] + 0.5)
        pnts[1].append(locs[i, 1] + 0.5)
        pnts[1].append(locs[i, 1] - 0.5)
        pnts[1].append(locs[i, 1] - 0.5)

    pnts = np.array(pnts).transpose()
    pnts = np.unique(pnts * 10, axis=0)
    pnts = np.array(pnts, dtype=np.int32)
    res = cv2.minEnclosingCircle(pnts)

    return (res[1] * 2 / 10) ** 2


# cv2.fitEllipse()
# def convex_hull(pulses):
#     pulses = np.array(pulses)
#     level_sets = measure.label(pulses,connectivity=1)
#     hulls = {}
#     for level_set in [a for a in np.unique(level_sets) if a > 0]:
#         pulse = pulses * (pulses==level_set)
#         cv2.convexHull(pulse, False)

# Ellipse idea
# X = points[:,0].reshape(-1,1)
# Y = points[:,1].reshape(-1,1)
# A = np.hstack([X**2, X * Y, Y**2, X, Y])
# b = np.ones_like(X)
# x = np.linalg.lstsq(A, b)[0].squeeze()

# x_coord = np.linspace(-5,5,300)
# y_coord = np.linspace(-5,5,300)
# X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
# Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
# plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
# plt.scatter(points[:,0],points[:,1])
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
