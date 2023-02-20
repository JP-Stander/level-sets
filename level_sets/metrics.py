import cv2
import numpy as np
from skimage import measure
from skimage.feature import greycomatrix, greycoprops

#Gray-level co-occurence matrices
class GLCM:
    def __init__(self, image, bins=None):
        if not bins:
            bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
        inds = np.digitize(image, bins)
        
        max_value = inds.max()+1
        matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)
        self.matrix_coocurrence = matrix_coocurrence
    # GLCM properties
    def contrast_feature(self):
        contrast = greycoprops(self.matrix_coocurrence, 'contrast')
        return contrast
    
    def dissimilarity_feature(self):
        dissimilarity = greycoprops(self.matrix_coocurrence, 'dissimilarity')    
        return dissimilarity
    
    def homogeneity_feature(self):
        homogeneity = greycoprops(self.matrix_coocurrence, 'homogeneity')
        return homogeneity
    
    def energy_feature(self):
        energy = greycoprops(self.matrix_coocurrence, 'energy')
        return energy
    
    def correlation_feature(self):
        correlation = greycoprops(self.matrix_coocurrence, 'correlation')
        return correlation
    
    
    def entropy_feature(self):
        entropy = greycoprops(self.matrix_coocurrence, 'entropy')
        return entropy

#To view points
#plt.scatter(locs[:,1],locs[:,0])
def _pixels_to_points(pixels):
    '''Returns coordinates of corner of pixels of each border pixel'''
    pixels = (pixels != 0).astype(int)
    pixels = np.pad(pixels, pad_width=1, mode='constant', constant_values=0)
    locs = np.where(pixels==1)
    locs = np.concatenate((pixels.shape[0]-locs[0].reshape(-1,1),locs[1].reshape(-1,1)),axis=1)
    
    points=[[],[]]
    for i in range(locs.shape[0]):
        neighbourhood = pixels[locs[0][0]-1:locs[0][0]+2,locs[0][1]-1:locs[0][1]+2]
        if sum((filter*neighbourhood==1).flatten())==4:
            continue
        points[0].append(locs[i,0]+0.5)
        points[0].append(locs[i,0]-0.5)
        points[0].append(locs[i,0]+0.5)
        points[0].append(locs[i,0]-0.5)
            
        points[1].append(locs[i,1]+0.5)
        points[1].append(locs[i,1]+0.5)
        points[1].append(locs[i,1]-0.5)
        points[1].append(locs[i,1]-0.5)
    
    return np.array(points).transpose()
    
def max_distance(pulses, ignore_0=True):
    pulses = np.array(pulses)
    level_sets = measure.label(pulses,connectivity=1)
    unique_level_sets = np.unique(level_sets).tolist()
    if ignore_0:
        level_set_0 = level_sets[pulses==0]
        unique_level_sets.remove(level_set_0[0])
    distances = {}
    for level_set in unique_level_sets:
        pulse = pulses * (pulses==level_set)
        points = _pixels_to_points(pulse)
        dist = 0
        for c1 in points:
            for c2 in points:
                dist1 = sum((c1-c2)**2)
                dist = dist1 if dist1 > dist else dist
        distances[level_set] = dist
    return distances
                
def _find_perimeter(pulse): 
    perimeter = 0
    pulse = (pulse != 0).astype(int)
    #Add a buffer of zeros so that the filter can be used on the edge of the pulse
    pulse = np.pad(pulse, pad_width=1, mode='constant', constant_values=0)
    r,c = pulse.shape
    #Filter to find 1-connectivity neighbhours
    filter = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]])
    #Because matrix consists of 1s and 0s it can be used as True of False when using if
    for i in range(0, r): 
        for j in range(0, c): 
            if (pulse[i][j]): 
              #The perimeter is 4 minus the number of 1-connectivity neighbhours
               perimeter += 4 - sum(sum((pulse[i-1:i+2,j-1:j+2]*filter)))
    return perimeter
                   
def perimeter(pulses, ignore_0=True):
    pulses = np.array(pulses)
    level_sets = measure.label(pulses,connectivity=1)
    unique_level_sets = np.unique(level_sets).tolist()
    if ignore_0:
        level_set_0 = level_sets[pulses==0]
        unique_level_sets.remove(level_set_0[0])
    perimeters = {}
    for level_set in unique_level_sets:
        pulse = pulses * (pulses==level_set)
        perimeters[level_set] = _find_perimeter(pulse)
    
    return perimeters

def _get_major_axis(pulse):
    points = _pixels_to_points(pulse)
    points = np.unique(points*10,axis=0)
    points = np.array(points,dtype=np.int32)
    res = cv2.minEnclosingCircle(points)
    return (res[1]*2/10)**2

def major_axis_length(pulses):
    pulses = np.array(pulses)
    level_sets = measure.label(pulses,connectivity=1)
    axes = {}
    for level_set in [a for a in np.unique(level_sets) if a > 0]:
        pulse = pulses * (pulses==level_set)
        points = _pixels_to_points(pulse)
        axes[level_set] = _get_major_axis(points)
    return axes

def major_axis(pulses, ignore_0=True, return_length=True, 
               return_coordinates=False, return_angle=False):

    pulses = np.array(pulses)
    level_sets = measure.label(pulses,connectivity=1)
    
    angles = {}
    coordinates = {}
    lengths = {}
    for level_set in [a for a in np.unique(level_sets) if a > 0]:
        pulse = pulses * (pulses==level_set)
        points = _pixels_to_points(pulse)
        dist = 0
        for c1 in points:
            for c2 in points:
                dist1 = np.sqrt(sum((c1-c2)**2))
                if dist1 > dist:
                    dist = dist1
                    major_axis_coordinates = [c1,c2]
        incline = (major_axis_coordinates[0][1]-major_axis_coordinates[1][1])
        incline /= (major_axis_coordinates[0][0]-major_axis_coordinates[1][0])
        
        angles[level_set] = np.arctan(incline)*180/np.pi
        coordinates[level_set] = major_axis_coordinates
        lengths[level_set] = dist
        
        output = {}
        if return_length:
            output['lenghts'] = lengths
        if return_coordinates:
            output['coordinates'] = coordinates
        if return_angle:
            output['angles'] = angles
        
    return output
    
cv2.fitEllipse()
def convex_hull(pulses):
    pulses = np.array(pulses)
    level_sets = measure.label(pulses,connectivity=1)
    hulls = {}
    for level_set in [a for a in np.unique(level_sets) if a > 0]:
        pulse = pulses * (pulses==level_set)
        cv2.convexHull(pulse, False)

#Ellipse idea
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