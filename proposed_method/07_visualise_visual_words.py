# %%
import numpy as np
import pickle
import joblib
from matplotlib import pyplot as plt
from config import experiment_loc, num_clusters, classes

# %%
# Load the features dictionary
with open(f"{experiment_loc}/feats_full.pkl", 'rb') as f:
    loaded_feats = pickle.load(f)

for key in loaded_feats:
    loaded_feats[key] = [np.array(arr) for arr in loaded_feats[key]]

kmeans = joblib.load(f"{experiment_loc}/kmeans.pkl")
model = joblib.load(f"{experiment_loc}/logistic_regression_model.pkl")
# %%

flattened_feats_full = [arr[:,:5] for key in loaded_feats for arr in loaded_feats[key]]
all_descriptors_full = np.vstack(flattened_feats_full)
all_descriptors = all_descriptors_full[:,:-1]
# %%

centroids = kmeans.cluster_centers_
all_descriptors = np.asarray(all_descriptors, dtype=np.float64)
centroids = np.asarray(centroids, dtype=np.float64)
distances = np.linalg.norm(all_descriptors[:, np.newaxis] - centroids, axis=2)

# Find the index of the minimum distance for each centroid
closest_indices = np.argmin(distances, axis=0)

lr_coefs = model.coef_[:, :centroids.shape[0]]
sorted_indices = np.argsort(lr_coefs.ravel())
smallest_indices = sorted_indices[:3].tolist()
largest_indices = sorted_indices[-3:].tolist()
# %%
for clas in classes:
    print(f"{clas} is class {classes.index(clas)}")
for index in largest_indices+smallest_indices:
    closest_word = all_descriptors_full[closest_indices[index], :]
    pixel_idx = [a for a in eval(closest_word[5,])] if ")," in closest_word[5,] else [eval(closest_word[5,])]
    height = max([a[0] for a in pixel_idx]) +2
    width = max([a[1] for a in pixel_idx]) +2
    image = np.zeros((height, width), dtype=np.uint8)

    # 2. Set pixel values to 1 at the provided indices
    for idx in pixel_idx:
        image[idx] = 1

    # 3. Cut the image
    # Get the bounding box around the 1s
    rows = [idx[0] for idx in pixel_idx]
    cols = [idx[1] for idx in pixel_idx]

    min_row, max_row = max(0, min(rows) - 1), min(height, max(rows) + 2)
    min_col, max_col = max(0, min(cols) - 1), min(width, max(cols) + 2)

    cropped_image = image[min_row:max_row, min_col:max_col]
    cropped_image = cropped_image*closest_word[4,]
    if closest_word[4,] < 0.5:
        cropped_image[cropped_image==0]=1
    
    plt.figure()
    plt.imshow(cropped_image, "gray")
    plt.title(f"Coefficient value: {lr_coefs[:,index]}")
    plt.show()

# %%
