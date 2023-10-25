# %%
import os
import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from config import classes, images_loc, img_size, fs_connectivity, fs_delta, sets_feature_names, trim
current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from level_sets.metrics import get_metrics
from level_sets.utils import get_fuzzy_sets, get_level_sets


metrics = ['angle', 'area', 'compactness', 'elongation', 'width_to_height', 'bbox_area', 'aspect_ratio', 'extent', 'orientation', 'convexity']
folders_loc = "../../dtd/images"
folders = os.listdir(folders_loc)
# %%
results_dict = {folder: {metric: [] for metric in metrics} for folder in folders}
for folder in folders:
    images = os.listdir(os.path.join(folders_loc, folder))
    for image in tqdm(images):
        img = load_image(os.path.join(folders_loc, folder, image), [img_size, img_size])
        level_sets = get_fuzzy_sets(img, 10, 8)
        uni_level_sets = pd.unique(level_sets.flatten())
        for i, ls in enumerate(uni_level_sets):
            subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
            level_set = np.array(level_sets == ls)
            metric_results = get_metrics(level_set.astype(int), img_size=[img_size, img_size])
            for metric, value in metric_results.items():
                results_dict[folder][metric].append(value)

df = pd.DataFrame.from_dict(
    {(i,j): results_dict[i][j] 
    for i in results_dict.keys() 
    for j in results_dict[i].keys()},
    orient='index'
)
df.to_csv("results/metrics.csv")
# %%

# df = pd.read_csv(".csv")

# Extract unique folders and metrics from the MultiIndex
folders = [x[0] for x in df.index.unique()]
metrics = [x[1] for x in df.index.unique()]

# Ensure that the folder and metric lists contain unique values
folders = list(set(folders))
metrics = list(set(metrics))

# Store each KS matrix in a dictionary with metric names as keys
ks_matrices = {}

for metric in metrics:
    # Initialize an empty matrix for the current metric
    ks_matrix = pd.DataFrame(index=folders, columns=folders)
    
    for folder1 in folders:
        for folder2 in folders:
            # Get the data lists for the current pair of folders and metric
            data1 = df.xs((folder1, metric)).dropna()
            data2 = df.xs((folder2, metric)).dropna()
            
            # Compute the KS statistic (we only take the statistic, not the p-value)
            ks_stat, _ = ks_2samp(data1, data2)
            
            # Store the KS statistic in the matrix
            ks_matrix.at[folder1, folder2] = ks_stat
            ks_matrix.at[folder2, folder1] = ks_stat
    
    # Save the matrix for the current metric
    ks_matrices[metric] = ks_matrix

# %%
colormaps = ["Reds", "Greens", "Blues", "Oranges", "Purples", "Greys", "YlOrBr", "BuPu", "GnBu", "YlGn"]

for idx, (metric, matrix) in enumerate(ks_matrices.items()):
    plt.figure()
    sns.heatmap(matrix.astype(float), cmap=colormaps[idx], vmin=0, vmax=0.5)
    plt.title(f'KS Matrix Heatmap for {metric}')
    plt.savefig(f"../../paper3_results/ks_stats/{metric}_ks_plot.png")
    
# %%
