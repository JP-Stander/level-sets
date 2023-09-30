# %% Imports
import os
import sys
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt

current_script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, "/".join(current_script_directory.split("/")[:-1]))
from images.utils import load_image
from level_sets.metrics import get_metrics
from level_sets.utils import get_fuzzy_sets, get_level_sets

# %%
folders = ['dotted', 'fibrous']
images_loc = "../../dtd/images"
images = []
for folder in folders:
    images += [f"{images_loc}/{folder}/" + file for file in os.listdir(f"{images_loc}/{folder}")]

# Initialize metrics dictionary
metrics = {folder: [] for folder in folders}

for image in tqdm(images[:5]+images[-5:]):
    img = load_image(image, [50,50])
    # level_sets = get_level_sets(img)
    level_sets = get_fuzzy_sets(img, 10, 8)
    uni_level_sets = pd.unique(level_sets.flatten())
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        level_set = np.array(level_sets == ls)
        metric_results = get_metrics(level_set.astype(int))
        folder = image.split("/")[-2]
        metrics[folder] += [metric_results]

# %% Plot distributions (optional)
df = pd.DataFrame([(folder, metric, value)
                for folder, metrics_list in metrics.items() 
                for metrics_dict in metrics_list 
                for metric, value in metrics_dict.items()],
            columns=["Folder", "Metric", "Value"]
            )
# Now, you can plot using seaborn
for metric in pd.unique(df['Metric']):
    plt.figure()
    for folder in folders:
        subset = df.loc[(df["Folder"] == folder) & (df["Metric"] == metric), "Value"]
        sns.kdeplot(subset, label=folder, fill=True)
    plt.title(f'Distribution of {metric} across classes')
    plt.tight_layout()
    plt.show()

# %% Mean and variance (optional
for folder, folder_metrics in metrics.items():
    for metric in pd.unique(df['Metric']):
        print(
            f"Mean(variance) for metric {metric} for texture {folder}: " +
            f"{df.loc[(df['Folder'] == folder) & (df['Metric'] == metric), 'Value'].mean()}" +
            f"({df.loc[(df['Folder'] == folder) & (df['Metric'] == metric), 'Value'].var()})"
            )

# %% KS-statistic
table_data = []

for metric_name in pd.unique(df['Metric']):
    values1 = df.loc[(df["Folder"] == folders[0]) & (df["Metric"] == metric_name), "Value"]
    values2 = df.loc[(df["Folder"] == folders[1]) & (df["Metric"] == metric_name), "Value"]
    ks_stat, p_value = ks_2samp(values1, values2)
    table_data.append((metric_name, ks_stat))

# Sort the table data by KS statistics in descending order
table_data.sort(key=lambda x: x[1], reverse=True)
print(pd.DataFrame(table_data))


# %%
