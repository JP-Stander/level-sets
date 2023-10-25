# %% Imports
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

# %%
images_loc = "../../dtd/images"
classes = ["striped", "honeycombed"]
images = []
for clas in classes:
    images += [f"{images_loc}/{clas}/" + file for file in os.listdir(f"{images_loc}/{clas}")]

# Initialize metrics dictionary
metrics = {clas: [] for clas in classes}

for image in tqdm(images):
    img = load_image(image, [img_size, img_size], trim=trim)
    # level_sets = get_level_sets(img)
    level_sets = get_fuzzy_sets(img, 35, fs_connectivity)
    uni_level_sets = pd.unique(level_sets.flatten())
    for i, ls in enumerate(uni_level_sets):
        subset = list(map(tuple, np.asarray(np.where(level_sets == ls)).T.tolist()))
        if len(subset) <= 1:
            continue
        level_set = np.array(level_sets == ls)
        metric_results = get_metrics(level_set.astype(int), img_size=[img_size, img_size], metric_names=["compactness", "elongation"])
        clas = image.split("/")[-2]
        metrics[clas] += [metric_results]

# %% Plot distributions (optional)
df = pd.DataFrame([(clas, metric, value)
                for clas, metrics_list in metrics.items() 
                for metrics_dict in metrics_list 
                for metric, value in metrics_dict.items()],
            columns=["Class", "Metric", "Value"]
            )

#%%
for metric_name in pd.unique(df["Metric"]):
    values1 = df.loc[(df["Class"] == classes[0]) & (df["Metric"] == metric_name), "Value"]
    values2 = df.loc[(df["Class"] == classes[1]) & (df["Metric"] == metric_name), "Value"]
    skewness1 = skew(values1)
    kurtosis1 = kurtosis(values1, fisher=True)
    skewness2 = skew(values2)
    kurtosis2 = kurtosis(values2, fisher=True)
    print(metric_name)
    print(f"Skewness of class 1: {skewness1:.2f}")
    print(f"Skewness of class 2: {skewness2:.2f}")
    print(f"Kurtosis of class 1: {kurtosis1:.2f}")
    print(f"Kurtosis of class 2: {kurtosis2:.2f}")
# %%

metric = "compactness"
data = df[df['Metric'] == metric]

plt.figure()
sns.kdeplot(data=data, x='Value', hue='Class', common_norm=False, legend=False)

# Add labels and a title
plt.xlabel(metric.capitalize())
plt.ylabel('Density')
# Show the plot
plt.savefig(f"../../paper3_results/honeycomb_v_striped_{metric}.png")
# %%
from scipy.stats import gaussian_kde

data = df[df['Metric'] == "compactness"]
data1 = data[data["Class"]=="honeycombed"]["Value"]
data2 = data[data["Class"]=="striped"]["Value"]
kde1 = gaussian_kde(data1)
kde2 = gaussian_kde(data2)

# Evaluate the PDFs on a grid
x = np.linspace(min(np.min(data1), np.min(data2)), max(np.max(data1), np.max(data2)), 1000)
pdf1 = kde1(x)
pdf2 = kde2(x)

# Calculate the intersection and union
intersection = np.minimum(pdf1, pdf2)
union_ = np.maximum(pdf1, pdf2)
intersection_area = np.trapz(intersection, x)
union_area = np.trapz(union_, x)
# %%

plt.figure(figsize=(10, 6))
plt.plot(x, pdf1, label='Honeycomb', color='blue', alpha=0.5)
plt.plot(x, pdf2, label='Striped', color='green', alpha=0.5)
plt.fill_between(x, 0, intersection, color='red', alpha=0.5, label='Intersection')
plt.plot(x, union_, label='Union', color='purple', linestyle='--')
plt.legend()
plt.show()

print(intersection_area/union_area)
# %%
